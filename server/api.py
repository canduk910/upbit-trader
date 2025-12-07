from fastapi import FastAPI, HTTPException # web framework
from contextlib import asynccontextmanager # for lifespan management
from pydantic import BaseModel # data validation, settings management
from typing import Any, Dict, List, Optional # type hints
import threading
import time
import os

# 시간대 설정
os.environ.setdefault('TZ', 'Asia/Seoul')
try:
    time.tzset()
except Exception:
    pass

# 외부라이브러리 임포트
import redis # Redis 클라이언트
import json
from concurrent.futures import ThreadPoolExecutor, as_completed # for parallel prefetching

# 내부API 모듈 임포트
from server import config               # 런타임 설정 관리
from server.upbit_api import UpbitAPI   # 업비트 API 연동
from server.logger import log           # 로깅 설정
from server.history import history_store
from server.history import order_history_store
from server.history import ai_history_store
from server.ws_listener_base import (
    load_ws_stats,
    summarize_ws_stats,
    read_exec_history,
)
from server.ws_listener_private import PrivateWebsocketListener
from server.ws_listener_public import PublicWebsocketlistener

# Token Bucket 구현 for prefetch
# 간단한 토큰 버킷(rate limiter) 구현
# rate: 초당 토큰 생성 속도
# capacity: 버킷 최대 토큰 수
# consume(tokens, timeout): 지정된 토큰 수를 소비 시도, 타임아웃 내에 성공 여부 반환
# 스레드 안전 구현
# 사용 예시:
# tb = TokenBucket(rate=5, capacity=10)  # 초당 5토큰, 최대 10토큰
# if tb.consume(tokens=1, timeout=2.0):
#     print("Token acquired")
# else:
#     print("Failed to acquire token within timeout")
class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = float(rate)         # 토큰 생성 속도 (초당)
        self.capacity = float(capacity) # 버킷 최대 용량
        self._tokens = float(capacity)  # 현재 토큰 수
        self._last = time.time()        # 마지막 토큰 갱신 시각
        self._lock = threading.Lock()   # 스레드 안전을 위한 락

    # 토큰 소비 메서드
    # tokens: 소비할 토큰 수
    # timeout: 최대 대기 시간 (초)
    # 반환값: 성공 시 True, 실패 시 False
    def consume(self, tokens: float = 1.0, timeout: float = 5.0) -> bool:
        end = time.time() + float(timeout)
        while time.time() < end:
            with self._lock:
                now = time.time()
                elapsed = max(0.0, now - self._last)
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
            time.sleep(0.01)
        return False


# 세마포어 및 토큰 버킷 초기화 (스케줄러 시작 시)
_prefetch_token_bucket: Optional[TokenBucket] = None
_prefetch_semaphore: Optional[threading.BoundedSemaphore] = None

# 기본 klines 캐시 TTL (초)
# 환경변수 'KLINES_CACHE_TTL'로 설정 가능, 기본값 600초
# 예: os.environ['KLINES_CACHE_TTL'] = '600'
# (기본값 600초로 설정하여 중복 Upbit 요청 감소, 실시간성은 다소 희생, 값이 클수록 캐시 지속시간 증가)
_KLINES_CACHE_TTL = int(os.getenv('KLINES_CACHE_TTL', str(600)))  # default 600s

# Balances cache TTL (seconds)
_BALANCES_CACHE_TTL = int(os.getenv('BALANCES_CACHE_TTL', '15'))

# FastAPI 앱 생성 및 수명 주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI): # 수명 주기 관리
    # start prefetch scheduler on startup
    try:
        # read interval from config, default 30s
        # if invalid, fallback to 30s
        # prefetch interval 설정
        # 기본값 30초
        # 환경변수나 config.json의 'prefetch_interval_sec' 키로 설정 가능
        interval = int(config._config.get('prefetch_interval_sec', 30))
    except Exception:
        interval = 30

    # 스케쥴러 시작
    start_prefetch_scheduler(interval=interval)
    try:
        start_ws_listener()
        start_ticker_listener()
    except Exception as exc:
        log.warning(f'Failed to start websocket listener on startup: {exc}')
    try:
        yield
    finally:
        # 스케쥴러 중지
        stop_prefetch_scheduler()
        stop_ws_listener()
        stop_ticker_listener()

app = FastAPI(title="Upbit Trader Runtime API", lifespan=lifespan) # FastAPI 앱 생성

# 데이터 모델 정의
class ConfigPayload(BaseModel):
    config: Dict[str, Any]

# 배치 klines 요청 모델
class KlinesBatchRequest(BaseModel):
    tickers: List[str]
    timeframe: Optional[str] = 'minute15'
    count: Optional[int] = 100

# 포지션 모델 정의
class Position(BaseModel):
    ticker: str
    amount: float
    avg_price: float
    current_price: float
    pnl: float

# --- Background Prefetch Scheduler 및 캐시 관리 ---
# 전역 상태 변수들
# Prefetch 스레드 및 제어 변수
# Simple in-memory watcher state (explicit typing to satisfy linters)
_watcher: Dict[str, Any] = {
    'running': False,
    'thread': None,
    'stop_event': None,
}

# 캐시 딕셔너리
# Simple in-memory cache for batch klines: { key: (timestamp, data) }
_klines_cache: Dict[str, Any] = {}


# 레디스(Redis) 클라이언트 초기화
# Redis setup (optional). Use local redis://localhost:6379 if REDIS_URL not set
# 환경변수 'REDIS_URL'로 Redis URL 설정 가능
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
_redis_client = None
try:
    # Redis 클라이언트 생성 및 연결 테스트
    _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    # ping 테스트
    _redis_client.ping()
    log.info(f'Redis cache connected: {REDIS_URL}')
except Exception as e:
    _redis_client = None
    log.warning(f'Redis not available ({REDIS_URL}): {e}. Falling back to in-memory cache')

# 캐시 설정 함수
# key: 캐시 키
# value: 캐시 값
# ttl: 캐시 만료 시간 (초)
# 기본 TTL은 _KLINES_CACHE_TTL 사용
# Redis 사용 가능 시 Redis에 저장, 아니면 메모리 내 딕셔너리에 저장
# 캐시 조회 시 만료 시간 확인
# 반환값: (타임스탬프, 값) 또는 None
def _cache_set(key: str, value: Any, ttl: int = _KLINES_CACHE_TTL):
    """Set cache in Redis if available, else in-memory."""
    now = time.time()
    if _redis_client:
        try:
            # store JSON string
            import json as _json
            payload = {'ts': now, 'value': value}
            _redis_client.setex(key, ttl, _json.dumps(payload)) # set with expiry
            return
        except Exception:
            pass
    _klines_cache[key] = (now, value) # store in-memory

# 캐시 조회 함수
# key: 캐시 키
# ttl: 캐시 만료 시간 (초)
# 기본 TTL은 _KLINES_CACHE_TTL 사용
# 반환값: (타임스탬프, 값) 또는 None
def _cache_get(key: str, ttl: int = _KLINES_CACHE_TTL):
    """Get cache value. Return (timestamp, value) or None."""
    if _redis_client:
        try:
            import json as _json
            s = _redis_client.get(key) # get JSON string
            if not s:
                return None
            obj = _json.loads(s) # parse JSON
            if time.time() - obj.get('ts', 0) > ttl:
                return None
            return (obj.get('ts'), obj.get('value')) # return (timestamp, value)
        except Exception:
            return None
    return _klines_cache.get(key) # return (timestamp, value) or None

# Websocket listener state
_ws_listener: Optional[PrivateWebsocketListener] = None
_ticker_listener: Optional[PublicWebsocketlistener] = None

def start_ws_listener() -> None:
    global _ws_listener
    if _ws_listener and _ws_listener._thread and _ws_listener._thread.is_alive():
        return
    _ws_listener = PrivateWebsocketListener(redis_client=_redis_client)
    _ws_listener.start()


def start_ticker_listener() -> None:
    global _ticker_listener
    if _ticker_listener and _ticker_listener._thread and _ticker_listener._thread.is_alive():
        return
    _ticker_listener = PublicWebsocketlistener(redis_client=_redis_client)
    _ticker_listener.start()


def stop_ws_listener() -> None:
    global _ws_listener
    if _ws_listener:
        _ws_listener.stop()


def stop_ticker_listener() -> None:
    global _ticker_listener
    if _ticker_listener:
        _ticker_listener.stop()

# Rate-limited Upbit klines fetcher
# ticker_local: 업비트 로컬 티커명 (예: KRW-BTC)
# timeframe: 캔들 시간대 (예: 'minute15')
# count: 조회할 캔들 개수
# 반환값: UpbitAPI.get_klines() 결과
# 전역 토큰 버킷 및 세마포어를 사용하여 호출 제한 및 동시성 제어
# 토큰 획득 및 세마포어 획득 시 타임아웃 처리
# 예외 발생 시 호출 실패
# 반환값: UpbitAPI.get_klines() 결과
def _rate_limited_get_klines(ticker_local: str, timeframe: str, count: int):
    # 전역 토큰 버킷 및 세마포어 사용
    global _prefetch_token_bucket, _prefetch_semaphore

    # 세마포어 및 토큰 획득 시도
    # 타임아웃은 config의 'prefetch_token_wait_timeout' 키로 설정 가능, 기본 10초
    # 실제 Upbit 호출 수행
    # 예외 발생 시 호출 실패
    acquired = False
    try:
        # 세마포어 획득 시도
        if _prefetch_semaphore is not None:
            # 세마포어 획득 (대기 시간 설정 기본값 10초)
            acquired = _prefetch_semaphore.acquire(timeout=10)
            if not acquired:
                raise RuntimeError('prefetch_semaphore_timeout')

        # 토큰 획득 시도
        if _prefetch_token_bucket is not None:
            # 토큰 대기 시간 설정
            # 기본값 10초 (즉시 호출 선호), config의 'prefetch_token_wait_timeout' 키로 설정 가능
            try:
                token_wait = float(config._config.get('prefetch_token_wait_timeout', 10.0))
            except Exception:
                token_wait = 10.0
            ok = _prefetch_token_bucket.consume(tokens=1.0, timeout=token_wait)
            if not ok:
                raise RuntimeError('rate_limited')
        # 업비트 공용 API 호출 수행
        # 반환값: UpbitAPI.get_klines() 결과
        # 참고: UpbitAPI 인스턴스는 전역 _upbit_public 사용
        # 이 인스턴스는 API 키를 사용하지 않음
        return _upbit_public.get_klines(ticker_local, timeframe, count=count)
    finally:
        if acquired and _prefetch_semaphore is not None:
            try:
                _prefetch_semaphore.release() # 세마포어 해제
            except Exception:
                pass


# 업비트 공용 API 인스턴스
_upbit_public = UpbitAPI()

# 업비트 인증(Private) API 인스턴스
try:
    access = getattr(config, 'UPBIT_ACCESS_KEY', None)
    secret = getattr(config, 'UPBIT_SECRET_KEY', None)
    if access and secret:
        _upbit_private = UpbitAPI(access_key=access, secret_key=secret)
        log.info('Upbit private API initialized with provided keys')
    else:
        _upbit_private = None
        log.warning('Upbit API keys not provided: private endpoints will be unavailable')
except Exception as e:
    _upbit_private = None
    log.warning(f'Failed to initialize Upbit private API: {e}')

# 스케쥴러 스레드 및 제어 변수
_prefetch_thread: Optional[threading.Thread] = None
_prefetch_stop = threading.Event()
_prefetch_index = 0

# 루프 함수 - 주기적으로 universe의 티커들에 대해 klines를 미리 가져와 캐시에 저장
# interval: 루프 주기 (초)
# 기본값 30초
def _prefetch_loop(interval: int = 30):
    log.info('Prefetch scheduler started')
    effective_interval = interval
    while not _prefetch_stop.is_set():
        try:
            # 설정 읽기
            cfg = config._config
            # 티커 유니버스
            universe = cfg.get('universe', [])
            # 실행 중 설정 체크 (런타임 config에서 재정의 허용)
            cfg_count = int(cfg.get('prefetch_count', 200))
            # Redis 미사용 시 보수적으로 설정, upper bound 허용
            if _redis_client is None:
                # When Redis is missing, be conservative but allow a configurable upper bound
                # Redis 미사용 시 최대값 설정, 기본 120, config의 'prefetch_no_redis_max_count' 키로 설정 가능
                try:
                    no_redis_max = int(cfg.get('prefetch_no_redis_max_count', 120))
                except Exception:
                    no_redis_max = 120
                count = min(cfg_count, no_redis_max)
                per_ticker_sleep = float(cfg.get('prefetch_sleep_sec', 1.0))
                log.info('Redis not available: using conservative prefetch settings (count=%s, sleep=%.2f)', count, per_ticker_sleep)
            else:
                count = cfg_count
                per_ticker_sleep = float(cfg.get('prefetch_sleep_sec', 0.2))

            # Redis 미사용 시 최소 간격 보장
            effective_interval = interval
            if _redis_client is None:
                effective_interval = max(interval, int(cfg.get('prefetch_min_interval_sec', 60)))
            if universe:
                # staggered batch processing to avoid bursts
                batch_size = int(cfg.get('prefetch_batch_size', 5))     # batch size per run
                parallelism = int(cfg.get('prefetch_parallelism', 3))   # max parallel fetches per batch
                global _prefetch_index
                n = len(universe)
                if n == 0:
                    pass
                else:
                    start = _prefetch_index % n # start index
                    end = start + batch_size    # end index (exclusive)
                    indices = list(range(start, min(end, n)))
                    # wrap-around if needed
                    if end > n:
                        indices += list(range(0, end - n))
                    tickers_to_process = [universe[i] for i in indices]
                    # advance index for next run
                    _prefetch_index = (start + len(tickers_to_process)) % max(n,1)

                    # helper to fetch and cache single ticker
                    def _prefetch_single(ticker_local: str):
                        try:
                            key_local = f"{ticker_local}|minute15|{count}"
                            cached_local = _cache_get(key_local)
                            if cached_local and (time.time() - cached_local[0]) < _KLINES_CACHE_TTL:
                                return (ticker_local, True, 'cached')
                            # use rate-limited fetch so prefetch respects global rate/concurrency limits
                            klines_local = _rate_limited_get_klines(ticker_local, 'minute15', count=count)
                            _cache_set(key_local, klines_local, ttl=_KLINES_CACHE_TTL)
                            return (ticker_local, True, 'fetched')
                        except Exception as exc:
                            log.error(f'Prefetch ticker error for {ticker_local}: {exc}')
                            return (ticker_local, False, str(exc))

                    # run in ThreadPoolExecutor with limited parallelism
                    with ThreadPoolExecutor(max_workers=min(parallelism, len(tickers_to_process))) as executor:
                        futures = {executor.submit(_prefetch_single, t): t for t in tickers_to_process}
                        for fut in as_completed(futures):
                            try:
                                ticker_res, ok, msg = fut.result()
                                log.debug(f'Prefetch result: {ticker_res} ok={ok} info={msg}')
                            except Exception as e:
                                log.error(f'Prefetch future error: {e}')
                    # after parallel batch, small pause to avoid immediate repeated calls
                    time.sleep(per_ticker_sleep)
        except Exception as e:
            log.error(f'Prefetch error: {e}')
        # wait using effective interval (recompute per loop)
        try:
            _prefetch_stop.wait(effective_interval)
        except UnboundLocalError:
            _prefetch_stop.wait(interval)
    log.info('Prefetch scheduler stopped')

# 스케쥴러 시작 함수
# interval: 루프 주기 (초)
# 기본값 30초
def start_prefetch_scheduler(interval: int = 30):
    global _prefetch_thread, _prefetch_stop
    if _prefetch_thread is not None and _prefetch_thread.is_alive():
        return
    _prefetch_stop.clear()
    # Redis 미사용 시 기본 간격 증가
    # Upbit 호출 부담 축소를 위한 조치
    # 기본 최소 60초
    if _redis_client is None:
        interval = max(interval, 60)
        log.info('Redis not connected: starting prefetch with interval %s seconds', interval)
    # Redis 미사용 시 기본 배치 크기 축소
    if _redis_client is None:
        try:
            # 배치사이즈 기본 3으로 축소, config에 없으면 설정
            if 'prefetch_batch_size' not in config._config:
                config._config['prefetch_batch_size'] = 3
        except Exception:
            pass
    # 프리페치 레이트 리미터 및 세마포어 초기화
    # 설정값 읽기 및 기본값 적용
    global _prefetch_token_bucket, _prefetch_semaphore # 전역 변수
    try:
        # 초당 5토큰
        rate = int(config._config.get('prefetch_rate_per_sec', 5))
    except Exception:
        rate = 5
    try:
        # 용량은 rate와 같게
        capacity = int(config._config.get('prefetch_rate_capacity', max(1, rate)))
    except Exception:
        capacity = max(1, rate)
    try:
        # 동시 3개
        max_concurrent = int(config._config.get('prefetch_max_concurrent', 3))
    except Exception:
        max_concurrent = 3
    try:
        # 토큰 버킷 및 세마포어 초기화
        _prefetch_token_bucket = TokenBucket(rate=float(rate), capacity=float(capacity))
        _prefetch_semaphore = threading.BoundedSemaphore(max_concurrent)
        log.info(f'Prefetch rate limiter initialized: rate={rate}/s, capacity={capacity}, max_concurrent={max_concurrent}')
    except Exception as e:
        _prefetch_token_bucket = None
        _prefetch_semaphore = None
        log.warning(f'Failed to initialize prefetch rate limiter: {e}')
    _prefetch_thread = threading.Thread(target=_prefetch_loop, args=(interval,), daemon=True)
    _prefetch_thread.start()

# 스케쥴러 중지 함수
# 스케쥴러 스레드 종료 대기 (최대 2초)
# 기본값 30초
def stop_prefetch_scheduler():
    global _prefetch_thread, _prefetch_stop
    _prefetch_stop.set()
    if _prefetch_thread is not None:
        _prefetch_thread.join(timeout=2)
    _prefetch_thread = None


@app.get("/health") # 헬스체크 엔드포인트
def health():
    return {"status": "ok"}


@app.get('/debug/status') # 디버그 상태 엔드포인트
def debug_status():
    """Return diagnostic info: pyupbit presence, redis connection, prefetch thread state, universe size."""
    try:
        import server.upbit_api as upbit_api
        has_pyupbit = bool(getattr(upbit_api, '_HAS_PYUPBIT', False))
    except Exception:
        has_pyupbit = False

    redis_up = False
    try:
        redis_up = _redis_client is not None
    except Exception:
        redis_up = False

    prefetch_running = False
    try:
        prefetch_running = (_prefetch_thread is not None and _prefetch_thread.is_alive())
    except Exception:
        prefetch_running = False

    universe_len = 0
    try:
        universe_len = len(config._config.get('universe', []))
    except Exception:
        universe_len = 0

    return {
        'pyupbit': has_pyupbit,
        'redis': redis_up,
        'prefetch_running': prefetch_running,
        'prefetch_index': _prefetch_index,
        'universe_len': universe_len,
    }


@app.get("/config") # 설정 조회 엔드포인트
def get_config():
    cfg = config._config
    return {"config": cfg}


@app.post("/config") # 설정 저장 엔드포인트
def post_config(payload: ConfigPayload):
    new_cfg = payload.config
    # 기본적인 검증: 반드시 strategy_name과 market이 있어야 함
    if not isinstance(new_cfg, dict) or 'strategy_name' not in new_cfg or 'market' not in new_cfg:
        raise HTTPException(status_code=400, detail="Invalid config payload. 'strategy_name' and 'market' required.")

    success = config.save_config(new_cfg)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save configuration")

    # 저장 후 재로딩
    config.reload_config()
    return {"status": "saved"}


@app.post("/reload") # 설정 재로딩 엔드포인트
def reload_config():
    config.reload_config()
    return {"status": "reloaded"}


# --- Screening endpoints ---
# 변동성 상위 N개 티커 조회
# market_prefix: 마켓 접두사 (기본값 "KRW")
# top_n: 상위 N개 (기본값 10)
# timeframe: 변동성 계산에 사용할 시간대 (기본값 "minute15")
# 반환값: 변동성 상위 N개 티커 리스트
# 변동성 계산은 (최고가 - 최저가) / 평균 종가 방식 사용
# Upbit의 공용 kline 엔드포인트 사용
# config.json의 'universe' 키에 티커 리스트가 없으면 기본 샘플 리스트 사용, 폴백 처리
# 반환값: {"top": [ {"ticker": 티커명, "volatility": 변동성}, ... ] }
# 캐시 사용으로 중복 Upbit 호출 최소화
# 캐시 TTL은 _KLINES_CACHE_TTL 사용
# 예외 발생 시 해당 티커는 건너뜀
@app.get("/screen/volatility_top") # 변동성 상위 티커 조회 엔드포인트
def volatility_top(market_prefix: str = "KRW", top_n: int = 10, timeframe: str = "minute15"):
    cfg = config._config # 설정 읽기
    universe = cfg.get('universe', []) # 유니버스 읽기
    if not universe:
        # 폴백: 기본 샘플 유니버스
        universe = [f"{market_prefix}-BTC", f"{market_prefix}-ETH", f"{market_prefix}-XRP", f"{market_prefix}-ADA", f"{market_prefix}-DOGE", f"{market_prefix}-SOL", f"{market_prefix}-DOT", f"{market_prefix}-MATIC", f"{market_prefix}-BCH", f"{market_prefix}-LTC"]

    results = []
    # Try to use internal cache to avoid hammering Upbit when checking multiple tickers
    now = time.time()
    for ticker in universe:
        # 캐시 키 생성 (가장 최근 15캔들 기준)
        key = f"{ticker}|{timeframe}|15"
        # 캐시 조회
        cached = _cache_get(key)
        # 캐시 유효성 검사
        if cached and (now - cached[0]) < _KLINES_CACHE_TTL: # 캐시 유효 시
            klines = cached[1] # 캐시된 값 사용
        else:
            try:
                # Upbit에서 변동성 계산용 klines 조회 (rate-limited)
                # 15캔들 기준 (폴백 200캔들 아님)
                klines = _rate_limited_get_klines(ticker, timeframe, count=15)
            except Exception as e:
                log.warning(f'Rate-limited fetch failed for {ticker}: {e}')
                klines = None
            # 캐시 설정, None도 캐시하여 반복 실패 방지
            _cache_set(key, klines, ttl=_KLINES_CACHE_TTL)
        if not klines:
            continue
        highs = [float(k['high_price']) for k in klines]
        lows = [float(k['low_price']) for k in klines]
        closes = [float(k['trade_price']) for k in klines]
        # 변동성 계산 : (최고가 - 최저가) / 평균 종가 방식
        try:
            vol = (max(highs) - min(lows)) / (sum(closes) / len(closes))
        except Exception:
            vol = 0
        results.append({'ticker': ticker, 'volatility': vol})

    # 변동성 기준 내림차순 정렬 후 상위 N개 반환
    results_sorted = sorted(results, key=lambda x: x['volatility'], reverse=True)[:top_n]
    return {"top": results_sorted}


# --- Background Event Watcher ---
# 단순 폴링 기반 워처 구현
# 워처는 별도 스레드에서 동작하며, 지정된 마켓의 klines를 주기적으로 조회
# 지정된 조건에 부합하는 이벤트 발생 시 로그 출력
# 조건은 JSON 배열로 전달되며, 각 조건은 다음과 같은 형태를 가짐
# {"type": "volatility_breakout", "k": 0.5} : 변동성 돌파 이벤트 (Larry Williams 스타일)
# {"type": "volume_spike", "multiplier": 3} : 거래량 급증 이벤트
# 워처 시작 엔드포인트
# 요청 본문 예시:
# {"market": "KRW-BTC",
# "interval": 1,
# "callbacks":[
#     {"type":"volatility_breakout", "k":0.5},
#     {"type":"volume_spike", "multiplier":3}
# ]}
# 워처 중지 엔드포인트
def _watcher_loop(stop_event, market: str, check_interval: float, callbacks: List[dict]):
    """Simple polling watcher that fetches latest klines and invokes callbacks when conditions met."""
    log.info(f"Starting watcher loop for {market} (interval {check_interval}s)")
    last_checked_time = None
    while not stop_event.is_set():
        try:
            try:
                # Upbit에서 최신 60캔들 조회 (rate-limited)
                klines = _rate_limited_get_klines(market, 'minute1', count=60)
            except Exception as e:
                log.error(f'Watcher fetch rate-limited or failed for {market}: {e}')
                klines = None

            # 이벤트 체크
            if klines:
                # 최근 캔들
                latest = klines[0]
                # 변동성 체크용 15캔들 윈도우 준비
                window = klines[:15]
                highs = [float(k['high_price']) for k in window]
                lows = [float(k['low_price']) for k in window]
                volumes = [float(k['candle_acc_trade_volume']) for k in window]
                closes = [float(k['trade_price']) for k in window]

                # 변동성 돌파 체크 (간단화된 Larry Williams 스타일)
                try:
                    prev_close = closes[1]
                    curr_close = closes[0]
                    volatility_range = max(highs) - min(lows)
                except Exception:
                    prev_close = curr_close = volatility_range = None

                # 콜백 조건 체크
                if prev_close is not None and curr_close is not None: # 유효한 데이터 시
                    avg_vol = sum(volumes[1:]) / (len(volumes)-1) if len(volumes) > 1 else 0
                    statuses = []
                    for cb in callbacks:
                        cb_type = cb.get('type')
                        # 변동성 돌파 체크
                        if cb_type == 'volatility_breakout':
                            k = cb.get('k', 0.5)
                            triggered = curr_close > (prev_close + volatility_range * k)
                            status = (
                                f"volatility_breakout(k={k}) current={curr_close:.0f} prev={prev_close:.0f} "
                                f"range={volatility_range:.0f} triggered={triggered}"
                            )
                            statuses.append(status)
                            if triggered:
                                log.info(f"Watcher detected volatility breakout on {market} (k={k})")
                        # 거래량 급증 체크
                        elif cb_type == 'volume_spike':
                            multiplier = cb.get('multiplier', 3)
                            triggered = avg_vol and volumes[0] > avg_vol * multiplier
                            status = (
                                f"volume_spike(mult={multiplier}) current_vol={volumes[0]:.0f} avg_vol={avg_vol:.0f} "
                                f"triggered={bool(triggered)}"
                            )
                            statuses.append(status)
                            if triggered:
                                log.info(f"Watcher detected volume spike on {market} (x{multiplier})")
                        else:
                            statuses.append(f"unknown callback {cb}")
                    config_desc = (f"market={market} interval={check_interval}s callbacks={len(callbacks)}")
                    log.info(f"WatcherCheck: {config_desc} | { ' ; '.join(statuses)}")

            time.sleep(check_interval)
        except Exception as e:
            log.error(f"Error in watcher loop: {e}")
            time.sleep(check_interval)
    log.info("Watcher loop stopped.")

# 워처 시작 엔드포인트
# 요청 본문 예시:
# {"market": "KRW-BTC",
# "interval": 1,
# "callbacks":[
#     {"type":"volatility_breakout", "k":0.5},
#     {"type":"volume_spike", "multiplier":3}
# ]}
@app.post("/watcher/start") # 워처 시작 엔드포인트
def start_watcher(payload: Dict[str, Any]):
    if _watcher['running']:
        raise HTTPException(status_code=400, detail="Watcher already running")

    # 파라미터 추출
    market = payload.get('market', config.MARKET)   # 마켓 (기본값 config.MARKET)
    interval = float(payload.get('interval', 1.0))  # 체크 간격 (초)
    callbacks = payload.get('callbacks', [])        # 콜백 조건 리스트

    # 워처 스레드 시작
    stop_event = threading.Event()
    # 워처 루프 스레드 생성 및 시작
    t = threading.Thread(target=_watcher_loop, args=(stop_event, market, interval, callbacks), daemon=True)
    _watcher['running'] = True              # 워처 상태 갱신
    _watcher['thread'] = t                  # 워처 스레드 저장
    _watcher['stop_event'] = stop_event     # 중지 이벤트 저장
    t.start()
    return {"status": "started"}

# 워처 중지 엔드포인트
# 워처 중지 이벤트 설정 및 스레드 종료 대기
# 워처 상태 초기화
# 반환값: {"status": "stopped"} 또는 {"status": "not_running"}
@app.post("/watcher/stop")
def stop_watcher():
    if not _watcher['running']:
        return {"status": "not_running"}
    _watcher['stop_event'].set()
    _watcher['thread'].join(timeout=5)
    _watcher['running'] = False
    _watcher['thread'] = None
    _watcher['stop_event'] = None
    return {"status": "stopped"}

# 배치 klines 조회 엔드포인트
# 티커/타임프레임/카운트 조합별로 캐시 키 생성 (인메모리 또는 Redis)
# 요청 본문 예시:
# {"tickers": ["KRW-BTC","KRW-ETH"], "timeframe":"minute15", "count":100}
# 반환값 예시:
# {"klines": {"KRW-BTC": [...], "KRW-ETH": [...]} }
# 각 티커별로 klines를 조회하여 결과 딕셔너리에 저장
# 내부적으로 캐시를 사용하여 중복 Upbit 호출 최소화
# 캐시 TTL은 _KLINES_CACHE_TTL 사용
# 캐시 미스 시 rate-limited fetcher를 사용하여 Upbit에서 klines 조회
# 예외 발생 시 해당 티커는 None으로 설정
@app.post('/klines_batch')
def klines_batch(payload: KlinesBatchRequest):
    req = payload.model_dump()
    tickers = req.get('tickers', []) or []
    timeframe = req.get('timeframe', 'minute15')
    count = int(req.get('count', 100))

    result = {}
    now = time.time()
    for ticker in tickers:
        key = f"{ticker}|{timeframe}|{count}" # 캐시 키 생성
        cached = _cache_get(key)
        # 캐시 유효성 검사
        # 캐시 유효 시 캐시된 값 사용
        if cached and (now - cached[0]) < _KLINES_CACHE_TTL:
            result[ticker] = cached[1]
            continue

        # 카운트 이상인 캐시 항목 검색 시도 (인메모리 및 Redis 모두 지원)
        # 가장 큰 count를 가진 항목 선택
        # 캐시 미스 시 rate-limited fetcher 사용
        klines = None
        try:
            # 가능한 경우 Redis에서 검색
            if _redis_client:
                try:
                    pattern = f"{ticker}|{timeframe}|*"
                    # 패턴 매칭 키 조회
                    keys = _redis_client.keys(pattern)
                    # 후보 탐색 및 선택 (요청보다 가장 큰 count)
                    best = None
                    best_cnt = 0
                    for k in keys:
                        try:
                            # 키 파싱 [ticker, timeframe, count]
                            parts = k.split('|')

                            # 유효한 키 형식 시 (3개 이상 파트로 구성)
                            if len(parts) >= 3:
                                # count 부분 (마지막 부분)
                                kcnt = int(parts[-1])
                                # 요청한 카운트보다 크고 현재 최상위 후보보다 큰 경우
                                if kcnt >= count and kcnt > best_cnt:
                                    best = k        # 후보 키 갱신
                                    best_cnt = kcnt # 후보 카운트 갱신
                        except Exception:
                            continue
                    # 후보 키가 발견된 경우
                    if best:
                        # 후보 키로 캐시 조회
                        cached2 = _cache_get(best, ttl=_KLINES_CACHE_TTL)
                        # 후보 캐시에서 klines 추출
                        if cached2:
                            klines_full = cached2[1]
                            # 유효한 klines 시
                            if isinstance(klines_full, list) and len(klines_full) > 0:
                                # 요청한 개수만큼 슬라이싱하여 반환
                                klines = klines_full[-count:]
                except Exception:
                    pass
            else:
                # 인-메모리 캐시에서 후보 탐색
                try:
                    candidates = []
                    # 인-메모리 캐시 순회
                    for k, v in list(_klines_cache.items()):
                        try:
                            # 키 파싱 [ticker, timeframe, count]
                            parts = k.split('|')
                            # 유효한 키 형식 시 (3개 이상 파트로 구성)
                            if parts[0] == ticker and parts[1] == timeframe:
                                kcnt = int(parts[2])            # count 부분
                                candidates.append((kcnt, v))    # 후보 리스트에 추가
                        except Exception:
                            continue
                    # 후보 정렬 및 선택 (요청보다 큰 count)
                    candidates = sorted(candidates, key=lambda x: x[0], reverse=True) # 내림차순 정렬
                    for kcnt, v in candidates:
                        if kcnt >= count:
                            klines_full = v[1]
                            # 유효한 klines 시
                            if isinstance(klines_full, list) and len(klines_full) > 0:
                                # 요청한 개수만큼 슬라이싱하여 반환
                                klines = klines_full[-count:]
                                break
                except Exception:
                    pass

            # 캐시에서 발견되지 않은 경우 rate-limited fetcher 사용
            if klines is None:
                try:
                    klines = _rate_limited_get_klines(ticker, timeframe, count=count)
                except Exception as e:
                    log.warning(f'Rate-limited batch fetch failed for {ticker}: {e}')
                    klines = None
        except Exception as e:
            log.warning(f'klines_batch lookup error for {ticker}: {e}')
            klines = None

        # 캐시 설정 (실패 시에도 캐시하여 반복 실패 방지)
        _cache_set(key, klines, ttl=_KLINES_CACHE_TTL)
        result[ticker] = klines

    return {'klines': result}

# --- Private API Endpoints ---
# 잔고 조회 엔드포인트
# Upbit 개인 API 키를 사용하여 잔고 조회
# 키가 구성되지 않은 경우 503 반환
# 이 엔드포인트는 짧은 TTL(_BALANCES_CACHE_TTL)로 잔고를 캐시하여 반복된 Upbit 호출을 줄임
# 반환값에는 추가 진단 필드 포함:
#   - balances: Upbit에서 반환된 원시 잔고 리스트
#   - reported_krw_balance: 잔고에서 보고된 KRW 잔고 (없으면 0)
#   - cached: 응답이 서버 캐시에서 왔는지 여부
#   - cached_ts: 캐시된 시점 타임스탬프
@app.get('/balances')
def get_balances():
    # 잔고 조회 엔드포인트
    if _upbit_private is None:
        raise HTTPException(status_code=503, detail='Upbit API keys not configured on server; balances unavailable')

    cache_key = 'upbit:balances:all'
    now = time.time()

    # 캐시 조회 시도
    cached = _cache_get(cache_key, ttl=_BALANCES_CACHE_TTL)
    if cached and (now - cached[0]) < _BALANCES_CACHE_TTL:
        bl = cached[1]
        cached_flag = True
        cached_ts = cached[0]
        log.debug('Balances: cache hit')
    else:
        cached_flag = False
        cached_ts = None
        try:
            bl = _upbit_private.get_balances()
        except Exception as e:
            log.error(f'Balances retrieval failed: {e}')
            # 캐시가 존재하면 캐시된 값 반환 (최선의 노력)
            if cached:
                bl = cached[1]
                cached_flag = True
                cached_ts = cached[0]
            else:
                raise HTTPException(status_code=502, detail=f'Upbit API call failed: {e}')
        # 캐시 설정 (실패 시에도 캐시하여 반복 실패 방지)
        _cache_set(cache_key, bl, ttl=_BALANCES_CACHE_TTL)

    # KRW 잔고 계산 (반환된 잔고에서)
    # 'currency' 또는 'unit' 필드 사용
    reported_krw = 0.0
    try:
        if isinstance(bl, list):
            for item in bl:
                # 업비트API /v1/accounts는 'currency'와 'balance' 필드를 가진 항목 반환
                try:
                    cur = str(item.get('currency') or item.get('unit') or '').upper()
                    bal = float(item.get('balance') or 0.0)
                except Exception:
                    continue
                if cur == 'KRW' or cur.startswith('KRW'):
                    reported_krw += bal
        elif isinstance(bl, dict):
            # 딕셔너리 형태인 경우 'balances' 키에서 리스트 추출
            lst = bl.get('balances') if 'balances' in bl else None
            if isinstance(lst, list):
                for item in lst:
                    try:
                        cur = str(item.get('currency') or item.get('unit') or '').upper()
                        bal = float(item.get('balance') or 0.0)
                    except Exception:
                        continue
                    if cur == 'KRW' or cur.startswith('KRW'):
                        reported_krw += bal
    except Exception:
        reported_krw = 0.0

    return {
        'balances': bl,
        'reported_krw_balance': reported_krw,
        'cached': bool(cached_flag),
        'cached_ts': cached_ts,
    }

# 포지션 조회 엔드포인트
# Upbit 개인 API 키를 사용하여 잔고 조회 후 현재 가격과 결합하여 포지션 계산
# 키가 구성되지 않은 경우 503 반환
# 각 자산별 포지션 정보와 요약 총계 반환
# 포지션 정보에는 다음 필드 포함:
#   - symbol: 마켓 심볼 (예: KRW-BTC)
#   - side: 포지션 방향 (항상 'LONG'으로 설정)
#   - size: 보유 수량
#   - entry_price: 평균 매수가 (없으면 null)
#   - current_price: 현재 가격
#   - unrealized_pnl: 미실현 손익 (없으면 null)
#   - unrealized_pnl_rate: 미실현 손익률 (없으면 null)
#   - notional_krw: 원화 기준 명목 가치
# 요약 정보에는 다음 필드 포함:
#   - total_equity_krw: 총 자산 가치 (원화 기준)
#   - available_krw: 사용 가능한 원화 잔고
#   - prices_fetched: 현재 가격을 성공적으로 조회한 자산 수
#   - excluded_assets: 현재 가격을 조회하지 못해 제외된 자산 목록 (심볼 및 사유 포함)
# 포지션 스냅샷은 히스토리 스토어에 기록됨
# 반환값 예시:
# {
#   "positions": [ {...}, {...}, ... ],
#   "total_equity_krw": 12345678.9,
#   "available_krw": 2345678.9,
#   "prices_fetched": 5,
#   "excluded_assets": [ {"symbol": "KRW-XYZ", "reason": "no_price"}, ... ]
# }
@app.get("/positions")
def get_positions():
    if _upbit_private is None:
        raise HTTPException(status_code=503, detail='Upbit API keys not configured on server; positions unavailable')

    try:
        bl = _upbit_private.get_balances() or []
    except Exception as e:
        log.error(f'Failed to retrieve balances for positions endpoint: {e}')
        raise HTTPException(status_code=502, detail=f'Failed to retrieve balances: {e}')

    positions = []
    total_equity = 0.0
    available_krw = 0.0

    #시장리스트 작성 및 통화맵 작성 (가격 조회용)
    markets = []
    currency_map = {}
    try:
        for item in bl:
            cur = str(item.get('currency') or item.get('unit') or '').upper()
            bal = float(item.get('balance') or 0) if item is not None else 0.0
            locked = float(item.get('locked') or 0) if item is not None else 0.0

            # 원화 현금잔고 처리
            if cur == 'KRW' or cur.startswith('KRW'):
                available_krw += bal
                total_equity += bal
                continue
            size = bal + locked
            if size <= 0:
                continue
            market = f'KRW-{cur}'
            markets.append(market)
            currency_map[market] = {
                'currency': cur,
                'size': size,
                'avg_buy_price': float(item.get('avg_buy_price') or 0)
            }
    except Exception as e:
        log.warning(f'Error while parsing balances for positions: {e}')

    # 현재가격 조회 (1개 캔들 minute1, count=1) (조회수 제한 주의)
    price_map = {}
    for m in set(markets):
        try:
            kl = None
            try:
                kl = _rate_limited_get_klines(m, 'minute1', count=1)
            except Exception as e:
                log.warning(f'Price fetch failed for {m}: {e}')
                kl = None
            price = None
            if kl and isinstance(kl, list) and len(kl) > 0:
                try:
                    first = kl[0]
                    price_candidate = None

                    # 딕셔너리 레코드 작업
                    # (Upbit API는 'trade_price' 사용)
                    if isinstance(first, dict):
                        price_candidate = first.get('trade_price') or first.get('close')
                    else:
                        # 속성 접근 시도 (일부 래퍼는 .close 또는 .trade_price 노출 가능)
                        price_candidate = getattr(first, 'trade_price', None) or getattr(first, 'close', None)
                    if price_candidate is None:
                        price = None
                    else:
                        price = float(price_candidate)
                except Exception:
                    price = None
            price_map[m] = price
        except Exception as e:
             log.warning(f'Unexpected error fetching price for {m}: {e}')
             price_map[m] = None

    # 포지션 구성 및 미실현 손익/명목 가치 계산
    excluded_assets = []
    for market, meta in currency_map.items():
        cur = meta['currency']
        size = float(meta['size'])
        avg_price = float(meta.get('avg_buy_price') or 0.0)
        current_price = price_map.get(market)

        # 자산 현재 가격이 없으면 건너뛰고 보고
        if current_price is None:
            excluded_assets.append({'symbol': market, 'reason': 'no_price'})
            continue

        notional = size * float(current_price)
        total_equity += notional
        unrealized = None
        unrealized_rate = None
        if avg_price and avg_price > 0:
            unrealized = (float(current_price) - avg_price) * size
            unrealized_rate = (float(current_price) - avg_price) / avg_price * 100

        pos = {
            'symbol': market,
            'side': 'LONG',
            'size': size,
            'entry_price': avg_price if avg_price > 0 else None,
            'current_price': current_price,
            'unrealized_pnl': unrealized,
            'unrealized_pnl_rate': unrealized_rate,
            'notional_krw': notional,
        }
        positions.append(pos)

    # Also include list of excluded assets so UI can show a friendly message.
    result = {
        'positions': positions,
        'total_equity_krw': total_equity,
        'available_krw': available_krw,
        'prices_fetched': len([p for p in price_map.values() if p is not None]),
        'excluded_assets': excluded_assets,
    }
    history_store.record_snapshot({
        'ts': time.time(),
        'total_equity': total_equity,
        'available_krw': available_krw,
        'positions': [
            {
                'symbol': pos['symbol'],
                'notional_krw': pos['notional_krw'],
                'unrealized_pnl': pos['unrealized_pnl'],
            }
            for pos in positions
        ],
    })
    return result


@app.get('/ai/history')
def get_ai_history(limit: int = 50):
    try:
        limit = max(1, min(int(limit), 200))
    except Exception:
        limit = 50
    history = ai_history_store.get_history(limit=limit)
    return {'items': history}


@app.get('/positions/history')
def get_positions_history(limit: int = 365, days: int = 365):
    since = time.time() - float(days) * 86400
    history = history_store.get_history(since=since, limit=limit)
    return {'history': history}


@app.post('/ws/start')
def ws_start():
    try:
        start_ws_listener()
        start_ticker_listener()
        return {'status': 'started'}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Failed to start websocket listener: {exc}')

@app.post('/ws/stop')
def ws_stop():
    try:
        stop_ws_listener()
        stop_ticker_listener()
        return {'status': 'stopped'}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Failed to stop websocket listener: {exc}')

@app.get('/ws/status')
def ws_status():
    running = bool(_ws_listener and _ws_listener._thread and _ws_listener._thread.is_alive())
    return {'running': running, 'targets': _ws_listener.targets if _ws_listener else []}

@app.get('/ws/stats')
def ws_stats(last_hour_sec: int = 3600, recent_limit: int = 10):
    raw_stats = load_ws_stats()
    summary = summarize_ws_stats(raw_stats, last_hour_secs=last_hour_sec, recent_limit=recent_limit)
    summary.update({
        'running': bool(_ws_listener and _ws_listener._thread and _ws_listener._thread.is_alive()),
        'targets': _ws_listener.targets if _ws_listener else [],
    })
    return summary

@app.get('/ws/executions')
def ws_executions(limit: int = 0):
    try:
        entries = read_exec_history(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Failed to load exec history: {exc}')
    return {'executions': entries}

@app.get('/ws/trades')
def ws_trades(symbol: str, limit: int = 20):
    if _redis_client is None:
        raise HTTPException(status_code=503, detail='Redis cache unavailable; cannot read trades.')
    if not symbol:
        raise HTTPException(status_code=400, detail='symbol query parameter is required.')
    max_limit = min(max(limit, 1), 200)
    key = f'ws:trades:{symbol}'
    raw = _redis_client.lrange(key, 0, max_limit - 1)
    trades = []
    try:
        for item in raw:
            import json as _json
            trades.append(_json.loads(item))
    except Exception:
        trades = []
    return {'symbol': symbol, 'trades': trades}


def _ws_ticker_targets() -> List[str]:
    if _ws_listener:
        return _ws_listener.targets
    universe = config._config.get('universe')
    if isinstance(universe, list) and universe:
        return universe
    return [
        'KRW-BTC',
        'KRW-ETH',
        'KRW-ADA',
        'KRW-XRP',
        'KRW-SOL',
    ]


@app.get('/ws/ticker_data')
def ws_ticker_data():
    if _redis_client is None:
        raise HTTPException(status_code=503, detail='Redis cache unavailable; cannot read ticker data.')
    targets = _ws_ticker_targets()
    payloads: List[Dict[str, Any]] = []
    for symbol in targets:
        key = f'ws:ticker:{symbol}'
        raw = _redis_client.get(key)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        payloads.append({
            'symbol': symbol,
            'opening_price': data.get('opening_price'),
            'high_price': data.get('high_price'),
            'low_price': data.get('low_price'),
            'trade_price': data.get('trade_price') or data.get('trade_price'),
            'prev_closing_price': data.get('prev_closing_price'),
            'change': data.get('change'),
            'timestamp': data.get('trade_timestamp') or data.get('timestamp'),
        })
    return {'tickers': payloads}

# --- Trading Bot Control API ---

# 봇 실행 중 여부 확인
def _bot_running():
    return config.BOT_ENABLED


@app.post('/bot/control')
def bot_control(payload: Dict[str, Any]):
    enabled = payload.get('enabled')
    interval = payload.get('interval_sec')
    if enabled is None and interval is None:
        raise HTTPException(status_code=400, detail='enabled or interval_sec required')
    try:
        updated = config.update_bot_control(bot_enabled=enabled, bot_interval_sec=interval)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Failed to update bot config: {exc}')
    return {'status': 'updated', 'bot_enabled': config.BOT_ENABLED, 'bot_interval_sec': config.BOT_INTERVAL_SEC, 'updated': updated}


@app.get('/bot/status')
def bot_status():
    return {
        'bot_enabled': config.BOT_ENABLED,
        'bot_interval_sec': config.BOT_INTERVAL_SEC,
        'running': config.BOT_ENABLED,
    }

