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
from concurrent.futures import ThreadPoolExecutor, as_completed # for parallel prefetching

# 내부API 모듈 임포트
from server import config               # 런타임 설정 관리
from server.upbit_api import UpbitAPI   # 업비트 API 연동
from server.logger import log           # 로깅 설정

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
        yield
    finally:
        # 스케쥴러 중지
        stop_prefetch_scheduler()

app = FastAPI(title="Upbit Trader Runtime API", lifespan=lifespan) # FastAPI 앱 생성

# 데이터 모델 정의
class ConfigPayload(BaseModel):
    config: Dict[str, Any]

# 배치 klines 요청 모델
class KlinesBatchRequest(BaseModel):
    tickers: List[str]
    timeframe: Optional[str] = 'minute15'
    count: Optional[int] = 100

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
    # If Redis unavailable, bump default interval to reduce Upbit calls
    if _redis_client is None:
        interval = max(interval, 60)
        log.info('Redis not connected: starting prefetch with interval %s seconds', interval)
    # ensure smaller default batch if no redis
    if _redis_client is None:
        try:
            # reduce batch size to 3 if not configured
            if 'prefetch_batch_size' not in config._config:
                config._config['prefetch_batch_size'] = 3
        except Exception:
            pass
    # initialize prefetch rate limiter and semaphore based on runtime config
    global _prefetch_token_bucket, _prefetch_semaphore
    try:
        rate = int(config._config.get('prefetch_rate_per_sec', 5))
    except Exception:
        rate = 5
    try:
        capacity = int(config._config.get('prefetch_rate_capacity', max(1, rate)))
    except Exception:
        capacity = max(1, rate)
    try:
        max_concurrent = int(config._config.get('prefetch_max_concurrent', 3))
    except Exception:
        max_concurrent = 3
    try:
        _prefetch_token_bucket = TokenBucket(rate=float(rate), capacity=float(capacity))
        _prefetch_semaphore = threading.BoundedSemaphore(max_concurrent)
        log.info(f'Prefetch rate limiter initialized: rate={rate}/s, capacity={capacity}, max_concurrent={max_concurrent}')
    except Exception as e:
        _prefetch_token_bucket = None
        _prefetch_semaphore = None
        log.warning(f'Failed to initialize prefetch rate limiter: {e}')
    _prefetch_thread = threading.Thread(target=_prefetch_loop, args=(interval,), daemon=True)
    _prefetch_thread.start()


def stop_prefetch_scheduler():
    global _prefetch_thread, _prefetch_stop
    _prefetch_stop.set()
    if _prefetch_thread is not None:
        _prefetch_thread.join(timeout=2)
    _prefetch_thread = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get('/debug/status')
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


@app.get("/config")
def get_config():
    cfg = config._config
    return {"config": cfg}


@app.post("/config")
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


@app.post("/reload")
def reload_config():
    config.reload_config()
    return {"status": "reloaded"}


# --- Screening endpoints ---
@app.get("/screen/volatility_top")
def volatility_top(market_prefix: str = "KRW", top_n: int = 10, timeframe: str = "minute15"):
    """Return top N tickers by volatility over the last timeframe. Uses public kline endpoint for multiple markets.
    Note: Upbit does not provide a single endpoint for all tickers' klines, so this endpoint expects the runtime/config.json to
    include a list of tickers to check under 'universe' key or will fallback to a small default list."""
    cfg = config._config
    universe = cfg.get('universe', [])
    if not universe:
        # fallback sample universe
        universe = [f"{market_prefix}-BTC", f"{market_prefix}-ETH", f"{market_prefix}-XRP", f"{market_prefix}-ADA", f"{market_prefix}-DOGE", f"{market_prefix}-SOL", f"{market_prefix}-DOT", f"{market_prefix}-MATIC", f"{market_prefix}-BCH", f"{market_prefix}-LTC"]

    results = []
    # Try to use internal cache to avoid hammering Upbit when checking multiple tickers
    now = time.time()
    for ticker in universe:
        key = f"{ticker}|{timeframe}|15"
        cached = _cache_get(key)
        if cached and (now - cached[0]) < _KLINES_CACHE_TTL:
            klines = cached[1]
        else:
            try:
                klines = _rate_limited_get_klines(ticker, timeframe, count=15)
            except Exception as e:
                log.warning(f'Rate-limited fetch failed for {ticker}: {e}')
                klines = None
            # cache response even if None to avoid repeated failures
            _cache_set(key, klines, ttl=_KLINES_CACHE_TTL)
        if not klines:
            continue
        highs = [float(k['high_price']) for k in klines]
        lows = [float(k['low_price']) for k in klines]
        closes = [float(k['trade_price']) for k in klines]
        # volatility measure: stddev of returns or (max-min)/mean
        try:
            vol = (max(highs) - min(lows)) / (sum(closes) / len(closes))
        except Exception:
            vol = 0
        results.append({'ticker': ticker, 'volatility': vol})

    results_sorted = sorted(results, key=lambda x: x['volatility'], reverse=True)[:top_n]
    return {"top": results_sorted}


# --- Background Event Watcher ---

def _watcher_loop(stop_event, market: str, check_interval: float, callbacks: List[dict]):
    """Simple polling watcher that fetches latest klines and invokes callbacks when conditions met."""
    log.info(f"Starting watcher loop for {market} (interval {check_interval}s)")
    last_checked_time = None
    while not stop_event.is_set():
        try:
            try:
                klines = _rate_limited_get_klines(market, 'minute1', count=60)
            except Exception as e:
                log.error(f'Watcher fetch rate-limited or failed for {market}: {e}')
                klines = None

            if klines:
                # latest candle
                latest = klines[0]
                # prepare a small window for volatility check (last 15 candles)
                window = klines[:15]
                highs = [float(k['high_price']) for k in window]
                lows = [float(k['low_price']) for k in window]
                volumes = [float(k['candle_acc_trade_volume']) for k in window]
                closes = [float(k['trade_price']) for k in window]

                # Volatility breakout check (Larry Williams style simplified)
                try:
                    prev_close = closes[1]
                    curr_close = closes[0]
                    volatility_range = max(highs) - min(lows)
                except Exception:
                    prev_close = curr_close = volatility_range = None

                if prev_close is not None and curr_close is not None:
                    for cb in callbacks:
                        if cb.get('type') == 'volatility_breakout':
                            k = cb.get('k', 0.5)
                            # when current close > prev_close + range * k
                            if curr_close > (prev_close + volatility_range * k):
                                log.info(f"Watcher detected volatility breakout on {market} (k={k})")
                        elif cb.get('type') == 'volume_spike':
                            multiplier = cb.get('multiplier', 3)
                            avg_vol = sum(volumes[1:]) / (len(volumes)-1) if len(volumes) > 1 else 0
                            if avg_vol and volumes[0] > avg_vol * multiplier:
                                log.info(f"Watcher detected volume spike on {market} (x{multiplier})")

            time.sleep(check_interval)
        except Exception as e:
            log.error(f"Error in watcher loop: {e}")
            time.sleep(check_interval)
    log.info("Watcher loop stopped.")


@app.post("/watcher/start")
def start_watcher(payload: Dict[str, Any]):
    """Start background watcher with simple JSON payload:
    {"market": "KRW-BTC", "interval": 1, "callbacks":[{"type":"volatility_breakout", "k":0.5}, {"type":"volume_spike", "multiplier":3}]}
    """
    if _watcher['running']:
        raise HTTPException(status_code=400, detail="Watcher already running")

    market = payload.get('market', config.MARKET)
    interval = float(payload.get('interval', 1.0))
    callbacks = payload.get('callbacks', [])

    stop_event = threading.Event()
    t = threading.Thread(target=_watcher_loop, args=(stop_event, market, interval, callbacks), daemon=True)
    _watcher['running'] = True
    _watcher['thread'] = t
    _watcher['stop_event'] = stop_event
    t.start()
    return {"status": "started"}


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


@app.post('/klines_batch')
def klines_batch(payload: KlinesBatchRequest):
    """Return klines for multiple tickers in a single request. Uses in-memory cache per ticker/timeframe/count key.
    Request body: {"tickers": ["KRW-BTC","KRW-ETH"], "timeframe":"minute15", "count":100}
    Response: {"klines": {"KRW-BTC": [...], ...}}
    """
    req = payload.model_dump()
    tickers = req.get('tickers', []) or []
    timeframe = req.get('timeframe', 'minute15')
    count = int(req.get('count', 100))

    result = {}
    now = time.time()
    for ticker in tickers:
        key = f"{ticker}|{timeframe}|{count}"
        cached = _cache_get(key)
        if cached and (now - cached[0]) < _KLINES_CACHE_TTL:
            result[ticker] = cached[1]
            continue

        # Attempt to find a cached entry with a larger count and slice it (supports both Redis and in-memory)
        klines = None
        try:
            # search in Redis first if available
            if _redis_client:
                try:
                    pattern = f"{ticker}|{timeframe}|*"
                    keys = _redis_client.keys(pattern)
                    # find candidate with largest count >= requested
                    best = None
                    best_cnt = 0
                    for k in keys:
                        try:
                            parts = k.split('|')
                            if len(parts) >= 3:
                                kcnt = int(parts[-1])
                                if kcnt >= count and kcnt > best_cnt:
                                    best = k
                                    best_cnt = kcnt
                        except Exception:
                            continue
                    if best:
                        cached2 = _cache_get(best, ttl=_KLINES_CACHE_TTL)
                        if cached2:
                            klines_full = cached2[1]
                            if isinstance(klines_full, list) and len(klines_full) > 0:
                                # return last `count` elements (most recent)
                                klines = klines_full[-count:]
                except Exception:
                    pass
            else:
                # in-memory search
                try:
                    candidates = []
                    for k, v in list(_klines_cache.items()):
                        try:
                            parts = k.split('|')
                            if parts[0] == ticker and parts[1] == timeframe:
                                kcnt = int(parts[2])
                                candidates.append((kcnt, v))
                        except Exception:
                            continue
                    # pick candidate with largest count >= requested
                    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
                    for kcnt, v in candidates:
                        if kcnt >= count:
                            klines_full = v[1]
                            if isinstance(klines_full, list) and len(klines_full) > 0:
                                klines = klines_full[-count:]
                                break
                except Exception:
                    pass

            # If still not found, perform a rate-limited fetch
            if klines is None:
                try:
                    klines = _rate_limited_get_klines(ticker, timeframe, count=count)
                except Exception as e:
                    log.warning(f'Rate-limited batch fetch failed for {ticker}: {e}')
                    klines = None
        except Exception as e:
            log.warning(f'klines_batch lookup error for {ticker}: {e}')
            klines = None

        # cache result (even if None) to prevent hammering
        _cache_set(key, klines, ttl=_KLINES_CACHE_TTL)
        result[ticker] = klines

    return {'klines': result}
