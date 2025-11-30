from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import threading
import time
import os
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed

from server import config
from server.upbit_api import UpbitAPI
from server.logger import log

@asynccontextmanager
async def lifespan(app: FastAPI):
    # start prefetch scheduler on startup
    try:
        interval = int(config._config.get('prefetch_interval_sec', 30))
    except Exception:
        interval = 30
    start_prefetch_scheduler(interval=interval)
    try:
        yield
    finally:
        stop_prefetch_scheduler()

app = FastAPI(title="Upbit Trader Runtime API", lifespan=lifespan)


class ConfigPayload(BaseModel):
    config: Dict[str, Any]


class KlinesBatchRequest(BaseModel):
    tickers: List[str]
    timeframe: Optional[str] = 'minute15'
    count: Optional[int] = 100


# Simple in-memory watcher state (explicit typing to satisfy linters)
_watcher: Dict[str, Any] = {
    'running': False,
    'thread': None,
    'stop_event': None,
}

# Simple in-memory cache for batch klines: { key: (timestamp, data) }
_klines_cache: Dict[str, Any] = {}
_KLINES_CACHE_TTL = 120  # seconds (increase to reduce Upbit public API calls)


# Redis setup (optional). Use local redis://localhost:6379 if REDIS_URL not set
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
_redis_client = None
try:
    _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    # quick ping
    _redis_client.ping()
    log.info(f'Redis cache connected: {REDIS_URL}')
except Exception as e:
    _redis_client = None
    log.warning(f'Redis not available ({REDIS_URL}): {e}. Falling back to in-memory cache')


def _cache_set(key: str, value: Any, ttl: int = _KLINES_CACHE_TTL):
    """Set cache in Redis if available, else in-memory."""
    now = time.time()
    if _redis_client:
        try:
            # store JSON string
            import json as _json
            payload = {'ts': now, 'value': value}
            _redis_client.setex(key, ttl, _json.dumps(payload))
            return
        except Exception:
            pass
    _klines_cache[key] = (now, value)


def _cache_get(key: str, ttl: int = _KLINES_CACHE_TTL):
    """Get cache value. Return (timestamp, value) or None."""
    if _redis_client:
        try:
            import json as _json
            s = _redis_client.get(key)
            if not s:
                return None
            obj = _json.loads(s)
            if time.time() - obj.get('ts', 0) > ttl:
                return None
            return (obj.get('ts'), obj.get('value'))
        except Exception:
            return None
    return _klines_cache.get(key)


# provide a lightweight UpbitAPI instance (no keys required for public endpoints)
_upbit_public = UpbitAPI()

# Prefetch scheduler controls
_prefetch_thread: Optional[threading.Thread] = None
_prefetch_stop = threading.Event()
_prefetch_index = 0


def _prefetch_loop(interval: int = 30):
    log.info('Prefetch scheduler started')
    effective_interval = interval
    while not _prefetch_stop.is_set():
        try:
            cfg = config._config
            universe = cfg.get('universe', [])
            # determine per-run settings (allow overrides in runtime config)
            cfg_count = int(cfg.get('prefetch_count', 200))
            # if Redis not available, be conservative
            if _redis_client is None:
                count = min(cfg_count, 100)
                per_ticker_sleep = float(cfg.get('prefetch_sleep_sec', 1.0))
                log.info('Redis not available: using conservative prefetch settings (count=%s, sleep=%.2f)', count, per_ticker_sleep)
            else:
                count = cfg_count
                per_ticker_sleep = float(cfg.get('prefetch_sleep_sec', 0.2))
            # ensure interval is at least a minimum when no Redis
            effective_interval = interval
            if _redis_client is None:
                effective_interval = max(interval, int(cfg.get('prefetch_min_interval_sec', 60)))
            if universe:
                # staggered batch processing to avoid bursts
                batch_size = int(cfg.get('prefetch_batch_size', 5))
                parallelism = int(cfg.get('prefetch_parallelism', 3))
                global _prefetch_index
                n = len(universe)
                if n == 0:
                    pass
                else:
                    start = _prefetch_index % n
                    end = start + batch_size
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
                            klines_local = _upbit_public.get_klines(ticker_local, 'minute15', count=count)
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
            klines = _upbit_public.get_klines(ticker, timeframe, count=15)
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
            klines = _upbit_public.get_klines(market, 'minute1', count=60)
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
                prev_close = closes[1]
                curr_close = closes[0]
                volatility_range = max(highs) - min(lows)

                for cb in callbacks:
                    if cb.get('type') == 'volatility_breakout':
                        k = cb.get('k', 0.5)
                        # when current close > prev_close + range * k
                        if curr_close > (prev_close + volatility_range * k):
                            log.info(f"Watcher detected volatility breakout on {market} (k={k})")
                    elif cb.get('type') == 'volume_spike':
                        multiplier = cb.get('multiplier', 3)
                        avg_vol = sum(volumes[1:]) / (len(volumes)-1)
                        if volumes[0] > avg_vol * multiplier:
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

        klines = _upbit_public.get_klines(ticker, timeframe, count=count)
        # cache result (even if None) to prevent hammering
        _cache_set(key, klines, ttl=_KLINES_CACHE_TTL)
        result[ticker] = klines

    return {'klines': result}
