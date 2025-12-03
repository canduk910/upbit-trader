import json
import os
import threading
import time
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import websocket  # websocket-client

from server.config import get_setting
from server.logger import get_logger

UPBIT_WS_URL = "wss://api.upbit.com/websocket/v1"
EXEC_HISTORY_DIR = Path(__file__).resolve().parents[1] / "runtime" / "history"
WS_STATS_TICKER_FILE = EXEC_HISTORY_DIR / "ws_ticker_stats.json"
KST = timezone(timedelta(hours=9))

logger = get_logger(name='UpbitTickerListener', log_file='ws_ticker.log', level=logging.INFO)


def _ensure_history_dir() -> None:
    EXEC_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _timeframe_to_seconds(timeframe: str) -> int:
    if not isinstance(timeframe, str):
        return 60
    tf = timeframe.lower()
    if tf.startswith("minute"):
        try:
            return max(int(tf.replace("minute", "")) * 60, 60)
        except ValueError:
            return 60
    if tf.startswith("hour"):
        try:
            return max(int(tf.replace("hour", "")) * 3600, 3600)
        except ValueError:
            return 3600
    if tf.startswith("day"):
        return 86400
    return 60


def _get_universe_targets() -> List[str]:
    universe = get_setting('universe')
    if isinstance(universe, list) and universe:
        return universe
    return ["KRW-BTC", "KRW-ETH", "KRW-ADA", "KRW-XRP", "KRW-SOL"]


def _load_stats() -> Dict[str, Any]:
    _ensure_history_dir()
    if not WS_STATS_TICKER_FILE.exists():
        return {'total_success': 0, 'total_failure': 0, 'history': []}
    try:
        with WS_STATS_TICKER_FILE.open('r', encoding='utf-8') as fp:
            return json.load(fp)
    except Exception:
        return {'total_success': 0, 'total_failure': 0, 'history': []}


class TickerWebsocketListener:
    def __init__(self, redis_client: Optional[Any] = None):
        self.redis_client = redis_client
        self._stats = _load_stats()
        self._stats_history = deque(self._stats.get('history', []), maxlen=4000)
        self._stats_lock = threading.Lock()
        self._targets = _get_universe_targets()
        self._timeframes = self._resolve_timeframes()
        self._tf_state: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._candle_state: Dict[str, Dict[str, Any]] = {}
        self._last_acc_volume: Dict[str, float] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws: Optional[websocket.WebSocketApp] = None

    def _resolve_timeframes(self) -> List[str]:
        cfg_frames = get_setting('ws_timeframes')
        frames: List[str] = []
        if isinstance(cfg_frames, list):
            for tf in cfg_frames:
                if isinstance(tf, str) and tf.strip():
                    frames.append(tf.strip())
        primary = get_setting('timeframe') or 'minute5'
        if primary not in frames:
            frames.append(primary)
        if 'minute1' not in frames:
            frames.insert(0, 'minute1')
        seen: List[str] = []
        for tf in frames:
            if tf not in seen:
                seen.append(tf)
        return seen

    def _save_stats(self) -> None:
        try:
            temp = WS_STATS_TICKER_FILE.with_suffix('.tmp')
            with temp.open('w', encoding='utf-8') as fp:
                json.dump({
                    'total_success': self._stats.get('total_success', 0),
                    'total_failure': self._stats.get('total_failure', 0),
                    'history': list(self._stats_history),
                }, fp)
            temp.replace(WS_STATS_TICKER_FILE)
        except Exception as exc:
            logger.warning(f'Failed to save ticker stats: {exc}')

    def _register_reception(self, payload: Dict[str, Any], success: bool) -> None:
        with self._stats_lock:
            self._stats.setdefault('total_success', 0)
            self._stats.setdefault('total_failure', 0)
            if success:
                self._stats['total_success'] += 1
            else:
                self._stats['total_failure'] += 1
            entry = {
                'ts': payload.get('trade_timestamp') or payload.get('timestamp') or int(time.time() * 1000),
                'symbol': payload.get('code'),
                'type': payload.get('type'),
                'success': success,
            }
            self._stats_history.append(entry)
            self._save_stats()

    def _payload_for_subscription(self) -> str:
        return json.dumps([
            {"ticket": "upbit-ws-ticker"},
            {"type": "ticker", "codes": self._targets, "isOnlyRealtime": False},
        ])

    def _push_candle(self, ticker: str, timeframe: str, candle: Dict[str, Any]) -> None:
        if self.redis_client is None:
            return
        key = f"ws:candles:{timeframe}:{ticker}"
        payload = candle.copy()
        ts = payload.get('timestamp')
        if ts:
            try:
                qualifier = datetime.fromtimestamp(ts / 1000.0, KST)
                payload['candle_date_time_kst'] = qualifier.strftime('%Y-%m-%dT%H:%M:%S%z')
            except Exception:
                pass
        payload.setdefault('candle_acc_trade_volume', payload.get('volume', 0.0))
        try:
            self.redis_client.lpush(key, json.dumps(payload))
            self.redis_client.ltrim(key, 0, int(os.getenv('WS_CANDLE_HISTORY_LIMIT', '240')) - 1)
        except Exception as exc:
            logger.warning(f"Failed to push ticker candle to redis: {exc}")

    def _emit_to_timeframes(self, ticker: str, base_ts: float, candle: Dict[str, Any]) -> None:
        bucket_map = self._tf_state.setdefault(ticker, {})
        for timeframe in self._timeframes:
            if timeframe == 'minute1':
                continue
            duration = _timeframe_to_seconds(timeframe) * 1000
            bucket_start = int(base_ts // duration) * duration
            state = bucket_map.get(timeframe)
            if not state or state.get('start') != bucket_start:
                if state:
                    self._push_candle(ticker, timeframe, state)
                bucket_map[timeframe] = {
                    'ticker': ticker,
                    'timeframe': timeframe,
                    'timestamp': bucket_start,
                    'start': bucket_start,
                    'open': candle.get('open'),
                    'high': candle.get('high'),
                    'low': candle.get('low'),
                    'close': candle.get('close'),
                    'volume': candle.get('volume', 0.0),
                }
            else:
                state['high'] = max(state.get('high', 0.0), candle.get('high', 0.0))
                state['low'] = min(state.get('low', state.get('high', 0.0)), candle.get('low', 0.0))
                state['close'] = candle.get('close')
                state['volume'] = state.get('volume', 0.0) + candle.get('volume', 0.0)

    def _aggregate_candle(self, payload: Dict[str, Any]) -> None:
        ticker = payload.get('code')
        if not ticker:
            return
        ts = payload.get('trade_timestamp') or payload.get('timestamp') or int(time.time() * 1000)
        try:
            ts_val = float(ts)
        except Exception:
            ts_val = float(time.time() * 1000)
        minute_ts = int(ts_val // 60000 * 60000)
        price = float(payload.get('trade_price') or payload.get('trade_price') or 0.0)
        if price <= 0:
            return
        state = self._candle_state.get(ticker)
        if not state or state.get('minute') != minute_ts:
            if state:
                self._push_candle(ticker, 'minute1', {
                    'ticker': ticker,
                    'timestamp': state['minute'],
                    'open': state['open'],
                    'high': state['high'],
                    'low': state['low'],
                    'close': state['close'],
                    'volume': state['volume'],
                })
                self._emit_to_timeframes(ticker, state['minute'], state)
            self._candle_state[ticker] = {
                'minute': minute_ts,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0.0,
            }
            self._last_acc_volume[ticker] = float(payload.get('acc_trade_volume', 0) or 0)
            return
        if price > state['high']:
            state['high'] = price
        if price < state['low']:
            state['low'] = price
        state['close'] = price
        acc_vol = float(payload.get('acc_trade_volume', 0) or 0)
        prev_acc = self._last_acc_volume.get(ticker, 0.0)
        if acc_vol >= prev_acc:
            state['volume'] += acc_vol - prev_acc
        else:
            state['volume'] += acc_vol
        self._last_acc_volume[ticker] = acc_vol

    def _handle_message(self, payload: Dict[str, Any]) -> None:
        if payload.get('type') != 'ticker':
            self._register_reception(payload, False)
            return
        try:
            self._aggregate_candle(payload)
            self._register_reception(payload, True)
        except Exception as exc:
            logger.warning(f"Ticker payload handling failed: {exc}")
            self._register_reception(payload, False)

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        self._ws = ws
        try:
            ws.send(self._payload_for_subscription())
            logger.info('Ticker websocket subscription sent')
        except Exception as exc:
            logger.warning(f"Failed to send ticker subscription: {exc}")

    def _on_close(self, _, code, msg) -> None:
        logger.info(f"Ticker websocket closed(code={code} msg={msg})")

    def _on_error(self, _, error: Any) -> None:
        logger.warning(f"Ticker websocket error: {error}")

    def _on_message(self, _, message: str) -> None:
        try:
            payloads = json.loads(message)
        except Exception as exc:
            logger.warning(f"Failed to parse ticker message: {exc}")
            return
        if isinstance(payloads, list):
            for payload in payloads:
                self._handle_message(payload)
        elif isinstance(payloads, dict):
            self._handle_message(payloads)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                ws_app = websocket.WebSocketApp(
                    UPBIT_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                logger.warning(f"Ticker websocket restart: {exc}")
            time.sleep(2)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Ticker websocket listener started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Ticker websocket listener stopped")

