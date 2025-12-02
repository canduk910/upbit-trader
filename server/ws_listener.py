import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import websocket  # websocket-client

from server.logger import log
from server.config import get_setting

UPBIT_WS_URL = "wss://api.upbit.com/websocket/v1"
EXEC_HISTORY_DIR = Path(__file__).resolve().parents[1] / "runtime" / "history"
EXEC_HISTORY_FILE = EXEC_HISTORY_DIR / "exec_history.json"
EXEC_HISTORY_MAX = int(os.getenv("EXEC_HISTORY_MAX_ENTRIES", "1024"))
EXEC_HISTORY_LOCK = threading.Lock()
KST = timezone(timedelta(hours=9))


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


def _ensure_exec_history_dir() -> None:
    EXEC_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


class ExecHistoryStore:
    def __init__(self, path: Path = EXEC_HISTORY_FILE, max_entries: int = EXEC_HISTORY_MAX):
        self.path = path
        self.max_entries = max_entries
        _ensure_exec_history_dir()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception as exc:
            log.warning(f"Failed to load exec history: {exc}")
            return []

    def _save(self, data: List[Dict[str, Any]]) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False)
        except Exception as exc:
            log.warning(f"Failed to write exec history: {exc}")

    def record(self, entry: Dict[str, Any]) -> None:
        with EXEC_HISTORY_LOCK:
            data = self._load()
            data.append(entry)
            if len(data) > self.max_entries:
                data = data[-self.max_entries :]
            self._save(data)


class WebsocketListener:
    def __init__(self, redis_client: Optional[Any] = None):
        self.redis_client = redis_client
        universe = get_setting("universe")
        if isinstance(universe, list) and universe:
            self.targets = universe
        else:
            # fallback sample list
            self.targets = ["KRW-BTC", "KRW-ETH", "KRW-ADA", "KRW-XRP", "KRW-SOL"]
        self.history_store = ExecHistoryStore()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws: Optional[websocket.WebSocketApp] = None
        self._candle_state: Dict[str, Dict[str, Any]] = {}
        self._candle_history_limit = int(os.getenv('WS_CANDLE_HISTORY_LIMIT', '240'))
        self._last_acc_volume: Dict[str, float] = {}
        self._entry_prices: Dict[str, float] = {}
        self._timeframes = self._resolve_timeframes()
        self._tf_state: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _payload_for_subscription(self) -> str:
        message = [
            {"ticket": "upbit-ws-listener"},
            {"type": "ticker", "codes": self.targets, "isOnlyRealtime": False},
            {"type": "order", "codes": ["KRW-" + t.split('-')[-1] if not t.startswith('KRW-') else t for t in self.targets]},
        ]
        return json.dumps(message)

    def _push_candle(self, ticker: str, timeframe: str, candle: Dict[str, Any]) -> None:
        if self.redis_client is None:
            return
        try:
            key = f"ws:candles:{timeframe}:{ticker}"
            candle = candle.copy()
            ts = candle.get("timestamp")
            if ts:
                try:
                    qualifier = datetime.fromtimestamp(ts / 1000.0, KST)
                    candle["candle_date_time_kst"] = qualifier.strftime("%Y-%m-%dT%H:%M:%S%z")
                except Exception:
                    pass
            candle.setdefault("candle_acc_trade_volume", candle.get("volume", 0.0))
            self.redis_client.lpush(key, json.dumps(candle))
            self.redis_client.ltrim(key, 0, self._candle_history_limit - 1)
        except Exception as exc:
            log.warning(f"Redis candle push failed: {exc}")

    def _read_cached_candles(self, ticker: str, timeframe: str = 'minute1', limit: int = 200) -> List[Dict[str, Any]]:
        if self.redis_client is None:
            return []
        key = f"ws:candles:{timeframe}:{ticker}"
        try:
            raw = self.redis_client.lrange(key, 0, limit - 1)
        except Exception:
            return []
        result = []
        for raw_item in reversed(raw):
            try:
                payload = json.loads(raw_item)
                result.append(payload)
            except Exception:
                continue
        return result

    def _aggregate_candle(self, payload: Dict[str, Any]) -> None:
        ticker = payload.get("code") or payload.get("symbol")
        if not ticker:
            return
        ts = payload.get("trade_timestamp") or payload.get("timestamp") or int(time.time() * 1000)
        try:
            ts_val = float(ts)
        except Exception:
            ts_val = float(time.time() * 1000)
        minute_ts = int(ts_val // 60000 * 60000)
        price = float(payload.get("trade_price") or payload.get("price") or 0.0)
        if price <= 0:
            return
        state = self._candle_state.get(ticker)
        if not state or state.get("minute") != minute_ts:
            if state:
                self._push_candle(ticker, 'minute1', {
                    "ticker": ticker,
                    "timestamp": state["minute"],
                    "open": state["open"],
                    "high": state["high"],
                    "low": state["low"],
                    "close": state["close"],
                    "volume": state["volume"],
                })
                self._emit_to_timeframes(ticker, state["minute"], state)
            self._candle_state[ticker] = {
                "minute": minute_ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
            }
            self._last_acc_volume[ticker] = float(payload.get("acc_trade_volume", 0) or 0)
            return
        if price > state["high"]:
            state["high"] = price
        if price < state["low"]:
            state["low"] = price
        state["close"] = price
        acc_vol = float(payload.get("acc_trade_volume", 0) or 0)
        prev_acc = self._last_acc_volume.get(ticker, 0.0)
        if acc_vol >= prev_acc:
            state["volume"] += acc_vol - prev_acc
        else:
            state["volume"] += acc_vol
        self._last_acc_volume[ticker] = acc_vol

    def _store_to_redis(self, payload: Dict[str, Any]) -> None:
        if self.redis_client is None:
            return
        try:
            code = payload.get("code") or payload.get("symbol")
            if not code:
                return
            key_base = f"ws:{payload.get('type', 'unknown')}:{code}"
            self.redis_client.set(key_base, json.dumps(payload))
            if payload.get("type") == "trade":
                list_key = f"ws:trades:{code}"
                self.redis_client.lpush(list_key, json.dumps(payload))
                self.redis_client.ltrim(list_key, 0, 200)
            if payload.get("type") == "ticker":
                self._aggregate_candle(payload)
        except Exception as exc:
            log.warning(f"Redis write failed for websocket payload: {exc}")

    def _record_exec_history(self, payload: Dict[str, Any]) -> None:
        if payload.get("type") != "order":
            return
        entry = {
            "ts": float(payload.get("timestamp", payload.get("trade_timestamp", time.time())) / 1000.0)
            if payload.get("timestamp")
            else time.time(),
            "symbol": payload.get("code") or payload.get("symbol"),
            "price": payload.get("price") or payload.get("order_price") or 0,
            "size": payload.get("trade_volume", payload.get("volume") or 0),
            "side": payload.get("side") or payload.get("order_side") or payload.get("ask_bid"),
            "order_id": payload.get("uuid") or payload.get("order_id"),
        }
        side = (entry.get("side") or "").lower()
        symbol = entry.get("symbol")
        avg_price = payload.get("avg_price") or payload.get("avg_buy_price") or payload.get("order_price") or payload.get("price")
        try:
            avg_price_val = float(avg_price) if avg_price is not None else 0.0
        except Exception:
            avg_price_val = 0.0

        entry_price_value = 0.0
        if side in ("bid", "buy", "매수"):
            self._entry_prices[symbol] = avg_price_val or self._entry_prices.get(symbol, 0.0)
        else:
            entry_price_value = self._entry_prices.get(symbol, avg_price_val)
        entry["entry_price"] = entry_price_value or 0.0
        if entry["symbol"]:
            self.history_store.record(entry)

    def _on_message(self, _, message: str) -> None:
        try:
            payloads = json.loads(message)
            if isinstance(payloads, list):
                for payload in payloads:
                    self._handle_payload(payload)
            elif isinstance(payloads, dict):
                self._handle_payload(payloads)
        except Exception as exc:
            log.warning(f"Failed to parse websocket message: {exc}")

    def _handle_payload(self, payload: Dict[str, Any]) -> None:
        self._store_to_redis(payload)
        self._record_exec_history(payload)

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

    def _on_error(self, _, error: Any) -> None:
        log.warning(f"Websocket listener error: {error}")

    def _on_close(self, _, close_status_code, close_msg) -> None:
        log.info(f"Websocket connection closed (code={close_status_code} msg={close_msg})")

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        self._ws = ws
        try:
            ws.send(self._payload_for_subscription())
        except Exception as exc:
            log.warning(f"Failed to send websocket subscription: {exc}")

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
                log.warning(f"Websocket listener restart: {exc}")
            time.sleep(2)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Websocket listener started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2)
        log.info("Websocket listener stopped.")
