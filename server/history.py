import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HISTORY_DIR = Path(__file__).resolve().parents[1] / "runtime" / "history"
HISTORY_FILE = HISTORY_DIR / "positions_history.json"
MAX_ENTRIES = int(os.getenv("POSITIONS_HISTORY_MAX_ENTRIES", "720"))
MIN_INTERVAL_SEC = int(os.getenv("POSITIONS_HISTORY_MIN_INTERVAL_SEC", "900"))
ORDER_HISTORY_FILE = HISTORY_DIR / "order_history.json"
ORDER_HISTORY_MAX = int(os.getenv("ORDER_HISTORY_MAX_ENTRIES", "1024"))
AI_HISTORY_FILE = HISTORY_DIR / "ai_decisions.json"
AI_HISTORY_MAX = int(os.getenv("AI_HISTORY_MAX_ENTRIES", "1024"))


def _ensure_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


class HistoryStore:
    def __init__(self, path: Path = HISTORY_FILE, max_entries: int = MAX_ENTRIES, min_interval: int = MIN_INTERVAL_SEC):
        self.path = path
        self.max_entries = max_entries
        self.min_interval = min_interval
        self.lock = threading.Lock()
        _ensure_dir()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return []

    def _save(self, data: List[Dict[str, Any]]) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False)
        except Exception:
            pass

    def record_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        now = snapshot.get("ts") or time.time()
        snapshot["ts"] = float(now)
        with self.lock:
            data = self._load()
            last_ts = float(data[-1].get("ts", 0)) if data else 0
            if now - last_ts < self.min_interval:
                return False
            data.append(snapshot)
            if len(data) > self.max_entries:
                data = data[-self.max_entries :]
            self._save(data)
        return True

    def get_history(self, since: Optional[float] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self.lock:
            data = self._load()
        if since is not None:
            data = [item for item in data if float(item.get("ts", 0)) >= since]
        if limit is not None and limit < len(data):
            data = data[-limit:]
        return data


history_store = HistoryStore()


class OrderHistoryStore:
    def __init__(self, path: Path = ORDER_HISTORY_FILE, max_entries: int = ORDER_HISTORY_MAX):
        self.path = path
        self.max_entries = max_entries
        self.lock = threading.Lock()
        _ensure_dir()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return []

    def _save(self, data: List[Dict[str, Any]]) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False)
        except Exception:
            pass

    def record(self, entry: Dict[str, Any]) -> None:
        entry.setdefault("ts", time.time())
        with self.lock:
            data = self._load()
            data.append(entry)
            if len(data) > self.max_entries:
                data = data[-self.max_entries :]
            self._save(data)


order_history_store = OrderHistoryStore()


class AIHistoryStore:
    def __init__(self, path: Path = AI_HISTORY_FILE, max_entries: int = AI_HISTORY_MAX):
        self.path = path
        self.max_entries = max_entries
        self.lock = threading.Lock()
        _ensure_dir()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return []

    def _save(self, data: List[Dict[str, Any]]) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False)
        except Exception:
            pass

    def record(self, entry: Dict[str, Any]) -> None:
        entry.setdefault("ts", time.time())
        with self.lock:
            data = self._load()
            data.append(entry)
            if len(data) > self.max_entries:
                data = data[-self.max_entries :]
            self._save(data)

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self.lock:
            data = self._load()
        if limit is not None and limit > 0 and len(data) > limit:
            data = data[-limit:]
        return data


ai_history_store = AIHistoryStore()
