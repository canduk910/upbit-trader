import logging
import sys
from logging.handlers import RotatingFileHandler
import time
import threading
import os

class DedupFilter(logging.Filter):
    """Filter that suppresses duplicate log messages within a time window.
    Identical messages (same level and message text) logged within
    `window_seconds` will be filtered out.
    """
    def __init__(self, name: str = "", window_seconds: float = 2.0):
        super().__init__(name)
        self.window = float(window_seconds)
        self._last: dict[tuple[int, str], float] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record)
        key = (record.levelno, msg)
        now = time.time()
        with self._lock:
            last_ts = self._last.get(key)
            if last_ts is not None and (now - last_ts) < self.window:
                # suppress duplicate
                return False
            # record new timestamp and allow
            self._last[key] = now
            return True

def setup_logger(name='UpbitBotLogger', log_file='trading_bot.log', level=logging.INFO):
    """
    로그 설정
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 이미 핸들러가 설정되어 있으면 중복 추가 방지
    if logger.hasHandlers():
        return logger

    # 포맷터 생성
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)7s] %(filename)22s:%(lineno)4d [%(name)10s - %(funcName)12s] : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 콘솔(Stream) 핸들러
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 2. 파일(RotatingFile) 핸들러
    # 5MB 크기로 3개의 로그 파일 유지
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Deduplication filter to avoid log flooding (window seconds configurable via env)
    try:
        window = float(os.environ.get('LOG_DEDUP_WINDOW', '2.0'))
    except Exception:
        window = 2.0
    dedup = DedupFilter(window_seconds=window)
    logger.addFilter(dedup)

    return logger

# 전역 로거 인스턴스 생성
log = setup_logger()