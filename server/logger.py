import logging
import sys
from logging.handlers import RotatingFileHandler
import time
import threading
import os
from pathlib import Path

# Ensure the logging process uses Korean Standard Time for timestamps
os.environ.setdefault('TZ', 'Asia/Seoul')
try:
    time.tzset()
except Exception:
    pass

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
    # Decide where to write logs; allow overriding via LOG_DIR for container-only placement
    log_dir_env = os.environ.get('LOG_DIR')
    try:
        project_root = Path(__file__).resolve().parents[1]
    except Exception:
        project_root = Path(os.getcwd())

    if log_dir_env:
        logs_dir = Path(log_dir_env)
    else:
        logs_dir = project_root / 'logs'
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logs_dir = Path(os.getcwd())

    # If log_file is an absolute path, keep it; otherwise place it under logs_dir
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = logs_dir / log_path

    file_handler = RotatingFileHandler(
        str(log_path),
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

    # Detect stray log files named 'trading_bot.log' outside the canonical logs directory.
    try:
        # look for files named like the logger filename under project root
        stray_paths = []
        # resolved canonical path of the active log file
        try:
            canonical_log = Path(str(log_path)).resolve()
        except Exception:
            canonical_log = Path(str(log_path))

        for p in project_root.rglob('trading_bot.log'):
            try:
                p_res = p.resolve()
            except Exception:
                p_res = p
            # skip the canonical log file under logs_dir
            if p_res == canonical_log:
                continue
            # skip files inside the logs directory (they are allowed)
            try:
                if logs_dir.resolve() in p_res.parents:
                    continue
            except Exception:
                pass
            stray_paths.append(str(p_res))

        if stray_paths:
            msg = (
                f"Found unexpected 'trading_bot.log' files outside '{logs_dir}': {stray_paths}. "
                "These files are considered abnormal; move or remove them so logs are centralized."
            )
            # Print to stdout for visibility during container startup and also log a warning.
            try:
                print(msg, file=sys.stderr)
            except Exception:
                pass
            try:
                logger.warning(msg)
            except Exception:
                pass
    except Exception:
        # don't fail logger setup because of stray-file check
        pass

    return logger


def get_logger(name='UpbitBotLogger', log_file='trading_bot.log', level=logging.INFO):
    return setup_logger(name=name, log_file=log_file, level=level)

# 전역 로거 인스턴스 생성
log = get_logger()
