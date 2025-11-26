import logging
import sys
from logging.handlers import RotatingFileHandler

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
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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

    return logger

# 전역 로거 인스턴스 생성
log = setup_logger()