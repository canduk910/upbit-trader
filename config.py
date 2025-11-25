# config.py
# 중요: 이 파일은 .gitignore 에 추가하여 절대로 외부에 노출되지 않도록 관리하세요.

# 업비트 API 키
UPBIT_ACCESS_KEY = "YOUR_ACCESS_KEY"    # 여기에 발급받은 Access Key를 입력하세요.
UPBIT_SECRET_KEY = "YOUR_SECRET_KEY"    # 여기에 발급받은 Secret Key를 입력하세요.

# --- 매매 설정 ---
MARKET = "KRW-BTC"       # 매매할 마켓 (예: KRW-BTC, KRW-ETH, KRW-XRP)
TIMEFRAME = "minute5"    # 캔들 봉 기준 (예: minute1, minute3, minute5, minute10, minute15, minute30, minute60, minute240, day, week, month)
CANDLE_COUNT = 200       # 전략 계산 시 사용할 캔들 수

# --- 주문 설정 ---
# 업비트 시장가 매수(ord_type='price')는 KRW 금액을 기준으로 주문합니다.
TRADE_AMOUNT_KRW = 5100  # 1회 매수 주문 금액 (업비트 최소 주문 금액 5,000 KRW 이상)

# --- 전략 파라미터 (RSI 예시) ---
# 간단한 RSI 전략 기준값
RSI_PERIOD = 14          # RSI 계산 기간
RSI_OVERSOLD = 30        # 과매도 기준
RSI_OVERBOUGHT = 70      # 과매수 기준

# --- 봇 실행 설정 ---
LOOP_INTERVAL_SEC = 5    # 봇이 다음 로직을 실행하기까지 대기하는 시간 (초)