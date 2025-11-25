# Upbit Spot AI Auto-Trading Bot 🤖

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

본 프로젝트는 업비트(Upbit) API와 **최신 생성형 AI(OpenAI GPT-5.1, Google Gemini 2.5)**를 결합한 하이브리드 자동매매 시스템입니다.

기존의 단순한 보조지표 매매 방식의 한계를 극복하기 위해, 기술적 지표로 1차 필터링을 거친 후 **두 개의 AI 모델이 합의(Ensemble)**했을 때만 매매를 진행하는 보수적이고 강력한 전략을 사용합니다.

> **⚠️ 중요: 투자 위험 경고**
> 본 프로그램은 투자를 보조하는 도구일 뿐이며, 수익을 보장하지 않습니다. 투자 결정에 따른 모든 책임은 본인에게 있습니다. 실제 자금을 투입하기 전에 충분한 백테스팅과 소액 테스트를 진행하십시오.

## 🚀 주요 기능

-   **하이브리드 전략**: 기술적 분석(RSI) + AI 판단(LLM)의 2단계 검증 시스템.
-   **AI 앙상블(Ensemble)**: **OpenAI(GPT-4o)**와 **Google(Gemini 1.5 Flash)** 두 모델에게 동시에 차트 분석을 의뢰하고, 두 AI의 의견이 일치할 때만 진입하는 '만장일치' 전략 구현.
-   **비용 효율성**: 매 틱마다 AI를 호출하지 않고, 기술적 지표(RSI)가 유효한 신호를 보낼 때만 AI에게 질문하여 API 비용을 절감합니다.
-   **시장가 매매**: 업비트의 시장가 매수/매도 기능을 사용하여 즉각적인 체결을 보장합니다.
-   **상세한 로깅**: 매매 판단의 근거, AI의 답변, 자산 변동 내역을 로그 파일로 상세히 기록합니다.

## 📂 프로젝트 구조

```
/upbit-ai-trader
|
├── bot.py             # (메인) 봇 실행 파일 (기술적 지표와 AI 결과 종합)
├── upbit_api.py       # (모듈) 업비트 API 연동 및 주문 처리
├── strategy.py        # (모듈) 1차 필터용 기술적 지표(RSI) 계산
├── ai_analyst.py      # (모듈) OpenAI 및 Gemini API 연동, 앙상블 로직 [NEW]
├── config.py          # (설정) API 키, 매매 전략, 앙상블 설정
├── logger.py          # (모듈) 로그 시스템 설정
├── requirements.txt   # (설치) 프로젝트 의존성 라이브러리
└── README.md          # 프로젝트 설명서
```

## 🛠 시작하기

### 1. 사전 준비

-   Python 3.8 이상
-   업비트 계정 및 API Key (`자산조회`, `주문조회`, `주문하기` 권한)
-   OpenAI API Key (유료 계정 권장)
-   Google Gemini API Key (AI Studio에서 발급)

### 2. 설치

1.  **저장소 복제 및 이동**
    ```bash
    git clone https://github.com/your-username/upbit-ai-trader.git
    cd upbit-ai-trader
    ```

2.  **가상 환경 생성 (권장)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **필요 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

### 3. 설정 (`config.py`)

프로젝트 루트에 `config.py` 파일을 생성하고 아래 내용을 참고하여 키와 설정을 입력합니다.

```python
# config.py

# --- API Keys ---
UPBIT_ACCESS_KEY = "YOUR_UPBIT_ACCESS_KEY"
UPBIT_SECRET_KEY = "YOUR_UPBIT_SECRET_KEY"

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
OPENAI_MODEL = "gpt-5.1-nano"  # 사용 가능한 최신 모델로 변경 가능

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash" # 사용 가능한 최신 모델로 변경 가능

# --- Strategy Settings ---
MARKET = "KRW-BTC"       # 매매할 코인
TIMEFRAME = "minute5"    # 캔들 기준 (e.g., "minute5", "minute30", "day")
TRADE_AMOUNT_KRW = 6000  # 1회 매수 금액 (KRW, 업비트 최소 주문 금액 이상)

# --- 1차 필터 (Technical Indicator) ---
RSI_PERIOD = 14
RSI_OVERSOLD = 40        # 이 값 이하면 '매수' 기회로 포착 -> AI에게 질문
RSI_OVERBOUGHT = 70      # 이 값 이상이면 '매도' 기회로 포착 -> AI에게 질문

# --- 2차 필터 (AI Ensemble) ---
# 'UNANIMOUS': 두 AI가 모두 동의해야 매매 (안전)
# 'ANY': 둘 중 하나라도 동의하면 매매 (공격적)
ENSEMBLE_STRATEGY = "UNANIMOUS"
```

### 4. 실행

```bash
python bot.py
```

## 🧠 작동 로직

1.  **데이터 수집**: `upbit_api.py`가 업비트에서 지정된 마켓의 캔들 데이터(OHLCV)를 주기적으로 가져옵니다.
2.  **데이터 가공**: `strategy.py`가 수집된 데이터를 바탕으로 RSI, 볼린저 밴드, 이동평균선 등 기술적 보조지표를 계산합니다.
3.  **1차 필터 (기술적 분석)**: `strategy.py`가 RSI 지표를 분석하여 1차 신호를 생성합니다.
    -   `RSI < 40` (과매도 구간): "매수(BUY)" 신호 발생 가능성 감지
    -   `RSI > 70` (과매수 구간): "매도(SELL)" 신호 발생 가능성 감지
4.  **2차 필터 (AI 앙상블 분석)**: 1차 신호가 발생한 경우에만 `ai_analyst.py`가 작동합니다.
    -   최근 차트 데이터와 보조지표를 AI가 이해하기 쉬운 JSON 형태로 변환합니다.
    -   변환된 데이터를 **OpenAI(GPT-4o)**와 **Google(Gemini)**에게 동시에 전송하여 시장 상황 분석을 요청합니다.
    -   각 AI는 시장을 분석하고 `"BUY"`, `"SELL"`, `"HOLD"` 중 하나의 의견과 그 근거를 반환합니다.
5.  **최종 판단 및 실행**: `bot.py`가 `ENSEMBLE_STRATEGY` 설정에 따라 AI들의 의견을 종합합니다.
    -   `UNANIMOUS` 설정 시: `RSI(BUY)` + `OpenAI(BUY)` + `Gemini(BUY)` 조건이 모두 충족될 때 최종 매수 주문을 실행합니다.
    -   조건이 하나라도 불만족 시, 거래를 실행하지 않고 관망(HOLD)합니다.

## ⚠️ 추가 주의사항

-   **API 비용**: OpenAI API는 사용량에 따라 과금됩니다. 봇의 루프 간격이나 RSI 민감도를 조절하여 불필요한 API 호출을 관리하세요.
-   **IP 제한**: 업비트 API는 보안을 위해 등록된 IP 주소에서만 요청을 허용합니다. 봇을 실행하는 서버의 IP가 고정되어 있는지 확인하세요.
-   **최소 주문 금액**: 업비트의 최소 주문 금액 정책(예: 5,000 KRW)을 확인하고 `TRADE_AMOUNT_KRW`를 그 이상으로 설정해야 합니다.
