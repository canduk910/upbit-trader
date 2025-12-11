# 🪙 Upbit 자동 트레이딩 봇

> **업비트(Upbit) 거래소 전용 자동매매 시스템**  
> 변동성 돌파, RSI, 듀얼 모멘텀 전략 + OpenAI/Gemini AI 앙상블 자문 기반 코인 자동매매

이 프로젝트는 업비트 현물 계좌에서 자동으로 코인을 매매하는 트레이딩 봇입니다.  
실시간 WebSocket으로 시세를 받고, 기술적 분석 전략과 AI 자문을 결합하여 매매 결정을 내립니다.

---

## 📋 목차
1. [프로젝트 목적](#-프로젝트-목적)
2. [핵심 기능](#-핵심-기능)
3. [시스템 아키텍처](#-시스템-아키텍처)
4. [프로세스 구조 (서비스 구성)](#-프로세스-구조-서비스-구성)
5. [기술 스택](#-기술-스택)
6. [디렉토리 구조](#-디렉토리-구조)
7. [소스 파일 설명](#-소스-파일-설명)
8. [설치 및 실행 방법](#-설치-및-실행-방법)
9. [설정 파일 (config.json)](#-설정-파일-configjson)
10. [주요 용어 설명](#-주요-용어-설명)

---

## 🎯 프로젝트 목적

| 목적 | 설명 |
|------|------|
| **자동매매** | 사람이 24시간 시장을 지켜보지 않아도 봇이 전략에 따라 자동으로 매수/매도 |
| **AI 자문 통합** | OpenAI(GPT)와 Gemini 두 AI 모델에게 매매 신호를 검증받아 신뢰도 향상 |
| **리스크 관리** | 켈리 공식, 손절가/익절가 자동 계산으로 자금 관리 |
| **실시간 모니터링** | Streamlit 웹 대시보드에서 계좌 상태, AI 자문 히스토리, 차트 확인 |
| **학습/연구용** | 전략, 자금관리, AI 프롬프트를 바꿔가며 백테스트 및 실험 가능 |

---

## ✨ 핵심 기능

### 1. 매매 전략 (Strategy)
```
┌─────────────────────────────────────────────────────────────┐
│  전략명              │  설명                                 │
├─────────────────────────────────────────────────────────────┤
│  VolatilityBreakout │  변동성 돌파: 전일 변동폭×k 이상 상승 시 매수   │
│  RSI                │  RSI 지표가 과매도/과매수 구간 탈출 시 매매      │
│  DualMomentum       │  절대/상대 모멘텀 조합으로 추세 추종            │
└─────────────────────────────────────────────────────────────┘
```

### 2. AI 앙상블 자문
- **OpenAI (GPT)** + **Google Gemini** 두 모델에 병렬로 질의
- 기술적 신호를 AI가 재검증하여 최종 매매 결정
- **앙상블 전략**:
  - `UNANIMOUS`: 두 AI 모두 동의해야 실행
  - `MAJORITY`: 과반수 동의 시 실행
  - `AVERAGE`: 신뢰도(confidence) 평균이 임계값 이상이면 실행
- 매수/매도 각각 다른 앙상블 전략 적용 가능

### 3. 자금 관리 (Money Manager)
- **켈리 공식(Kelly Criterion)**: 승률과 손익비로 최적 투자 비율 계산
- **포지션 사이징**: 손절가 기반으로 리스크 한도 내 매수 수량 산정
- **손절가/익절가 자동 계산**: AI 응답의 price_plan 활용

### 4. 실시간 데이터 수집
- **WebSocket 리스너**: 업비트 공개/비공개 WebSocket으로 실시간 시세/체결 수신
- **Redis 캐시**: 캔들 데이터를 Redis에 저장하여 중복 API 호출 방지
- **REST API 폴백**: WebSocket 캐시 미스 시 REST API로 조회

### 5. 웹 대시보드 (UI)
- **설정 편집**: 전략, 마켓, 주문금액, 켈리 파라미터 실시간 수정
- **자동매매봇 관리**: 봇 시작/중지, 루프 간격 조절
- **AI 자문 리포트**: 최신 AI 응답 비교, 입력 캔들 차트, 자문 히스토리
- **종목 스크리닝**: 변동성 상위 종목 탐색
- **계좌 현황**: 잔고, 포지션, 손익 확인

---

## 🏗 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        사용자 (브라우저)                              │
│                              ↓                                       │
│                    ┌─────────────────┐                               │
│                    │   Streamlit UI   │ ← 포트 8501                  │
│                    │ (ui_dashboard.py)│                               │
│                    └────────┬─────────┘                               │
│                             │ HTTP (API 호출)                         │
│                             ↓                                         │
│                    ┌─────────────────┐                               │
│                    │   FastAPI 서버   │ ← 포트 8000                  │
│                    │    (api.py)      │                               │
│                    └────────┬─────────┘                               │
│                             │                                         │
│         ┌───────────────────┼───────────────────┐                    │
│         ↓                   ↓                   ↓                    │
│ ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│ │  Trading Bot  │  │  WS Listener  │  │  WS Listener  │             │
│ │   (bot.py)    │  │   (Public)    │  │   (Private)   │             │
│ └───────┬───────┘  └───────┬───────┘  └───────┬───────┘             │
│         │                  │                  │                      │
│         └──────────────────┴──────────────────┘                      │
│                             │                                         │
│                    ┌────────┴────────┐                               │
│                    │      Redis       │ ← 캔들/시세 캐시             │
│                    │   (redis:6379)   │                               │
│                    └─────────────────┘                               │
│                             │                                         │
│         ┌───────────────────┼───────────────────┐                    │
│         ↓                   ↓                   ↓                    │
│ ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │
│ │    OpenAI     │  │    Gemini     │  │   Upbit API   │             │
│ │   (GPT API)   │  │  (Google AI)  │  │   (거래소)     │             │
│ └───────────────┘  └───────────────┘  └───────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 프로세스 구조 (서비스 구성)

Docker Compose로 5개의 서비스가 실행됩니다:

| 서비스 | 컨테이너명 | 역할 | 포트 |
|--------|-----------|------|------|
| **redis** | upbit-trader-redis | 캔들/시세 데이터 캐시 저장소 | 6379 |
| **backend** | upbit-trader-backend | FastAPI 서버 (REST API 제공) | 8000 |
| **ui** | upbit-trader-ui | Streamlit 웹 대시보드 | 8501 |
| **bot** | upbit-trader-bot | 자동매매 메인 루프 실행 | - |
| **ws_listener_public** | upbit-trader-ws-listener-public | 공개 WebSocket (시세 수신) | - |
| **ws_listener_private** | upbit-trader-ws-listener-private | 비공개 WebSocket (체결 수신) | - |

### 데이터 흐름

```
[업비트 거래소]
      │
      ├──── WebSocket ────→ ws_listener_public ──→ Redis 캔들 캐시
      │                                                  │
      ├──── WebSocket ────→ ws_listener_private ─→ 체결 히스토리
      │                                                  │
      └──── REST API ←───────────────────────────────────┤
                                                         │
                                                         ↓
                                                    Trading Bot
                                                         │
                          ┌──────────────────────────────┤
                          ↓                              ↓
                   [전략 신호 생성]              [AI 앙상블 자문]
                          │                              │
                          └──────────────┬───────────────┘
                                         ↓
                                  [매매 결정: BUY/SELL/HOLD]
                                         │
                                         ↓
                             ┌──────────────────────┐
                             │   Order Executor     │
                             │   (주문 실행기)        │
                             └──────────┬───────────┘
                                        │
                                        ↓
                              [업비트 주문 API 호출]
```

---

## 🛠 기술 스택

| 분류 | 기술 | 용도 |
|------|------|------|
| **언어** | Python 3.11+ | 전체 백엔드/봇/UI 구현 |
| **웹 프레임워크** | FastAPI | REST API 서버 |
| **UI 프레임워크** | Streamlit | 웹 대시보드 |
| **데이터 분석** | pandas, ta | 캔들 데이터 처리, 기술적 지표 계산 |
| **시각화** | Plotly | 캔들차트, RSI, 손익 차트 |
| **AI/LLM** | OpenAI SDK, Google Generative AI | AI 자문 (GPT, Gemini) |
| **캐시** | Redis | 실시간 시세 캐시 |
| **WebSocket** | websocket-client | 업비트 실시간 데이터 수신 |
| **거래소 API** | pyupbit, requests, PyJWT | 업비트 REST/WebSocket API 연동 |
| **컨테이너** | Docker, Docker Compose | 서비스 배포 및 관리 |
| **환경 관리** | python-dotenv | API 키 등 민감 정보 관리 |

---

## 📁 디렉토리 구조

```
/upbit-trader
│
├── server/                      # 백엔드 핵심 코드
│   ├── __init__.py
│   ├── api.py                   # FastAPI REST API 엔드포인트
│   ├── bot.py                   # 자동매매 봇 메인 로직
│   ├── ai_analyst.py            # AI 앙상블 분석기 (OpenAI + Gemini)
│   ├── strategy.py              # 매매 전략 (RSI, 변동성돌파, 모멘텀)
│   ├── money_manager.py         # 자금 관리 (켈리 공식)
│   ├── order_executor.py        # 주문 실행기 (큐 기반 비동기 처리)
│   ├── upbit_api.py             # 업비트 API 래퍼
│   ├── config.py                # 설정 파일 로드/저장
│   ├── logger.py                # 로깅 설정
│   ├── history.py               # 히스토리 저장 (포지션, 주문, AI)
│   ├── ws_listener_base.py      # WebSocket 리스너 베이스 클래스
│   ├── ws_listener_public.py    # 공개 WebSocket (시세)
│   ├── ws_listener_private.py   # 비공개 WebSocket (체결)
│   └── .env                     # API 키 (gitignore 대상)
│
├── ui/                          # Streamlit 대시보드
│   └── ui_dashboard.py          # 웹 UI 전체 구현
│
├── runtime/                     # 런타임 데이터 (설정, 히스토리)
│   ├── config.json              # 봇 설정 파일
│   └── history/                 # JSON 히스토리 저장소
│       ├── ai_decisions.json    # AI 자문 기록
│       ├── order_history.json   # 주문 실행 기록
│       ├── positions_history.json # 포지션 변화 기록
│       ├── exec_history.json    # WebSocket 체결 기록
│       └── ws_stats.json        # WebSocket 통계
│
├── logs/                        # 로그 파일 저장소
│   ├── trading_bot.log
│   ├── ui.log
│   ├── ws_listener_public.log
│   └── ws_listener_private.log
│
├── scripts/
│   ├── run_dev.sh               # 로컬 개발용 실행 스크립트
│   └── docker-entrypoint.sh     # Docker 엔트리포인트
│
├── docker-compose.yml           # Docker 서비스 정의
├── docker-compose.override.yml  # 개발용 볼륨 마운트
├── Dockerfile.backend           # 백엔드/봇 이미지
├── Dockerfile.bot               # 봇 전용 이미지
├── Dockerfile.ui                # UI 전용 이미지
├── requirements.txt             # Python 의존성
└── README.md                    # 이 문서
```

---

## 📚 소스 파일 설명

### 🔵 server/ (백엔드)

#### `api.py` - FastAPI REST API
```python
# 주요 엔드포인트:
GET  /health              # 서버 상태 확인
GET  /config              # 현재 설정 조회
POST /config              # 설정 저장
POST /reload              # 설정 리로드
GET  /balances            # 계좌 잔고 조회
GET  /positions           # 보유 포지션 조회
POST /klines_batch        # 캔들 데이터 배치 조회
GET  /screen/volatility_top  # 변동성 상위 종목
GET  /ws/status           # WebSocket 상태
GET  /bot/status          # 봇 상태
POST /bot/control         # 봇 제어 (시작/중지)
GET  /ai/history          # AI 자문 히스토리
```

#### `bot.py` - 자동매매 봇
```python
class TradingBot:
    """
    메인 루프:
    1. 설정 변경 감지 → 리로드
    2. Redis 캐시에서 캔들 조회 (없으면 REST)
    3. 전략으로 기술적 신호 생성 (BUY/SELL/HOLD)
    4. BUY/SELL 신호 시 AI 앙상블에 자문 요청
    5. AI 승인 시 OrderExecutor로 주문 제출
    6. AI 거절 시 쿨다운 설정 (N개 캔들 동안 스킵)
    """
```

#### `ai_analyst.py` - AI 앙상블 분석기
```python
class EnsembleAnalyzer:
    """
    OpenAI(GPT)와 Gemini에 병렬로 질의하여
    TradingContext → TradingDecision JSON 응답을 받음.
    
    앙상블 전략:
    - UNANIMOUS: 모두 동의
    - MAJORITY: 과반수 동의
    - AVERAGE: 신뢰도 평균 임계값 이상
    
    출력 포함 정보:
    - decision: BUY/SELL/HOLD
    - confidence: 0.0~1.0
    - price_plan: entry_price, stop_loss_price, take_profit_price, market_factor
    """
```

#### `strategy.py` - 매매 전략
```python
class RSIStrategy(Strategy):
    """RSI가 과매도(30) 상향돌파 → BUY, 과매수(70) 하향돌파 → SELL"""

class VolatilityBreakoutStrategy(Strategy):
    """
    당일시가 + (전일고가-전일저가)*k 돌파 → BUY
    손절 조건: ATR 기반, 트레일링 스톱, 전일 저가 이탈
    """

class DualMomentumStrategy(Strategy):
    """절대 모멘텀 + 상대 모멘텀 조합"""
```

#### `money_manager.py` - 자금 관리
```python
class KellyCriterionManager:
    """
    켈리 공식: f* = (p*b - (1-p)) / b
    - p: 승률
    - b: 손익비
    
    get_position_size():
    - 리스크 기반 수량 = 허용손실 / (매수가 - 손절가)
    - 켈리 기반 금액 = 총자산 × 켈리비율
    - 최종 = min(리스크기반, 켈리기반) × market_factor
    """
```

#### `order_executor.py` - 주문 실행기
```python
class OrderExecutor:
    """
    별도 스레드에서 큐 기반으로 주문 처리
    - OrderRequest 객체 수신
    - 최소 주문금액, 잔고 검증
    - 업비트 API로 실제 주문 실행
    - order_history.json에 기록
    """
```

#### `ws_listener_public.py` / `ws_listener_private.py`
```python
# Public: ticker 메시지로 실시간 시세 수신 → Redis 캔들 캐시
# Private: myOrder 메시지로 체결 알림 수신 → exec_history.json
```

---

### 🟢 ui/ (대시보드)

#### `ui_dashboard.py` - Streamlit 웹 UI
```
페이지 구성:
├── 📊 설정 편집         # 전략, 마켓, 주문금액, 켈리 파라미터
├── 🤖 자동매매봇 관리    # 봇 시작/중지, 루프 간격
├── 🧠 AI 자문 리포트    # OpenAI/Gemini 응답 비교, 캔들 차트
├── 📈 종목 스크리닝     # 변동성 상위 종목 탐색
├── 💰 원화 잔고 현황    # KRW 잔고 조회
├── 📦 보유 포지션 분석   # 포지션 테이블, 손익 차트
└── 🔌 WebSocket 상태    # 연결 상태, 수신 통계
```

---

### 🟡 runtime/ (런타임 데이터)

#### `config.json` - 봇 설정
```json
{
  "strategy_name": "VolatilityBreakout",  // 전략 선택
  "market": "KRW-BTC",                    // 거래 마켓
  "timeframe": "minute15",                // 캔들 시간단위
  "candle_count": 200,                    // 분석할 캔들 수
  "order_settings": {
    "min_order_amount": 5500,             // 최소 주문금액
    "trade_amount_krw": 6000              // 1회 거래금액
  },
  "use_kelly_criterion": true,
  "kelly_criterion": {
    "win_rate": 0.65,                     // 승률 65%
    "payoff_ratio": 1.2,                  // 손익비 1.2
    "fraction": 0.5                       // Half Kelly
  },
  "ai_ensemble": {
    "buy_strategy": "UNANIMOUS",          // 매수 시 모두 동의 필요
    "sell_strategy": "MAJORITY"           // 매도 시 과반수
  }
}
```

---

## 🚀 설치 및 실행 방법

### 사전 준비
1. **업비트 API 키 발급**: [업비트 개발자센터](https://upbit.com/mypage/open_api_management)
2. **OpenAI API 키** (선택): [OpenAI Platform](https://platform.openai.com)
3. **Google AI API 키** (선택): [Google AI Studio](https://aistudio.google.com)

### 방법 1: Docker Compose (권장)

```bash
# 1. 저장소 클론
git clone https://github.com/your-repo/upbit-trader.git
cd upbit-trader

# 2. 환경 변수 설정
cp server/.env.example server/.env
# server/.env 파일에 API 키 입력:
# UPBIT_ACCESS_KEY=your_access_key
# UPBIT_SECRET_KEY=your_secret_key
# OPENAI_API_KEY=your_openai_key
# GEMINI_API_KEY=your_gemini_key

# 3. Docker 빌드 및 실행
docker compose up --build -d

# 4. 접속
# - UI: http://localhost:8501
# - API: http://localhost:8000
# - API 문서: http://localhost:8000/docs
```

### 방법 2: 로컬 실행

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. Redis 실행 (별도 터미널)
docker run -d -p 6379:6379 redis:7-alpine

# 4. 환경 변수 설정
cp server/.env.example server/.env
# API 키 입력

# 5. 개발 서버 실행
./scripts/run_dev.sh all
# 또는 개별 실행:
# ./scripts/run_dev.sh backend
# ./scripts/run_dev.sh ui
```

---

## ⚙️ 설정 파일 (config.json)

| 키 | 타입 | 설명 |
|----|------|------|
| `strategy_name` | string | `VolatilityBreakout`, `RSI`, `DualMomentum` |
| `market` | string | 거래 마켓 (예: `KRW-BTC`) |
| `timeframe` | string | 캔들 단위 (예: `minute15`, `day`) |
| `candle_count` | int | 분석할 캔들 개수 |
| `loop_interval_sec` | int | 봇 루프 간격(초) |
| `bot_enabled` | bool | 봇 활성화 여부 |
| `bot_interval_sec` | float | 봇 체크 간격(초) |
| `order_settings.min_order_amount` | int | 최소 주문금액 |
| `order_settings.trade_amount_krw` | int | 1회 거래금액 |
| `use_kelly_criterion` | bool | 켈리 공식 사용 여부 |
| `kelly_criterion.win_rate` | float | 승률 (0~1) |
| `kelly_criterion.payoff_ratio` | float | 손익비 |
| `kelly_criterion.fraction` | float | 켈리 비중 (0~1) |
| `strategy_params.VolatilityBreakout.k_value` | float | 변동성 계수 k |
| `strategy_params.VolatilityBreakout.target_vol_pct` | float | 목표 변동성(%) |
| `strategy_params.RSI.period` | int | RSI 기간 |
| `strategy_params.RSI.oversold` | int | 과매도 기준 |
| `strategy_params.RSI.overbought` | int | 과매수 기준 |
| `ai_ensemble.buy_strategy` | string | 매수 앙상블: `UNANIMOUS`/`MAJORITY`/`AVERAGE` |
| `ai_ensemble.sell_strategy` | string | 매도 앙상블 |
| `ai_ensemble.average_threshold` | float | AVERAGE 전략의 임계값 |

---

## 📖 주요 용어 설명

| 용어 | 설명 |
|------|------|
| **캔들 (Candle/Kline)** | 특정 시간 동안의 시가/고가/저가/종가/거래량을 담은 봉차트 데이터 |
| **RSI** | 상대강도지수. 0~100 사이 값으로 과매수(70↑)/과매도(30↓) 판단 |
| **변동성 돌파** | 전일 변동폭의 일정 비율(k) 이상 상승 시 매수하는 전략 |
| **켈리 공식** | 승률과 손익비로 최적 베팅 비율을 계산하는 수학 공식 |
| **손절가 (Stop Loss)** | 손실을 제한하기 위해 자동 매도하는 가격 |
| **익절가 (Take Profit)** | 목표 이익 달성 시 자동 매도하는 가격 |
| **앙상블** | 여러 모델의 결과를 종합하는 방식 |
| **WebSocket** | 서버와 실시간 양방향 통신을 위한 프로토콜 |

---

## ⚠️ 주의사항

1. **실제 자금 투자 시 주의**: 이 프로젝트는 학습/연구 목적입니다. 실제 투자 손실에 대한 책임은 사용자에게 있습니다.
2. **API 키 보안**: `.env` 파일을 절대 공개 저장소에 커밋하지 마세요.
3. **업비트 API 제한**: 초당 요청 수 제한이 있으므로 prefetch 설정을 적절히 조정하세요.
4. **AI 비용**: OpenAI/Gemini API는 유료입니다. 사용량을 모니터링하세요.

---
*마지막 업데이트: 2025년 12월 12일*

