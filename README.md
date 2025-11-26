# Upbit Spot AI Auto-Trading Bot 🤖

## 프로젝트 소개

이 프로젝트는 업비트(Upbit) 거래소에서 자동으로 암호화폐를 사고팔 수 있는 트레이딩 봇입니다. 
주요 목표는 다음과 같습니다:

- **자동화된 거래**: 사람이 직접 매매하지 않아도, 설정한 전략에 따라 자동으로 거래를 실행합니다.
- **다양한 전략 지원**: RSI, 변동성 돌파, 듀얼 모멘텀 등 여러 전략을 선택할 수 있습니다.
- **자금 관리**: 켈리 공식(Kelly Criterion)을 사용해 자금을 효율적으로 관리합니다.
- **사용자 친화적 설정**: JSON 파일이나 간단한 UI를 통해 설정을 쉽게 변경할 수 있습니다.

이 봇은 초보자도 쉽게 사용할 수 있도록 설계되었으며, 개발자라면 원하는 대로 커스터마이징할 수도 있습니다.

---

## 주요 기능

- **전략 기반 거래**: 다양한 트레이딩 전략을 지원하며, 필요에 따라 새로운 전략을 추가할 수 있습니다.
- **켈리 공식 적용**: 자금 관리 전략으로 손실을 최소화하고 수익을 극대화합니다.
- **실시간 설정 변경**: 설정 파일(runtime/config.json)을 수정하면 봇이 자동으로 변경 사항을 반영합니다.
- **REST API**: FastAPI를 사용해 설정을 변경하거나 봇을 제어할 수 있습니다.
- **Streamlit UI**: 간단한 웹 인터페이스를 통해 설정을 편집하고 저장할 수 있습니다.

---

## 프로젝트 구조

```
/upbit-trader
├── server/                # 서버 코드 및 주요 로직
│   ├── bot.py             # 트레이딩 봇의 핵심 로직
│   ├── config.py          # 설정 파일 로드 및 저장
│   ├── api.py             # REST API 엔드포인트
│   ├── strategy.py        # 트레이딩 전략 구현
│   ├── money_manager.py   # 켈리 공식 기반 자금 관리
│   └── ... 기타 파일들
├── runtime/               # 런타임 설정 파일 저장소
│   └── config.json        # 봇의 설정 파일
├── ui/                    # 사용자 인터페이스
│   └── streamlit_app.py   # Streamlit 기반 설정 편집 UI
├── README.md              # 프로젝트 설명서
└── requirements.txt       # 필요한 Python 패키지 목록
```

### 주요 파일 설명

- **server/bot.py**: 트레이딩 봇의 핵심 파일로, 설정에 따라 거래를 실행합니다.
- **runtime/config.json**: 봇의 동작을 제어하는 설정 파일입니다. (예: 전략, 거래 금액 등)
- **ui/streamlit_app.py**: 설정 파일을 편집할 수 있는 간단한 웹 인터페이스입니다.
- **server/api.py**: REST API를 통해 설정을 변경하거나 봇을 제어할 수 있습니다.

---

## 작동 원리

1. **설정 파일 로드**: `runtime/config.json`에서 봇의 동작에 필요한 설정을 불러옵니다.
2. **전략 선택**: 설정에 따라 RSI, 변동성 돌파, 듀얼 모멘텀 등의 전략을 실행합니다.
3. **거래 실행**: 선택한 전략에 따라 업비트 API를 통해 매수/매도를 실행합니다.
4. **자금 관리**: 켈리 공식을 사용해 거래 금액을 동적으로 조정합니다.
5. **설정 변경 감지**: 설정 파일이 변경되면 자동으로 새로운 설정을 반영합니다.

---

## 사용 방법

### 1. 개발 환경 준비

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
# FastAPI 서버 실행
uvicorn server.api:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Streamlit UI 실행

```bash
# Streamlit 기반 설정 편집 UI 실행
streamlit run ui/streamlit_app.py
```

### 4. 봇 실행

```bash
# 트레이딩 봇 실행
python -m server.bot
```

---

## 설정 파일 예시

`runtime/config.json` 파일은 봇의 동작을 제어하는 설정 파일입니다. 주요 필드는 다음과 같습니다:

```json
{
  "strategy_name": "RSI",  // 사용할 전략 (RSI, VolatilityBreakout, DualMomentum)
  "market": "KRW-BTC",    // 거래할 마켓
  "order_settings": {
    "trade_amount_krw": 100000  // 거래 금액 (KRW)
  },
  "use_kelly_criterion": true,  // 켈리 공식 사용 여부
  "kelly_criterion": {
    "win_rate": 0.6,            // 승률
    "payoff_ratio": 2.0,        // 보상비율
    "fraction": 0.5             // 투자 비율
  }
}
```

---

## 자주 묻는 질문 (FAQ)

### Q1. 이 봇은 어떻게 동작하나요?
A1. 설정 파일(runtime/config.json)에 따라 트레이딩 전략을 실행하며, 업비트 API를 통해 자동으로 매수/매도를 수행합니다.

### Q2. 새로운 전략을 추가할 수 있나요?
A2. 네, `server/strategy.py` 파일에 새로운 전략을 구현하고 설정 파일에 추가하면 됩니다.

### Q3. 이 봇은 안전한가요?
A3. 민감 정보(API 키)는 `.env` 파일에 저장되며, 운영 환경에서는 추가적인 보안 조치를 권장합니다.

---

## 향후 작업

- **UI 개선**: 더 직관적인 사용자 인터페이스 제공
- **백테스트 기능 추가**: 과거 데이터를 기반으로 전략 성능 평가
- **추가 전략 구현**: All-Weather, AI 기반 전략 등
- **보안 강화**: 인증 및 접근 제어 기능 추가

---

이 프로젝트에 기여하거나 질문이 있다면 언제든지 연락주세요! 😊
