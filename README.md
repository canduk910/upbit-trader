# Upbit 자동 트레이딩 봇 — 쉽게 이해하는 안내서

이 문서는 이 프로젝트가 무엇인지, 폴더와 중요한 파일들이 어떤 역할을 하는지, 그리고 로컬과 Docker에서 어떻게 실행하는지 초등학생도 이해할 수 있게 쉽고 친절하게 설명합니다.

요약 (한 문장)
- 이 프로그램은 업비트(Upbit)에서 코인 시세를 읽고 기본 전략 규칙에 따라 언제 사고 팔지 판단해 자동으로 주문을 내는 연습용 트레이딩 봇입니다. UI로 설정을 바꾸고, REST API로 결과를 확인할 수 있습니다.

왜 이 프로젝트가 있나요? (목적)
- 사람이 24시간 시장을 보기 어렵습니다. 간단한 규칙(전략)을 코드로 만들어 자동으로 시장을 관찰하고, 손쉽게 전략과 자금관리를 바꿔 가며 테스트하기 위함입니다.
- 연구/학습/시뮬레이션 목적으로 안전하게 로컬에서 돌려볼 수 있도록 설계했습니다. 실제 매매를 하려면 API 키와 위험 관리가 필요합니다.

핵심 개념을 아주 쉽게 설명
- 전략(strategy): 코인 가격 데이터(캔들)를 보고 ‘사야 한다/팔아야 한다’고 결정하는 규칙입니다.
- 켈리공식(Kelly criterion): 이길 확률과 기대 수익에 근거해 한 번에 투자할 금액을 계산하는 방법입니다.
- 스크리닝(screening): 여러 종목을 모아 놓고 변동성이 큰 종목을 골라내는 기능입니다.
- 워처(watcher): 실시간(폴링)으로 조건을 체크해 이벤트가 발생하면 로그/알림을 남기는 작은 백그라운드 작업입니다.

프로젝트 구조 (중요한 파일과 역할)
```
/upbit-trader
├── server/                # 서버(백엔드) 코드
│   ├── api.py             # FastAPI REST API: 설정, 스크리닝, watcher, klines_batch 등
│   ├── bot.py             # 봇의 메인 루프: 시세 조회 → 전략 평가 → 주문(모의/실제)
│   ├── upbit_api.py       # Upbit와 통신하는 로직(pyupbit 또는 직접 HTTP 사용)
│   ├── strategy.py        # 여러 전략 구현 (VolatilityBreakout, DualMomentum, RSI 등)
│   ├── money_manager.py   # 켈리공식 등 자금관리 로직
│   ├── config.py          # runtime/config.json 읽기/쓰기, 전역 설정
│   └── logger.py          # 로그 설정
├── ui/                    # Streamlit 기반 사용자 인터페이스
│   └── ui_dashboard.py    # 설정 편집, 종목 스크리닝, 워처 제어 페이지
├── runtime/               # 실행 시 사용하는 설정
│   └── config.json        # 기본 실행 설정(전략, 마켓, 파라미터 등)
├── scripts/
│   └── run_dev.sh         # 개발용 실행 스크립트 (백엔드 + UI 동시 실행 등)
├── docker-compose.yml
├── docker-compose.override.yml  # 개발용 바인드 마운트(코드 수정시 재시작 흐름 제어)
├── requirements.txt
└── README.md
```

주요 기능 (짧게)
- 종목 스크리닝: 변동성이 큰 상위 종목을 찾아 보여줍니다.
- 이벤트 감시자(Watcher): 지정한 마켓을 주기적으로 검사해 변동성 돌파 또는 거래량 폭증을 감지합니다.
- UI: Streamlit으로 설정을 편집하고 즉시 서버로 전송할 수 있습니다.
- API: FastAPI로 스크리닝, 배치 klines 조회, watcher 제어, 설정 저장/재로딩 등을 제공합니다.
- 캐시/프리페치: 서버에서 미리 캔들 데이터를 수집(prefetch)해 UI 요청 시 업비트 호출을 줄입니다.

설치 및 빠른 시작 (로컬 개발용)
1) Python 가상환경 만들기 및 패키지 설치
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) 개발용 스크립트로 백엔드와 UI 동시에 실행
```bash
chmod +x scripts/run_dev.sh
./scripts/run_dev.sh all
```
- 위 스크립트는 로컬에서 backend(uvicorn)과 UI(streamlit)를 동시에 실행합니다.
- `--no-reload` 또는 `--detached` 같은 플래그로 동작을 조절할 수 있습니다. `./scripts/run_dev.sh --help`로 확인하세요.

개별 컴포넌트 실행
- 백엔드만: uvicorn server.api:app --host 127.0.0.1 --port 8000
- UI만: streamlit run ui/ui_dashboard.py
- 봇: python -m server.bot

Docker로 실행하기 (권장: 간단한 통합 테스트)
- Docker Compose를 이용하면 Redis, 백엔드, UI, 봇을 함께 띄울 수 있습니다.

1) 컨테이너 빌드 및 실행 (일반적)
```bash
docker compose up --build
```

2) 개발형(코드 바인드-마운트가 설정된 경우)
```bash
docker compose up
```
- 개발 모드에서는 `docker-compose.override.yml`가 로컬 소스를 컨테이너에 마운트해 편집 즉시 반영되도록 합니다.
- 주의: 개발 모드에서 UI(8501)는 호스트에서 접근 가능하며, UI 컨테이너는 백엔드 컨테이너를 내부 네트워크로 호출합니다. 이때 `STREAMLIT_API_BASE` 환경변수로 백엔드 주소(http://backend:8000)를 자동으로 사용하도록 설정되어 있습니다.

중요 환경변수
- STREAMLIT_API_BASE: UI 컨테이너(또는 로컬 UI)가 호출할 API 기본 URL입니다. Docker에서는 `http://backend:8000`로 설정되어 컨테이너 네트워크를 통해 호출합니다. 로컬 개발 시 기본값은 `http://127.0.0.1:8000`입니다.
- REDIS_URL: Redis 캐시를 사용할 경우 이 URL로 연결합니다. 예: `redis://redis:6379/0`

설정 파일 설명 (runtime/config.json)
- strategy_name: 사용할 전략 이름(예: VolatilityBreakout, DualMomentum, RSI)
- market: 거래할 마켓 (예: KRW-BTC)
- timeframe: 캔들 단위(예: minute5, minute15)
- candle_count: 과거 캔들 수
- loop_interval_sec: 봇이 다음 루프를 시작하기까지 대기 시간(초)
- order_settings: 주문과 관련된 최소/기본 금액 설정
- use_kelly_criterion: 켈리공식 사용 여부
- kelly_criterion: 켈리용 파라미터(win_rate, payoff_ratio, fraction)
- strategy_params: 전략별 세부 파라미터(예: VolatilityBreakout.k_value)

간단한 예시 (runtime/config.json)
```json
{
  "strategy_name": "VolatilityBreakout",
  "market": "KRW-BTC",
  "timeframe": "minute5",
  "candle_count": 200,
  "loop_interval_sec": 5,
  "order_settings": { "min_order_amount": 5500, "trade_amount_krw": 6000 },
  "use_kelly_criterion": true,
  "kelly_criterion": { "win_rate": 0.65, "payoff_ratio": 1.2, "fraction": 0.5 }
}
```

문제 해결(자주 발생하는 상황과 대응)
- UI에서 "서버 호출 실패" 또는 Connection refused
  - 백엔드가 동작 중인지 확인: `curl -sS http://127.0.0.1:8000/health` (정상 시 {"status":"ok"} 반환)
  - 도커 사용시: `docker compose ps`와 `docker compose logs -f backend` 확인
- Upbit API에서 429(Too Many Requests)가 발생
  - 서버는 캐시와 재시도(backoff) 로직을 사용하도록 설계되어 있지만, 요청량이 많으면 업비트에서 차단됩니다.
  - 해결 방법: 프리페치 간격 늘리기(runtime/config.json에서 prefetch 관련 값 수정), Redis 사용으로 공유 캐시 적용, 또는 요청 빈도 자체를 낮추세요.
- SSL/네트워크 오류 또는 API 인증 오류
  - 인증키 문제가 있으면 `.env`에 키를 넣고 `server/config.py`를 통해 로드하세요. 테스트 단계에서는 인증이 필요없도록 설정되어 있습니다.

개발자 팁
- UI를 편집하면 Streamlit이 자동으로 리로딩합니다. backend(uvicorn)는 개발모드에서 server 디렉터리만 감시하도록 설정돼 있으므로 UI 수정으로 backend가 재시작되지 않도록 했습니다.
- `scripts/run_dev.sh`는 로컬 개발을 쉽게 하기 위해 백엔드 헬스 체크(wait_for_backend) 기능을 포함합니다. UI는 백엔드가 준비될 때까지 대기합니다.

도움이 필요하면
- 문제 상황(오류 메시지, 로그, 재현 방법)을 붙여서 알려 주세요. 제가 로그를 보고 더 구체적으로 도와 드리겠습니다.

---

이제 이 README를 보고도 모르겠다면, 원하는 항목(예: "로컬에서 UI만 띄우는 방법을 더 자세히 알려줘")을 한 가지만 말해 주세요 — 제가 단계별로 더 자세히 설명해 드립니다.
