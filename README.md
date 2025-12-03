# Upbit 자동 트레이딩 봇 — 전체 구조와 흐름을 쉽게 이해하기

이 프로젝트는 Upbit(업비트) 거래소에 연결해 자동으로 코인 전략을 실행하고, Streamlit UI와 FastAPI를 통해 설정·상황을 확인할 수 있는 개발·학습용 시스템입니다. 이 안내서는 사용 기술, 소스별 기능, 실행 흐름을 자주 쓰는 순서로 정리해 초등학생도 흐름을 따라갈 수 있게 설명합니다.

## 🔎 핵심 목적과 사용자에게 주는 가치
- **24시간 시장 관찰 자동화**: 변동성 돌파, 듀얼 모멘텀 등 전략을 자동으로 계산해 주문 후보를 만들고 켈리공식을 섞어 투자할 금액을 조절합니다.
- **UI·API·스크립트로 제어**: Streamlit UI, FastAPI 엔드포인트, `scripts/run_dev.sh` 등을 통해 설정, 스크리닝, 워처, 백그라운드 작업을 쉽게 조작하고 로그를 확인할 수 있습니다.
- **만들기 쉬운 실험 환경**: Docker와 로컬 모드 모두에서 실행 가능하므로 전략, 자금관리, 프리페치 성능을 바꿔가며 연구 가능합니다.

## 🧰 기술 스택 요약
| 층           | 핵심 기술                                                           | 역할 요약                                                                             |
|-------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 애플리케이션 레이어  | **Python 3.13**, `FastAPI`, `streamlit`, `plotly`, `requests`   | 서버 API, UI, 차트, HTTP 통신을 처리합니다. FastAPI는 설정/스크리닝/API 먹통 체크, Streamlit는 UI를 구현합니다. |
| 전략·자금관리 로직  | `server.strategy`, `server.money_manager`, `Kelly criterion` 구현 | 변동성 돌파·모멘텀·RSI 전략과 Kelly 공식을 하나의 파이프라인으로 결합하여 주문 결정을 내립니다.                        |
| 데이터 캐시·히스토리 | `Redis` (옵션), 내부 캐시, `runtime/history/positions_history.json`   | 클라이언트(특히 UI)의 429 발생을 막기 위해 prefetch 데이터를 모아두고, 파일 기반 히스토리로 시계열 차트를 그립니다.         |
| 실행 인프라      | `uvicorn`, `docker compose`, `bash scripts`                     | 개발/배포 모드에서 백엔드와 UI를 동시에 띄우고 Docker 네트워크에서 서로 통신합니다.                               |

핵심 개념을 아주 쉽게 설명
- 전략(strategy): 코인 가격 데이터(캔들)를 보고 ‘사야 한다/팔아야 한다’고 결정하는 규칙입니다.
- 켈리공식(Kelly criterion): 이길 확률과 기대 수익에 근거해 한 번에 투자할 금액을 계산하는 방법입니다.
- 스크리닝(screening): 여러 종목을 모아 놓고 변동성이 큰 종목을 골라내는 기능입니다.
- 워처(watcher): 실시간(폴링)으로 조건을 체크해 이벤트가 발생하면 로그/알림을 남기는 작은 백그라운드 작업입니다.

프로젝트 구조 (중요한 파일과 역할)
```
/upbit-trader
├── server/                # 서버(백엔드) 코드 (FastAPI + 감시봇 로직)
│   ├── api.py             # FastAPI REST API: 설정, 스크리닝, watcher, klines_batch 등
│   ├── bot.py             # 봇의 메인 루프: 시세 조회 → 전략 평가 → 주문(모의/실제)
│   ├── upbit_api.py       # Upbit와 통신하는 로직(pyupbit 또는 직접 HTTP 사용)
│   ├── strategy.py        # 여러 전략 구현 (VolatilityBreakout, DualMomentum, RSI 등)
│   ├── money_manager.py   # 켈리공식 등 자금관리 로직
│   ├── config.py          # runtime/config.json 읽기/쓰기, 전역 설정
│   └── logger.py          # 로그 설정
├── ui/                    # Streamlit 기반 사용자 인터페이스
│   └── ui_dashboard.py    # 설정 편집, 종목 스크리닝, 워처 제어 페이지
├── runtime/               # 실행 시 사용하는 설정 (PID, 히스토리 등)
│   └── config.json        # 기본 실행 설정(전략, 마켓, 파라미터 등)
│   └── history/            # 포지션 히스토리 JSON 파일 저장소
├── logs/                  # 실행 로그 파일 저장소
│   ├── backend.log        # 백엔드 로그
│   ├── ui.log             # UI 로그
│   └── trading_bot.log    # 트레이딩 봇 로그
├── scripts/
│   └── run_dev.sh         # 개발용 실행 스크립트 (백엔드 + UI 동시 실행 등)
├── docker-compose.yml      # Docker 서비스 정의
├── docker-compose.override.yml  # 개발용 바인드 마운트(코드 수정시 재시작 흐름 제어)
├── requirements.txt        # Python 의존성 목록
└── README.md
```

### 주요 폴더와 역할
- `server/`: FastAPI 엔드포인트(API, 설정 저장/재로딩, 스크리닝, watcher·prefetch 등), Upbit 연동, 전략/자금 관리, 로그 설정, 히스토리 저장기를 포함합니다.
- `ui/`: Streamlit 대시보드. 설정편집, 종목스크리닝, 이벤트 감시자 제어, 원화잔고/포지션 분석 등 여러 메뉴를 버튼 형식의 사이드바로 제공합니다.
- `runtime/`: `config.json`, 실행 중 PID 파일, `history/positions_history.json` 등을 담아 상태를 보존합니다. 히스토리 디렉터리는 새로 만든 JSON 파일을 통해 과거 포지션 가치를 저장합니다.
- `scripts/run_dev.sh`: 로컬에서 backend + UI를 순서대로 띄우고 로그를 모으며, `all`, `backend`, `ui` 모드를 지원합니다.
- `docker-compose*.yml` + `Dockerfile.*`: Redis, backend, UI, bot 컨테이너를 조합해 통합 테스트/운영 환경을 제공합니다.

## 📚 주요 소스별 요약
| 파일/모듈                                  | 기술                 | 기능 한 줄 요약                                                                                                                                                                                                   |
|----------------------------------------|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `server/api.py`                        | FastAPI            | `/balances`, `/positions`, `/screen/volatility_top`, `/klines_batch`, `/watcher/*`, `/config`, `/reload`, `/positions/history`를 노출하여 설정/데이터/워처 제어를 처리한다. 토큰 버킷·세마포어로 Upbit 호출을 제한하고 Redis 또는 인메모리 캐시를 사용한다. |
| `server/bot.py`                        | Python 스레드 기반 루프   | `runtime/config.json`을 읽고 주기적으로 전략을 평가하여 주문을 생성하고, 로깅 및 상태 업데이트를 남긴다.                                                                                                                                       |
| `server/strategy.py`                   | 전략 구현              | 변동성 돌파, 듀얼 모멘텀(절대/상대), RSI 조합 등을 정의하고, 단일 전략으로부터 신호를 만드는 도우미 함수가 있다.                                                                                                                                        |
| `server/money_manager.py`              | Kelly criterion    | 승률·수익비를 받아 주문 당 투자금과 레버리지 등을 계산한다.                                                                                                                                                                          |
| `server/history.py`                    | JSON 기반 히스토리 스토어   | `runtime/history/positions_history.json`에 `ts`, `total_equity`, 각 종목의 평가금액·손익을 기록하고 읽는 헬퍼. UI 시계열 차트를 위해 `GET /positions/history`를 제공한다.                                                                    |
| `server/config.py`                     | 설정 파싱              | `.env`, `runtime/config.json` 등을 읽어 글로벌 설정을 관리하고, 변경 시 저장/재시작을 안내한다.                                                                                                                                        |
| `server/logger.py`                     | logging            | 로컬/컨테이너 환경 모두에서 통일된 로그 포맷과 파일/스트림 핸들러를 만든다.                                                                                                                                                                 |
| `server/upbit_api.py`                  | pyupbit + https 호출 | 인증 키가 있으면 private API를 사용; 그 외에는 public klines만 호출하며 rate 제한을 최소화하기 위한 반복 로직이 있다.                                                                                                                           |
| `ui/ui_dashboard.py`                   | Streamlit          | 버튼형 사이드바, 설정 폼, 화면별 상태(스크리닝 차트, watcher 상태, 포지션 테이블/차트) 및 `/positions/history`를 이용한 시계열 라인 및 손익 바 차트를 렌더링한다.                                                                                                |
| `scripts/run_dev.sh`                   | Bash               | backend(Uvicorn) → `wait_for_backend()` → UI(Streamlit)를 순차 실행하며 `logs/`로 실시간 로그를 남긴다.                                                                                                                      |
| `Dockerfile.*` + `docker-compose*.yml` | Docker             | backend/api/ui/bot 컨테이너와 Redis를 정의하고, UI에서는 `STREAMLIT_API_BASE`로 `http://backend:8000`을 기본값으로 잡아 통합 서비스를 실행한다.                                                                                             |

## 🧩 데이터 흐름 (prefetch → 히스토리 → UI)
1. `server/api.py` prefetched klines/positions → Redis 또는 인메모리 캐시에 저장 (`_cache_set/_cache_get`).
2. `/positions` 호출 시 현재 포지션/평가금액/손익을 계산 → nodal history JSON(새로 만든 `server/history.py`)에 `ts` 단위로 누적 저장.
3. Streamlit 포지션 페이지는 `/positions`, `/positions/history`를 연속 호출 → 표, 캔들+MA+RSI, 평가금액/손익 차트를 각각 렌더링.
4. `runtime/history` 디렉터리가 존재하므로 컨테이너 재기동 후에도 최대 N개(기본 720개) 히스토리 유지해 시계열 차트를 부드럽게 만듭니다.

### 🌀 WebSocket→캐시→봇/API/UI 흐름
1. `server/ws_listener_private.py`(MyOrder 전용)와 `server/ws_listener_public.py`(공용 ticker 전용)로 WebSocket 핸들러가 분리되어 있고, `ws_timeframes`(`runtime/config.json`의 `timeframe`, `ws_timeframes` 필드 포함)를 기반으로 `ticker` 메시지로부터 1분봉을 집계합니다.
2. 실제 업비트 주문 `order` 이벤트는 `runtime/history/exec_history.json`에 기록되고, `ws:trades:{ticker}` 리스트와 `ws:candles:{timeframe}:{ticker}` 키로 Redis에도 저장됩니다.
3. `server/bot.py`는 가장 먼저 Redis `ws:candles:{desired_timeframe}:{ticker}` 캐시를 읽고(없으면 REST 호출), `ws_listener_private`이 만든 여러 타임프레임을 그대로 전략에 공급합니다.
4. FastAPI의 `/klines_batch`는 기존 `ticker|timeframe|count` 캐시를 사용하므로 WebSocket 캔들 캐시와는 별개이며, UI는 항상 `/klines_batch` → `FastAPI` → `_cache_get` 흐름을 통해 데이터를 가져옵니다.

```
[Upbit WS ticker/order]
           ↘
      WebSocket Listener
           ↘───────────┬──────────────┬─────────────
                       │              │
             redis ws:candles:*  runtime/history/exec_history.json
                       │              │
        ┌──────────────┴──────────────┴────────────┐
        │                    │                    │
    bot (Redis 우선)   FastAPI /klines_batch   Streamlit UI
 (ws:candles:tf:ticker)   (ticker|tf|count)     → /klines_batch
```

- 이 다이어그램은 WebSocket 수신이 Redis에 다양한 타임프레임 캐시를 만들고, bot/API/UI가 각자 필요한 저장소에서 먼저 읽도록 설계된 점을 강조합니다.
- `runtime/config.json`에 `ws_timeframes`를 넣으면 `ws_listener`가 지정한 순서대로 간격을 추가로 계산합니다(예: `["minute1","minute5","minute15"]`).
- Redis가 없는 환경에서는 `bot`/`ui`/`api`가 각각 fallback 로직으로 REST 호출을 사용하지만, WebSocket과 `exec_history`는 계속 로깅됩니다.

## 🛰️ 실시간 WebSocket 수신 및 exec 기록
1. `server/ws_listener.py`는 Upbit WebSocket(`ticker`+`trade`)을 구독해 Redis에 실시간 payload를 저장하고 `/ws/*` 엔드포인트로 상태를 제어할 수 있게 합니다.
2. 체결(trade) 이벤트는 `runtime/history/exec_history.json`에 append 되어 영구 보존되며, Redis에는 `ws:trades:<ticker>` 리스트(최대 200개)도 유지합니다.
3. FastAPI에서 `ws_start`, `ws_stop`, `ws_status`를 제공하므로 UI나 `curl`로 리스너를 직접 시작/중지하거나 상태를 확인할 수 있습니다.
4. 이 데이터를 UI나 봇이 실시간 UI/알림용으로 조회하려면 Redis `ws:ticker:...` 키 또는 `runtime/history/exec_history.json`을 읽도록 새 위젯/로직을 추가하세요.

## 🚀 실행하기 (로컬 & Docker)
### 1) 로컬 개발
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x scripts/run_dev.sh
./scripts/run_dev.sh all
```
- 위 스크립트는 로컬에서 backend(uvicorn)과 UI(streamlit)를 동시에 실행합니다.
- `--no-reload` 또는 `--detached` 같은 플래그로 동작을 조절할 수 있습니다. `./scripts/run_dev.sh --help`로 확인하세요.

개별 컴포넌트 실행
- 백엔드만: uvicorn server.api:app --host 127.0.0.1 --port 8000
- UI만: streamlit run ui/ui_dashboard.py
- 봇: python -m server.bot

### 2) Docker (권장 통합 테스트)
```bash
docker compose up --build
docker compose up
```
- `docker-compose.override.yml`는 개발용 바인드 마운트를 추가하고, UI는 `STREAMLIT_API_BASE=http://backend:8000`으로 backend를 호출합니다.
- 필요시 `docker compose logs -f backend` 등으로 로그를 확인하고 `docker compose down`으로 정리합니다.

### 3) 개별 컴포넌트 (안정적으로 짧게 실행)
```bash
uvicorn server.api:app --reload --host 127.0.0.1 --port 8000
streamlit run ui/ui_dashboard.py
python -m server.bot
```
- FastAPI는 `--reload`로 코드 변경시 재시작(개발 전용).
- Streamlit UI는 `--server.port` 등을 조정해 여러 창 테스트 가능합니다.

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
  "kelly_criterion": { "win_rate": 0.65, "payoff_ratio": 1.2, "fraction": 0.5 },
  "strategy_params": {
    "VolatilityBreakout": { "k_value": 0.5 },
    "DualMomentum": { "window": 12 },
    "RSI": { "period": 14, "oversold": 30, "overbought": 70 }
  },
  "prefetch_count": 200,
  "prefetch_interval_sec": 60,
  "prefetch_batch_size": 5,
  "prefetch_rate_per_sec": 5,
  "prefetch_rate_capacity": 5,
  "prefetch_max_concurrent": 3,
  "KLINES_CACHE_TTL": 600
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
- `prefetch_*` 필드를 조정하면 Upbit 호출 빈도 제한, Redis 여부, cache ttl 등을 튜닝할 수 있습니다.

## 🛠 운영 팁
- **로그 장소**: `logs/backend.log`, `logs/ui.log`, `logs/trading_bot.log`에 각 시스템 로그가 기록됩니다. `scripts/run_dev.sh`는 `wait_for_backend`로 backend 준비를 기다립니다.
- **API 오류 대응**: `curl http://127.0.0.1:8000/health` → `{"status":"ok"}` 확인. 문제가 있으면 `docker compose logs -f backend` 등으로 로그를 확인하고, 컨테이너 재시작 후에도 문제가 지속되면 설정이나 코드 문제일 수 있습니다.
