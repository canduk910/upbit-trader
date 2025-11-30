import streamlit as st
# reload-test: touch ui file to verify backend reload behavior (do not remove)
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
# Ensure process timezone is KST (Asia/Seoul) so Streamlit displays local times correctly
os.environ.setdefault('TZ', 'Asia/Seoul')
try:
    time.tzset()
except Exception:
    # time.tzset may not be available on all platforms (Windows), ignore if unavailable
    pass

# Import configuration helpers from server package
try:
    from server.config import load_config, save_config as save_local
except Exception:
    # In case UI runs in an environment where server package isn't available on PYTHONPATH,
    # fall back to a stub that will raise an informative error when used.
    def load_config():
        raise RuntimeError('server.config.load_config not available in PYTHONPATH')

    def save_local(cfg):
        raise RuntimeError('server.config.save_config not available in PYTHONPATH')


def validate_config(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Basic validation for runtime config used by the UI form.
    Returns (True, '') if valid, otherwise (False, 'reason').
    """
    if not isinstance(cfg, dict):
        return False, '설정 데이터가 딕셔너리가 아닙니다.'
    if not cfg.get('market'):
        return False, 'Market 값이 필요합니다.'
    if not isinstance(cfg.get('candle_count', 0), int) or cfg.get('candle_count', 0) <= 0:
        return False, 'Candle count 는 1 이상의 정수여야 합니다.'
    order = cfg.get('order_settings', {})
    if not isinstance(order.get('trade_amount_krw', 0), (int, float)) or order.get('trade_amount_krw', 0) <= 0:
        return False, 'Trade amount_krw 는 0보다 큰 숫자여야 합니다.'
    kelly = cfg.get('kelly_criterion', {})
    if cfg.get('use_kelly_criterion'):
        wr = float(kelly.get('win_rate', 0))
        pr = float(kelly.get('payoff_ratio', 0))
        frac = float(kelly.get('fraction', 0))
        if not (0 <= wr <= 1):
            return False, 'Kelly win_rate 는 0~1 범위여야 합니다.'
        if pr <= 0:
            return False, 'Kelly payoff_ratio 는 양수여야 합니다.'
        if not (0 <= frac <= 1):
            return False, 'Kelly fraction 은 0~1 범위여야 합니다.'
    return True, ''


RUNTIME_CONFIG = Path(__file__).resolve().parents[1] / 'runtime' / 'config.json'

st.set_page_config(page_title='Upbit Trader', layout='wide')

# Helper to render DataFrame safely across Streamlit versions and hide index when requested
def _safe_dataframe(df: 'pd.DataFrame', hide_index: bool = False, **kwargs):
    """Render a pandas DataFrame while attempting to hide the index and use full width.
    Tries the newer `width='stretch'` API first, then falls back to `use_container_width=True`,
    and finally to the basic st.dataframe if needed.
    """
    try:
        return st.dataframe(df, hide_index=hide_index, width='stretch', **kwargs)
    except TypeError:
        try:
            return st.dataframe(df, hide_index=hide_index, use_container_width=True, **kwargs)
        except TypeError:
            # Older Streamlit may not support hide_index; fallback to st.table (no hide)
            try:
                return st.table(df)
            except Exception:
                return st.write(df)


def _safe_plotly_chart(fig, **kwargs):
    """Render a plotly figure using newer width='stretch' API if available, otherwise fall back to use_container_width."""
    try:
        return st.plotly_chart(fig, width='stretch', **kwargs)
    except TypeError:
        try:
            return st.plotly_chart(fig, use_container_width=True, **kwargs)
        except Exception:
            return st.write('차트를 표시할 수 없습니다.')


# Callback for strategy selectbox to trigger immediate rerun so fields update
def _on_strategy_change():
    # mark change and request a rerun so Streamlit re-renders dynamic fields immediately
    st.session_state['_strategy_changed'] = True
    try:
        # prefer experimental_rerun compatibility
        rerun = getattr(st, 'rerun', None) or getattr(st, 'experimental_rerun', None)
        if rerun:
            rerun()
    except Exception:
        pass


# API base URL for backend calls: read from env var STREAMLIT_API_BASE (set by docker-compose)
# If not set, fall back to localhost for local development.
API_BASE = os.getenv('STREAMLIT_API_BASE', 'http://127.0.0.1:8000')

# Create a requests session with retry/backoff to make UI-server comms more resilient
def _build_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

_API_SESSION = _build_session()

def api_request(method: str, path: str, params=None, json=None, timeout=10):
    """Call server API with retries and return requests.Response or raise Exception with friendly message."""
    if not API_BASE:
        raise RuntimeError("API Base URL is not set in the sidebar")
    url = API_BASE.rstrip("/") + "/" + path.lstrip("/")
    try:
        if method.lower() == "get":
            resp = _API_SESSION.get(url, params=params, timeout=timeout)
        else:
            resp = _API_SESSION.post(url, json=json, timeout=timeout)
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        # Wrap low-level error to present user-friendly message in UI
        raise RuntimeError(f"서버 호출 실패: {e}") from e


# --- Upbit public klines helper (cached) ---
@st.cache_data(ttl=10)
def fetch_klines_cached(market: str, timeframe: str = 'minute1', count: int = 200) -> pd.DataFrame | None:
    """
    UI-side fetch function disabled.
    The UI must not call Upbit public API directly — always go through the backend `/klines_batch` endpoint.
    This function returns None to force the UI to rely on backend data and avoid causing 429 Too Many Requests.
    """
    # Return None immediately to prevent direct Upbit calls from the UI.
    st.warning('UI는 직접 Upbit 호출을 하지 않습니다. 백엔드의 prefetch가 완료될 때까지 대기하세요.')
    return None


def fetch_klines_batch_from_backend(tickers: list[str], timeframe: str = 'minute15', count: int = 100) -> Dict[str, pd.DataFrame | None]:
    """Call server /klines_batch to get klines for multiple tickers. Returns mapping ticker->DataFrame or None.
    Falls back to empty dict on error.
    """
    url = f"{API_BASE.rstrip('/')}/klines_batch"
    try:
        resp = requests.post(url, json={'tickers': tickers, 'timeframe': timeframe, 'count': count}, timeout=15)
        resp.raise_for_status()
        data = resp.json().get('klines', {})
        result = {}
        for t, v in data.items():
            if not v:
                result[t] = None
                continue
            df = pd.DataFrame(v)
            df = df.rename(columns={
                'candle_date_time_kst': 'time',
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume'
            })
            cols = [c for c in ['time','open','high','low','close','volume'] if c in df.columns]
            df = df[cols]
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
            for col in ['open','high','low','close','volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.sort_values('time', ascending=True).reset_index(drop=True)
            result[t] = df
        return result
    except Exception as e:
        st.warning(f'백엔드 batch klines 호출 실패: {e}')
        return {}


def compute_volatility_from_df(df: pd.DataFrame | None) -> float | None:
    """Compute volatility as (max_high - min_low) / mean_close * 100.
    Returns percent float or None if cannot compute.
    """
    if df is None:
        return None
    try:
        if not hasattr(df, 'empty') or df.empty:
            return None
        if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            return None
        max_high = float(df['high'].max())
        min_low = float(df['low'].min())
        mean_close = float(df['close'].mean()) if float(df['close'].mean() or 0) != 0 else None
        if mean_close is None or mean_close == 0:
            return None
        vol = (max_high - min_low) / mean_close * 100.0
        return float(vol)
    except Exception:
        return None


def _normalize_klines_df(df: pd.DataFrame | None, min_length: int = 30) -> pd.DataFrame | None:
    """Validate and normalize a kline DataFrame for plotting.
    - Ensures time column exists and is datetime
    - Renames common Upbit keys if needed
    - Sorts ascending by time, drops duplicate timestamps
    - Converts numeric columns and fills small gaps
    - Returns None if data is invalid
    """
    if df is None:
        return None
    # Accept list-of-dicts as well
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return None

    # Map common Upbit field names to expected ones
    rename_map = {
        'candle_date_time_kst': 'time',
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # required columns
    required = ['time', 'open', 'high', 'low', 'close', 'volume']
    if 'time' not in df.columns:
        return None

    # time -> datetime
    try:
        df['time'] = pd.to_datetime(df['time'])
    except Exception:
        try:
            df['time'] = pd.to_datetime(df['time'].astype(str))
        except Exception:
            return None

    # sort ascending and dedupe
    df = df.sort_values('time', ascending=True).reset_index(drop=True)
    if df['time'].duplicated().any():
        df = df[~df['time'].duplicated(keep='last')].reset_index(drop=True)

    # ensure numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.NA

    # fill small gaps for price columns
    try:
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill().bfill()
        df['volume'] = df['volume'].fillna(0)
    except Exception:
        pass

    if len(df) < min_length:
        st.warning(f'차트에 필요한 데이터가 부족합니다: {len(df)}개 (최소 {min_length} 필요)')
        return df
    return df


# --- Indicator helpers and plotting ---
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Standard Wilder's RSI calculation without forcing initial values to 0.
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=period).mean()
    ma_down = down.rolling(period, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def plot_candles_with_indicators(df: pd.DataFrame, ticker: str, ma_windows: list[int], rsi_period: int):
    # Create subplot: row1 = candlestick+MA, row2 = volume bars, row3 = RSI
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)

    # Row 1: Candlestick — set colors: rising(red), falling(blue)
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='OHLC', increasing=dict(line=dict(color='red'), fillcolor='red'), decreasing=dict(line=dict(color='blue'), fillcolor='blue')), row=1, col=1)

    # moving averages on top
    for w in ma_windows:
        ma = df['close'].rolling(w).mean()
        fig.add_trace(go.Scatter(x=df['time'], y=ma, mode='lines', name=f'MA{w}', line=dict(width=1.2)), row=1, col=1)

    # Row 2: Volume bars
    # color volume bars green when close >= open else red
    try:
        # rising -> red, falling -> blue
        colors = ['red' if c >= o else 'blue' for o, c in zip(df['open'], df['close'])]
    except Exception:
        colors = 'gray'
    fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name='Volume', marker=dict(color=colors), hovertemplate='Volume: %{y:,.0f}<extra></extra>'), row=2, col=1)
    # Keep volume axis independent
    fig.update_yaxes(title_text='Volume', row=2, col=1)

    # Row 3: RSI
    rsi = compute_rsi(df['close'], period=rsi_period)
    fig.add_trace(go.Scatter(x=df['time'], y=rsi, mode='lines', name=f'RSI({rsi_period})', line=dict(color='purple', width=2.0), connectgaps=False, hovertemplate='RSI: %{y:.2f}<extra></extra>'), row=3, col=1)
    # RSI bands (draw as shapes for visibility)
    fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=3, col=1)
    fig.update_yaxes(range=[0,100], row=3, col=1)

    # layout tweaks
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=20), showlegend=True, hovermode='x unified')
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])], rangeslider_visible=False)
    return fig


# Load existing config
cfg = load_config()

# --- Page renderers ---

def render_config_page(cfg: Dict[str, Any]):
    st.title('설정 편집')

    # Strategy selector placed outside the form so changes immediately re-render detail fields.
    _strategy_opts = ['VolatilityBreakout', 'DualMomentum', 'RSI']
    _default = cfg.get('strategy_name', 'VolatilityBreakout')
    try:
        _default_idx = _strategy_opts.index(_default) if _default in _strategy_opts else 0
    except Exception:
        _default_idx = 0
    # initialize session state key if missing so we can read it inside the form
    if 'cfg_strategy' not in st.session_state:
        st.session_state['cfg_strategy'] = _strategy_opts[_default_idx]
    strategy_name = st.selectbox('전략 (어떤 방식을 쓸까요?)', options=_strategy_opts, index=_default_idx, key='cfg_strategy', help='어떤 거래 전략을 쓸지 골라요. 예: 변동성 돌파, 모멘텀, RSI')

    # Now the form (other widgets and save buttons). Inside the form read the currently selected strategy
    with st.form('config_form', clear_on_submit=False):
        # use the session_state value so fields update immediately when the selectbox changes
        strategy_name = st.session_state.get('cfg_strategy', _strategy_opts[_default_idx])

        # --- 전략별 상세 설정: 전략 선택 바로 아래에 위치하도록 함 ---
        st.caption('선택한 전략에 따라 여기서 세부 옵션을 정해요.')
        strategy_params = cfg.get('strategy_params', {})
        if strategy_name == 'RSI':
            rsi = strategy_params.get('RSI', {})
            rsi_period = st.number_input('RSI 기간 (몇개 캔들로 계산할지)', min_value=1, value=int(rsi.get('period', 14)), help='RSI를 계산할 때 몇 개의 캔들을 볼지 정해요. 보통 14')
            rsi_oversold = st.number_input('과매도 기준 (0~100)', min_value=0, max_value=100, value=int(rsi.get('oversold', 30)), help='이 값 아래면 너무 싸게 팔려서 사도 되는 상태예요.')
            rsi_overbought = st.number_input('과매수 기준 (0~100)', min_value=0, max_value=100, value=int(rsi.get('overbought', 70)), help='이 값 위면 너무 비싸게 사져서 팔아야 할 수도 있어요.')
        elif strategy_name == 'VolatilityBreakout':
            vb = strategy_params.get('VolatilityBreakout', {})
            k_value = st.number_input('변동성 비율 k (0~1)', min_value=0.0, max_value=1.0, value=float(vb.get('k_value', 0.5)), help='지난 기간의 변동성에서 몇 퍼센트만큼 돌파를 보면 진입할지 정하는 수치예요. 0.5가 보통 사용됩니다.')
        elif strategy_name == 'DualMomentum':
            dm = strategy_params.get('DualMomentum', {})
            window = st.number_input('모멘텀 계산 창 길이', min_value=1, value=int(dm.get('window', 12)), help='모멘텀을 계산할 때 몇 기간을 볼지 정해요. 숫자가 크면 더 긴 흐름을 봐요.')

        # 기본 정보(전략 상세 바로 아래에 위치)
        market = st.text_input('거래할 마켓 (예: KRW-BTC)', value=cfg.get('market', 'KRW-BTC'), help='어떤 코인을 거래할지 적어요. 예: KRW-BTC는 비트코인입니다.')
        timeframe = st.text_input('캔들 시간 단위 (예: minute5)', value=cfg.get('timeframe', 'minute5'), help='한 개의 캔들이 몇 분/시간인지 적어요. 예: minute5 = 5분')
        candle_count = st.number_input('캔들 개수(그래프 표시 길이)', min_value=1, value=int(cfg.get('candle_count', 200)), help='차트에 보여줄 과거 데이터의 개수예요. 숫자가 크면 긴 기간을 보여줘요.')
        loop_interval_sec = st.number_input('루프 실행 간격(초)', min_value=1, value=int(cfg.get('loop_interval_sec', 5)), help='자동으로 확인할 때 몇 초마다 할지 정해요.')

        st.subheader('주문 관련 설정 (돈/수량)')
        order_settings = cfg.get('order_settings', {})
        min_order_amount = st.number_input('최소 주문 금액 (원)', min_value=1000, value=int(order_settings.get('min_order_amount', 5500)), help='거래소에서 허용하는 최소 주문 금액이에요. 이 값보다 작으면 주문 못해요.')
        trade_amount_krw = st.number_input('한 번 거래할 금액 (원)', min_value=1000, value=int(order_settings.get('trade_amount_krw', 6000)), help='한 번 매수할 때 쓰는 돈이에요. 예: 6000원')

        st.subheader('켈리공식 (돈을 얼마나 쓸지 계산하는 방법)')
        use_kelly = st.checkbox('켈리공식 사용하기', value=bool(cfg.get('use_kelly_criterion', True)), help='켈리공식을 사용하면 이길 확률과 수익비율로 한 번에 투자할 돈을 계산해줘요.')
        kelly = cfg.get('kelly_criterion', {})
        win_rate = st.number_input('승률 (0~1)', min_value=0.0, max_value=1.0, value=float(kelly.get('win_rate', 0.65)), help='거래했을 때 이길 확률을 0부터 1 사이로 적어요. 예: 0.65 = 65%')
        payoff_ratio = st.number_input('평균 이익/손실 비율', min_value=0.0, value=float(kelly.get('payoff_ratio', 1.2)), help='이길 때 평균 이익이 손실보다 몇 배인지 적어요. 예: 1.2면 이익이 손실의 1.2배')
        fraction = st.number_input('적용 비율 (0~1)', min_value=0.0, max_value=1.0, value=float(kelly.get('fraction', 0.5)), help='켈리로 계산한 금액 중 얼마만 실제로 쓸지 0~1로 적어요. 예: 0.5는 반만 사용')

        submit = st.form_submit_button('미리보기 업데이트')

        # --- Prefetch & Cache settings ---
        st.subheader('미리받기(캐시) 설정 — 서버가 데이터를 미리 모아놓는 방법')
        prefetch_cfg = cfg.get('prefetch', {}) if isinstance(cfg.get('prefetch', {}), dict) else {}
        # Top-level prefetch keys (stored at root level, not inside 'prefetch' for backward compatibility)
        prefetch_count = st.number_input('미리 받을 캔들 수 (종목당)', min_value=1, value=int(cfg.get('prefetch_count', cfg.get('candle_count', 200))), help='서버가 각 종목에서 미리 받아둘 캔들 수예요. 그래프 길이에 영향을 줍니다.')
        prefetch_interval_sec = st.number_input('미리수집 반복 간격(초)', min_value=1, value=int(cfg.get('prefetch_interval_sec', 30)), help='서버가 미리 데이터를 모으는 간격이에요. 숫자가 작으면 더 자주 업데이트합니다.')
        prefetch_batch_size = st.number_input('한번에 모을 종목 수', min_value=1, value=int(cfg.get('prefetch_batch_size', 5)), help='한 번에 몇 종목씩 모을지 정해요. 너무 크면 호출이 몰려서 실패할 수 있어요.')
        prefetch_parallelism = st.number_input('동시 작업 수(스레드)', min_value=1, value=int(cfg.get('prefetch_parallelism', 3)), help='몇 개의 작업을 동시에 실행할지 정해요. 숫자가 크면 빨라지지만 컴퓨터에 부담이 들어요.')
        prefetch_sleep_sec = st.number_input('종목 사이 쉬는 시간(초)', min_value=0.0, value=float(cfg.get('prefetch_sleep_sec', 0.2)), help='한 종목을 처리한 뒤 잠깐 쉬는 시간이에요. 서버 과부하를 줄여요.')
        prefetch_min_interval_sec = st.number_input('Redis 없을 때 최소 간격(초)', min_value=1, value=int(cfg.get('prefetch_min_interval_sec', 60)), help='Redis가 없으면 미리받기 사이 간격을 더 길게 해요.')
        prefetch_no_redis_max_count = st.number_input('Redis 없을 때 최대 캔들 수', min_value=1, value=int(cfg.get('prefetch_no_redis_max_count', 120)), help='Redis가 없을 때는 너무 많이 가져오지 않도록 제한해요.')
        prefetch_rate_per_sec = st.number_input('초당 허용 호출(토큰)', min_value=0.0, value=float(cfg.get('prefetch_rate_per_sec', 5)), help='초당 몇 번의 호출을 허용할지 토큰으로 정해요.')
        prefetch_rate_capacity = st.number_input('버스트 허용 토큰(추가 여유)', min_value=1, value=int(cfg.get('prefetch_rate_capacity', int(prefetch_rate_per_sec or 1))), help='잠깐 동안 더 많은 호출을 허용할 수 있는 여유량이에요.')
        prefetch_max_concurrent = st.number_input('최대 동시 작업 수', min_value=1, value=int(cfg.get('prefetch_max_concurrent', 3)), help='한 번에 병렬로 실행할 작업 최대 수예요.')
        prefetch_token_wait_timeout = st.number_input('토큰 대기 최대 시간(초)', min_value=0.0, value=float(cfg.get('prefetch_token_wait_timeout', 10.0)), help='토큰을 기다리는 최대 시간이에요. 너무 작으면 실패할 수 있어요.')
        # Cache TTL
        klines_cache_ttl = st.number_input('차트 데이터 캐시 유지시간(초)', min_value=1, value=int(cfg.get('KLINES_CACHE_TTL', os.getenv('KLINES_CACHE_TTL', '600'))), help='서버가 저장해두는 데이터가 얼마나 오래 유지될지 초 단위로 적어요.')

        # AI ensemble settings
        st.subheader('AI 조합 방법 (여러 AI를 어떻게 합칠까?)')
        ai_ensemble = cfg.get('ai_ensemble', {}) if isinstance(cfg.get('ai_ensemble', {}), dict) else {}
        ai_strategy = st.selectbox('AI 합치는 방식', options=['UNANIMOUS', 'MAJORITY', 'AVERAGE'], index=0 if ai_ensemble.get('strategy') is None else ['UNANIMOUS','MAJORITY','AVERAGE'].index(ai_ensemble.get('strategy', 'UNANIMOUS')), help='여러 AI가 모두 같은 의견일 때만 따를지(UNANIMOUS), 과반수 의견을 따를지(MAJORITY), 평균을 낼지(AVERAGE) 골라요.')
        openai_model = st.text_input('OpenAI 모델 이름', value=ai_ensemble.get('openai_model', cfg.get('OPENAI_MODEL', 'gpt-5.1-nano')), help='OpenAI에서 사용할 모델 이름을 적어요. 특별히 모르면 기본값 그대로 두세요.')
        gemini_model = st.text_input('Gemini 모델 이름', value=ai_ensemble.get('gemini_model', cfg.get('GEMINI_MODEL', 'gemini-2.5-flash')), help='Gemini에서 사용할 모델 이름을 적어요. 모르면 기본값 사용')

        # Universe (comma separated tickers)
        st.subheader('관심 종목 목록 (우리가 볼 종목들)')
        universe_list = cfg.get('universe')
        if isinstance(universe_list, list):
            universe_str = ','.join(universe_list)
        else:
            universe_str = str(universe_list or '')
        universe_input = st.text_area('관심 종목 (콤마로 구분, 예: KRW-BTC,KRW-ETH)', value=universe_str, height=80, help='여기에 보고 싶은 종목들을 쉼표로 구분해서 적어요. 예: KRW-BTC,KRW-ETH')

    new_cfg = {
        'strategy_name': strategy_name,
        'market': market,
        'timeframe': timeframe,
        'candle_count': candle_count,
        'loop_interval_sec': loop_interval_sec,
        'order_settings': {
            'min_order_amount': min_order_amount,
            'trade_amount_krw': trade_amount_krw,
        },
        'use_kelly_criterion': use_kelly,
        'kelly_criterion': {
            'win_rate': win_rate,
            'payoff_ratio': payoff_ratio,
            'fraction': fraction,
        },
        'strategy_params': {}
    }

    if strategy_name == 'RSI':
        new_cfg['strategy_params']['RSI'] = {
            'period': rsi_period,
            'oversold': rsi_oversold,
            'overbought': rsi_overbought,
        }
    elif strategy_name == 'VolatilityBreakout':
        new_cfg['strategy_params']['VolatilityBreakout'] = {'k_value': k_value}
    elif strategy_name == 'DualMomentum':
        new_cfg['strategy_params']['DualMomentum'] = {'window': window}

    # attach AI ensemble and universe
    new_cfg['ai_ensemble'] = {
        'strategy': ai_strategy,
        'openai_model': openai_model,
        'gemini_model': gemini_model,
    }
    # parse universe string into list
    universe_parsed = [s.strip() for s in universe_input.split(',') if s.strip()]
    if universe_parsed:
        new_cfg['universe'] = universe_parsed
    else:
        # keep existing if empty
        if isinstance(cfg.get('universe'), list):
            new_cfg['universe'] = cfg.get('universe')

    st.divider()
    st.header('설정 액션')
    valid, message = validate_config(new_cfg)
    if not valid:
        st.error('유효성 검사 실패: ' + message)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button('로컬에 저장'):
            if not valid:
                st.error('저장 실패: 유효성 검사 실패 - ' + message)
            else:
                try:
                    save_local(new_cfg)
                    st.success('저장 완료: runtime/config.json')
                except Exception as e:
                    st.error('로컬 저장 실패: ' + str(e))
    with col_b:
        if st.button('서버에 전송'):
            if not valid:
                st.error('전송 실패: 유효성 검사 실패 - ' + message)
            else:
                try:
                    resp = api_request('post', '/config', json={'config': new_cfg}, timeout=10)
                    st.success('서버에 전송 완료')
                except Exception as e:
                    st.error(str(e))
    with col_c:
        if st.button('서버 재로딩 요청'):
            try:
                resp = api_request('post', '/reload', timeout=5)
                st.success('서버 재로딩 요청 성공')
            except Exception as e:
                st.error(str(e))


def render_screening_page():
    st.title('종목 스크리닝')
    st.caption('변동성 TOP 종목을 조회합니다. (최근 캔들 기준)')
    st.code('변동성은 (max_high - min_low) / mean_close 로 계산됩니다.')

    # Controls moved into the page (previously in sidebar)
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1,1,1])
    market_prefix = ctrl_col1.text_input('마켓 접두사 (예: KRW)', value='KRW', key='scr_market_prefix')
    top_n = ctrl_col2.number_input('Top N', min_value=1, max_value=50, value=10, key='scr_top_n')
    # timeframe placed in the third column for grouping
    timeframe = ctrl_col3.selectbox('Timeframe', options=['minute5','minute15','minute60','day'], index=1, key='scr_timeframe')

    # Place the search button in a dedicated full-width row so it spans the page
    full_row = st.columns([1])[0]
    try:
        # Use width='stretch' when supported
        search_clicked = full_row.button('조회', key='scr_search', width='stretch')
    except TypeError:
        # Fallback for older Streamlit versions
        search_clicked = full_row.button('조회', key='scr_search')

    # When search clicked, fetch data and then render charts below
    if 'search_clicked' not in st.session_state:
        st.session_state['search_clicked'] = False
    if search_clicked:
        st.session_state['search_clicked'] = True
        # remember last query params to re-render on config changes
        st.session_state['screening_query'] = {
            'market_prefix': market_prefix,
            'top_n': top_n,
            'timeframe': timeframe,
        }

    if not st.session_state.get('search_clicked'):
        st.info('조회 버튼을 눌러 변동성 TOP 종목을 검색하세요.')
        return

    # read query
    query = st.session_state.get('screening_query', {})
    market_prefix = query.get('market_prefix', 'KRW')
    requested_n = int(query.get('top_n', 10))
    timeframe = query.get('timeframe', 'minute15')

    with st.spinner('변동성 TOP 종목 조회 중...'):
        # Try multiple backend endpoints for compatibility
        candidate_paths = [
            ('post', '/screening/top_volatility'),
            ('get', '/screen/volatility_top'),
            ('get', '/screen/volatility_top'),
        ]
        tickers = []
        last_exception = None
        for method, path in candidate_paths:
            try:
                if method == 'get':
                    resp = api_request('get', path, params={'market_prefix': market_prefix, 'top_n': requested_n, 'timeframe': timeframe}, timeout=8)
                else:
                    resp = api_request('post', path, json={'market_prefix': market_prefix, 'top_n': requested_n, 'timeframe': timeframe}, timeout=10)
                j = resp.json()
                # support different key names
                tickers = j.get('tickers') or j.get('top') or j.get('top_tickers') or j.get('top', [])
                # if server returns objects, try to extract ticker strings
                if tickers and isinstance(tickers, list) and isinstance(tickers[0], dict):
                    # expect dict with 'ticker' or 'symbol'
                    tickers = [it.get('ticker') or it.get('symbol') or it.get('market') for it in tickers]
                # filter empties
                tickers = [t for t in (tickers or []) if t]
                if tickers:
                    break
            except Exception as e:
                last_exception = e
                continue

        if not tickers:
            if last_exception:
                st.error(f'스케리닝 조회 실패: {last_exception}')
            else:
                st.warning('조건에 맞는 종목이 없습니다.')
            return

        # fetch klines in batch
        backend_data = fetch_klines_batch_from_backend(tickers, timeframe=timeframe, count=200)

        # Compute volatility (%) per ticker from backend_data and show a ranked table
        vol_rows = []
        for t in tickers:
            dfk = backend_data.get(t)
            dfk_norm = _normalize_klines_df(dfk, min_length=5)
            vol = compute_volatility_from_df(dfk_norm)
            vol_rows.append({'ticker': t, 'volatility_pct': None if vol is None else float(vol)})
        try:
            df_top = pd.DataFrame(vol_rows)
            # sort by volatility (ascending), NaNs at the end
            df_top = df_top.sort_values('volatility_pct', ascending=False, na_position='last').reset_index(drop=True)
            df_top.insert(0, 'rank', range(1, len(df_top) + 1))
            # pretty formatting column
            def _fmt(v):
                try:
                    if v is None or (isinstance(v, float) and (pd.isna(v))):
                        return '-'
                    return f"{float(v):.2f}%"
                except Exception:
                    return '-'
            df_top['변동성(%)'] = df_top['volatility_pct'].apply(_fmt)
            df_display = df_top[['rank','ticker','변동성(%)']]
            _safe_dataframe(df_display, hide_index=True)
        except Exception as e:
            st.warning(f'상단 변동성 표 생성 중 오류: {e}')

        # Diagnostics: show which tickers have data and which not
        missing = []
        insufficient = []
        for t in tickers:
            df = backend_data.get(t)
            if df is None:
                missing.append(t)
            else:
                try:
                    if hasattr(df, 'shape') and df.shape[0] < 10:
                        insufficient.append((t, int(df.shape[0])))
                except Exception:
                    pass
        if missing:
            st.warning(f"데이터가 없는 종목: {', '.join(missing)} (백엔드가 캐시를 아직 채우지 않았거나 수집 실패)")
        if insufficient:
            st.info('샘플이 적은 종목: ' + ', '.join([f"{t}({n})" for t,n in insufficient]))

        # Render a grid: 5 rows x 2 cols (max 10)
        max_slots = requested_n
        display_items = tickers[:max_slots]
        display_n = len(display_items)
        rows = (max_slots + 1) // 2
        idx = 0
        for r in range(rows):
            cols = st.columns(2, gap='small')
            for c in range(2):
                if idx >= display_n:
                    with cols[c]:
                        st.empty()
                    idx += 1
                    continue
                ticker = display_items[idx]
                with cols[c]:
                    st.subheader(f"{idx+1}. {ticker}")
                    df = backend_data.get(ticker)
                    df = _normalize_klines_df(df, min_length=30)
                    if df is None or (hasattr(df, 'empty') and df.empty):
                        st.info('차트 데이터를 불러올 수 없습니다. (백엔드 캐시 또는 수집 상태를 확인하세요)')
                    else:
                        # smaller chart per grid cell
                        fig = plot_candles_with_indicators(df, ticker, ma_windows=[20, 60], rsi_period=14)
                        fig.update_layout(height=320, margin=dict(l=8, r=8, t=24, b=12))
                        _safe_plotly_chart(fig)
                idx += 1
        st.caption(f"표시된 종목: {display_n} / 요청한 TopN: {requested_n}")


def render_positions_page():
    st.title('원화잔고 및 포지션 분석')
    st.write('원화 잔고와 보유 포지션을 요약하고 간단한 시각화를 제공합니다.')

    balances = None
    positions = None
    api_errors: list[str] = []

    # Try dedicated endpoints first
    reported_krw_from_server = 0.0
    balances_response_meta = {}
    try:
        resp = api_request('get', '/balances', timeout=6)
        if resp and resp.status_code == 200:
            j = resp.json()
            # API returns {'balances': [...], 'reported_krw_balance':..., 'cached':..., 'cached_ts':...}
            if isinstance(j, dict):
                balances = j.get('balances') if 'balances' in j else j
                reported_krw_from_server = float(j.get('reported_krw_balance') or 0.0)
                balances_response_meta = {k: j.get(k) for k in ('cached','cached_ts') if k in j}
            else:
                balances = j
    except Exception as e:
        api_errors.append(f"/balances 호출 실패: {e}")

    try:
        resp2 = api_request('get', '/positions', timeout=6)
        if resp2 and resp2.status_code == 200:
            j2 = resp2.json()
            if isinstance(j2, dict) and 'positions' in j2:
                positions = j2.get('positions')
            else:
                positions = j2
    except Exception as e:
        api_errors.append(f"/positions 호출 실패: {e}")

    # If both endpoints failed, try returning /status for diagnostics
    if balances is None and positions is None:
        try:
            resp3 = api_request('get', '/debug/status', timeout=4)
            if resp3 and resp3.status_code == 200:
                st.info('백엔드 상태:')
                st.json(resp3.json())
        except Exception:
            pass

    # Show API warnings (but avoid noisy raw exceptions)
    for err in api_errors:
        if 'keys not configured' in err or '503' in err:
            st.warning('서버에 API 키가 설정되지 않아 잔고를 조회할 수 없습니다. (관리자 설정 필요)')
        else:
            st.info(err)

    # Normalize balances into DataFrame
    try:
        if balances and isinstance(balances, list):
            bal_df = pd.DataFrame(balances)
        elif balances and isinstance(balances, dict):
            # sometimes API returns map
            bal_df = pd.DataFrame(balances.get('balances') if 'balances' in balances else [balances])
        else:
            bal_df = pd.DataFrame(columns=['currency', 'balance', 'locked', 'avg_buy_price'])

        # Accept multiple possible field names and normalize
        if 'currency' not in bal_df.columns and 'unit' in bal_df.columns:
            bal_df = bal_df.rename(columns={'unit': 'currency'})
        if 'currency' not in bal_df.columns and 'coin' in bal_df.columns:
            bal_df = bal_df.rename(columns={'coin': 'currency'})

        # numeric conversions
        for col in ('balance', 'locked', 'avg_buy_price'):
            if col in bal_df.columns:
                bal_df[col] = pd.to_numeric(bal_df[col], errors='coerce')

        # If currency column named differently
        if 'currency' not in bal_df.columns and 'currency_name' in bal_df.columns:
            bal_df = bal_df.rename(columns={'currency_name': 'currency'})

    except Exception:
        bal_df = pd.DataFrame(columns=['currency', 'balance', 'locked', 'avg_buy_price'])

    # Normalize positions
    try:
        if positions and isinstance(positions, list):
            pos_df = pd.DataFrame(positions)
        elif positions and isinstance(positions, dict):
            pos_df = pd.DataFrame(positions.get('positions') if 'positions' in positions else [positions])
        else:
            pos_df = pd.DataFrame(columns=['symbol','side','qty','entry_price','unrealized_pnl'])
        # common conversions
        for c in ['qty','entry_price','unrealized_pnl','unrealized_pnl_usdt','avg_price']:
            if c in pos_df.columns:
                pos_df[c] = pd.to_numeric(pos_df[c], errors='coerce')
        if 'unrealized_pnl' not in pos_df.columns and 'unrealized_pnl_usdt' in pos_df.columns:
            pos_df['unrealized_pnl'] = pos_df['unrealized_pnl_usdt']
        if 'symbol' not in pos_df.columns and 'ticker' in pos_df.columns:
            pos_df = pos_df.rename(columns={'ticker':'symbol'})
        if 'side' not in pos_df.columns:
            for cand in ('position_side','direction'):
                if cand in pos_df.columns:
                    pos_df['side'] = pos_df[cand]
                    break
    except Exception:
        pos_df = pd.DataFrame(columns=['symbol','side','qty','entry_price','unrealized_pnl'])

    # Compute KRW-equivalent values for balances using backend price fetch
    try:
        bal_df = bal_df.copy()
        bal_df['currency'] = bal_df['currency'].astype(str)
        # build market tickers for non-KRW currencies (e.g., BTC -> KRW-BTC)
        markets = []
        currency_to_market = {}
        for cur in bal_df['currency'].unique():
            cur_up = cur.upper()
            if cur_up == 'KRW' or cur_up.startswith('KRW'):
                continue
            market = f"KRW-{cur_up}"
            markets.append(market)
            currency_to_market[cur_up] = market

        price_map: dict = {}
        if markets:
            # fetch latest close price for those markets via backend batch (count=1)
            try:
                kline_map = fetch_klines_batch_from_backend(markets, timeframe='minute1', count=1)
                for m, df in kline_map.items():
                    price = None
                    try:
                        if df is not None and hasattr(df, 'empty') and not df.empty and 'close' in df.columns:
                            # get last available close
                            price = float(df['close'].iloc[-1])
                    except Exception:
                        price = None
                    price_map[m] = price
            except Exception:
                price_map = {}

        # add price_krw and value_krw columns
        def _get_price_krw(cur):
            try:
                cu = str(cur).upper()
                if cu == 'KRW' or cu.startswith('KRW'):
                    return 1.0
                m = currency_to_market.get(cu)
                if not m:
                    return None
                return price_map.get(m)
            except Exception:
                return None

        bal_df['price_krw'] = bal_df['currency'].apply(_get_price_krw)
        # value_krw: for KRW currency, balance is already KRW; otherwise balance * price_krw
        def _compute_value(row):
            try:
                bal = float(row.get('balance') or 0)
            except Exception:
                bal = 0.0
            pr = row.get('price_krw')
            cur = str(row.get('currency') or '').upper()
            if cur == 'KRW' or cur.startswith('KRW'):
                return bal
            if pr is None:
                return None
            return bal * float(pr)

        bal_df['value_krw'] = bal_df.apply(_compute_value, axis=1)

    except Exception:
        # fallback to original simple KRW sum
        bal_df['price_krw'] = None
        bal_df['value_krw'] = None

    # Top metrics: use server-reported KRW cash and compute converted asset value from available prices only
    krw_cash = float(reported_krw_from_server or 0.0)
    conv_sum = 0.0
    try:
        if not bal_df.empty and 'value_krw' in bal_df.columns and 'currency' in bal_df.columns:
            # only include non-KRW assets where value_krw is a real number (not None/NaN)
            nonkrw = bal_df[~bal_df['currency'].astype(str).str.upper().str.contains('KRW', na=False)]
            conv_sum = float(nonkrw['value_krw'].dropna().sum())
    except Exception:
        conv_sum = 0.0

    # Compute total unrealized PnL
    total_unrealized = 0.0
    try:
        if not pos_df.empty and 'unrealized_pnl' in pos_df.columns:
            total_unrealized = pos_df['unrealized_pnl'].dropna().sum()
    except Exception:
        total_unrealized = 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric('원화 현금 잔고', f"{krw_cash:,.0f} 원")
    c2.metric('환산 자산 가치(원)', f"{conv_sum:,.0f} 원")
    c3.metric('총 미확정 손익', f"{total_unrealized:,.0f} 원")

    # Show balances table with KRW conversion columns
    try:
        # Prepare display dataframe
        if not bal_df.empty:
            disp = bal_df[['currency', 'balance', 'price_krw', 'value_krw']].copy()

            def _fmt_amount(v):
                try:
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return '-'
                    f = float(v)
                    if abs(f) >= 1:
                        return f"{f:,.0f}"
                    return f"{f:.8f}"
                except Exception:
                    return str(v)

            def _fmt_price(v):
                try:
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return '-'
                    f = float(v)
                    if f >= 1:
                        return f"{f:,.0f}"
                    return f"{f:.8f}"
                except Exception:
                    return str(v)

            disp['잔고'] = disp['balance'].apply(_fmt_amount)
            disp['현재가(원)'] = disp['price_krw'].apply(_fmt_price)
            disp['가치(원)'] = disp['value_krw'].apply(lambda v: _fmt_amount(v) if v is not None and not pd.isna(v) else '-')
            disp = disp.rename(columns={'currency': '화폐'})
            disp = disp[['화폐', '잔고', '현재가(원)', '가치(원)']]
            _safe_dataframe(disp, hide_index=True)

            # show total caption and note missing price entries
            missing_prices = bal_df[bal_df['value_krw'].isna() & ~bal_df['currency'].astype(str).str.upper().str.contains('KRW')]['currency'].tolist()
            st.caption(f"추정 포트폴리오 총액(원화, 현금+환산): {krw_cash+conv_sum:,.0f} 원")
            if missing_prices:
                st.info(f"아래 종목은 현재가가 없어 환산에서 제외되었습니다: {', '.join(missing_prices)}")
        else:
            st.info('잔고 정보가 없습니다.')
    except Exception as e:
        st.warning(f'잔고 테이블 표시 중 오류: {e}')

    st.divider()

    # Positions table and PnL chart
    try:
        if pos_df.empty:
            st.info('보유 포지션이 없습니다.')
        else:
            disp_cols = [c for c in ['symbol','side','qty','entry_price','avg_price','unrealized_pnl'] if c in pos_df.columns]
            disp = pos_df[disp_cols].copy()
            for c in ['qty','entry_price','avg_price','unrealized_pnl']:
                if c in disp.columns:
                    disp[c] = pd.to_numeric(disp[c], errors='coerce')
            _safe_dataframe(disp.fillna(''), hide_index=True)

            if 'unrealized_pnl' in pos_df.columns:
                pnl = pos_df[['symbol','unrealized_pnl']].dropna(subset=['unrealized_pnl']).sort_values('unrealized_pnl', ascending=False)
                if not pnl.empty:
                    colors = ['red' if v>0 else 'blue' for v in pnl['unrealized_pnl']]
                    fig2 = go.Figure(data=[go.Bar(x=pnl['symbol'], y=pnl['unrealized_pnl'], marker=dict(color=colors))])
                    fig2.update_layout(title='포지션별 미확정 손익', xaxis_title='심볼', yaxis_title='손익 (원)', height=320, margin=dict(l=10,r=10,t=30,b=30))
                    st.plotly_chart(fig2, width='stretch')
    except Exception as e:
        st.error(f'포지션 표시 중 오류: {e}')


def render_watcher_page(cfg: Dict[str, Any]):
    st.title('이벤트 감시자 관리')
    st.write('백엔드의 이벤트 감시자(Watcher)를 시작/중지할 수 있습니다. 웹소켓 대신 폴링 기반으로 동작하도록 구성할 수 있어요.')

    watch_market = st.text_input('감시할 마켓', value=cfg.get('market', 'KRW-BTC'))
    watch_interval = st.number_input('체크 간격(초)', min_value=0.1, value=float(cfg.get('watch_interval', 1.0)))
    vol_k = float(cfg.get('strategy_params', {}).get('VolatilityBreakout', {}).get('k_value', 0.5))
    cb_vol_k = st.slider('변동성 K 값', min_value=0.0, max_value=1.0, value=vol_k)
    cb_vol_mul = st.number_input('거래량 폭증 배수', min_value=1, value=int(cfg.get('vol_spike_multiplier', 3)))

    colx, coly = st.columns(2)
    with colx:
        if st.button('Watcher 시작'):
            payload = {
                'market': watch_market,
                'interval': float(watch_interval),
                'callbacks': [
                    {'type': 'volatility_breakout', 'k': float(cb_vol_k)},
                    {'type': 'volume_spike', 'multiplier': int(cb_vol_mul)},
                ]
            }
            try:
                resp = api_request('post', '/watcher/start', json=payload, timeout=10)
                st.success('Watcher 시작 요청을 보냈습니다.')
            except Exception as e:
                st.error(f'Watcher 시작 오류: {e}')
    with coly:
        if st.button('Watcher 중지'):
            try:
                resp = api_request('post', '/watcher/stop', timeout=10)
                st.success('Watcher 중지 요청을 보냈습니다.')
            except Exception as e:
                st.error(f'Watcher 중지 오류: {e}')

    # Show recent watcher status if available
    try:
        resp = api_request('get', '/watcher/status', timeout=5)
        if resp and resp.status_code == 200:
            st.subheader('Watcher 상태')
            st.json(resp.json())
    except Exception:
        pass


# --- Main app logic ---
# Sidebar button menu for page selection (user prefers buttons, fixed order)
sb = st.sidebar
sb.title('업비트 트레이더')
sb.caption('원하는 기능으로 바로 가기')

# Fixed NAV order requested by user
NAV_OPTIONS = [
    '종목스크리닝',
    '이벤트 감시자 관리',
    '원화잔고 및 포지션 분석',
    '설정 편집',
]

# Ensure page state exists
if 'page' not in st.session_state:
    st.session_state['page'] = NAV_OPTIONS[0]

selected_tab = st.session_state['page']
nav_container = sb.container()
pending_selection = None
for nav_option in NAV_OPTIONS:
    is_selected = selected_tab == nav_option
    btn_type = 'primary' if is_selected else 'secondary'
    try:
        clicked = nav_container.button(
            nav_option,
            key=f"nav_btn_{nav_option.replace(' ', '_')}",
            width='stretch',
            type=btn_type,
        )
    except TypeError:
        # Older Streamlit doesn't support `type` or `width` kwargs; try fallbacks
        try:
            clicked = nav_container.button(nav_option, key=f"nav_btn_{nav_option.replace(' ', '_')}", use_container_width=True)
        except Exception:
            clicked = nav_container.button(nav_option, key=f"nav_btn_{nav_option.replace(' ', '_')}")
    if clicked:
        pending_selection = nav_option

if pending_selection and pending_selection != selected_tab:
    st.session_state['page'] = pending_selection
    # trigger a rerun so UI reflects selection immediately
    try:
        rerun_fn = getattr(st, 'rerun', None) or getattr(st, 'experimental_rerun', None)
        if rerun_fn:
            rerun_fn()
    except Exception:
        pass

# page dispatch using session state
page = st.session_state.get('page', NAV_OPTIONS[0])
if page == '종목스크리닝':
    render_screening_page()
elif page == '이벤트 감시자 관리':
    render_watcher_page(cfg)
elif page == '원화잔고 및 포지션 분석':
    try:
        render_positions_page()
    except Exception as e:
        st.error(f'원화잔고/포지션 페이지 렌더링 중 오류: {e}')
elif page == '설정 편집':
    render_config_page(cfg)
