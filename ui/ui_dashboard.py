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
import random
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
    with st.form('config_form', clear_on_submit=False):
        strategy_name = st.selectbox('전략 선택', options=['VolatilityBreakout', 'DualMomentum', 'RSI'], index=['VolatilityBreakout', 'DualMomentum', 'RSI'].index(cfg.get('strategy_name', 'VolatilityBreakout')) if cfg.get('strategy_name') in ['VolatilityBreakout', 'DualMomentum', 'RSI'] else 0)
        market = st.text_input('Market (예: KRW-BTC)', value=cfg.get('market', 'KRW-BTC'))
        timeframe = st.text_input('Timeframe (예: minute5)', value=cfg.get('timeframe', 'minute5'))
        candle_count = st.number_input('Candle count', min_value=1, value=int(cfg.get('candle_count', 200)))
        loop_interval_sec = st.number_input('Loop interval (sec)', min_value=1, value=int(cfg.get('loop_interval_sec', 5)))

        st.subheader('주문 설정')
        order_settings = cfg.get('order_settings', {})
        min_order_amount = st.number_input('Min order amount (KRW)', min_value=1000, value=int(order_settings.get('min_order_amount', 5500)))
        trade_amount_krw = st.number_input('Trade amount (KRW)', min_value=1000, value=int(order_settings.get('trade_amount_krw', 6000)))

        st.subheader('켈리공식 설정')
        use_kelly = st.checkbox('Kelly 적용', value=bool(cfg.get('use_kelly_criterion', True)))
        kelly = cfg.get('kelly_criterion', {})
        win_rate = st.number_input('Win rate (0~1)', min_value=0.0, max_value=1.0, value=float(kelly.get('win_rate', 0.65)))
        payoff_ratio = st.number_input('Payoff ratio', min_value=0.0, value=float(kelly.get('payoff_ratio', 1.2)))
        fraction = st.number_input('Fraction (0~1)', min_value=0.0, max_value=1.0, value=float(kelly.get('fraction', 0.5)))

        st.subheader('전략별 파라미터')
        strategy_params = cfg.get('strategy_params', {})
        if strategy_name == 'RSI':
            rsi = strategy_params.get('RSI', {})
            rsi_period = st.number_input('RSI period', min_value=1, value=int(rsi.get('period', 14)))
            rsi_oversold = st.number_input('RSI oversold', min_value=0, max_value=100, value=int(rsi.get('oversold', 30)))
            rsi_overbought = st.number_input('RSI overbought', min_value=0, max_value=100, value=int(rsi.get('overbought', 70)))
        elif strategy_name == 'VolatilityBreakout':
            vb = strategy_params.get('VolatilityBreakout', {})
            k_value = st.number_input('k value (0~1)', min_value=0.0, max_value=1.0, value=float(vb.get('k_value', 0.5)))
        elif strategy_name == 'DualMomentum':
            dm = strategy_params.get('DualMomentum', {})
            window = st.number_input('Window (periods)', min_value=1, value=int(dm.get('window', 12)))

        submit = st.form_submit_button('미리보기 업데이트')

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
        # remember last query params to re-render charts on interaction
        st.session_state['last_search'] = {'market_prefix': market_prefix, 'top_n': top_n, 'timeframe': timeframe}

    if st.session_state.get('search_clicked'):
        params = st.session_state.get('last_search', {})
        try:
            resp = api_request('get', '/screen/volatility_top', params={'market_prefix': params.get('market_prefix','KRW'), 'top_n': params.get('top_n',10), 'timeframe': params.get('timeframe','minute15')}, timeout=10)
            data = resp.json().get('top', [])
            # determine requested N early
            requested_n = int(params.get('top_n', 10))
            # present a ranked table with ranks starting from 1
            df_top = None
            try:
                df_top = pd.DataFrame(data)
                # If volatility present, sort by it descending to ensure ranking order
                if 'volatility' in df_top.columns:
                    df_top = df_top.sort_values(by='volatility', ascending=False).reset_index(drop=True)
                if not df_top.empty:
                    df_top.insert(0, 'rank', range(1, len(df_top) + 1))
                # display table without the default index column
                _safe_dataframe(df_top, hide_index=True)
            except Exception:
                try:
                    df_fallback = pd.DataFrame(data)
                    _safe_dataframe(df_fallback, hide_index=True)
                except Exception:
                    # last resort: plain st.table
                    st.table(data)

            # build items list from sorted df_top when available, otherwise fallback to raw data
            if isinstance(df_top, pd.DataFrame) and not df_top.empty:
                sorted_items = df_top.to_dict(orient='records')
            else:
                sorted_items = data

            # Chart controls (placed near the table)
            control_cols = st.columns([1,1,1])
            with control_cols[0]:
                ma1 = st.number_input('MA1 window', min_value=1, value=20, key='ma1_s')
            with control_cols[1]:
                ma2 = st.number_input('MA2 window', min_value=1, value=60, key='ma2_s')
            with control_cols[2]:
                rsi_p = st.number_input('RSI period', min_value=5, value=14, key='rsi_p_s')

            # limit to requested top_n and render in a grid
            items = sorted_items[:requested_n]
            display_n = len(items)
            # Try batch fetch from backend to reduce Upbit calls
            tickers = [it.get('ticker') for it in items]
            backend_data = fetch_klines_batch_from_backend(tickers, timeframe=params.get('timeframe','minute15'), count=200)
            # If some tickers are missing in the backend result (None), retry via backend a few times
            missing = [t for t in tickers if backend_data.get(t) is None]
            if missing:
                retries = 3
                for attempt in range(1, retries+1):
                    if not missing:
                        break
                    # request only the missing tickers as a batch
                    try:
                        more = fetch_klines_batch_from_backend(missing, timeframe=params.get('timeframe','minute15'), count=200)
                        for k, v in (more or {}).items():
                            backend_data[k] = v
                        # recompute missing list
                        missing = [t for t in tickers if backend_data.get(t) is None]
                    except Exception:
                        # ignore and backoff
                        pass
                    if missing and attempt < retries:
                        time.sleep(0.8 * attempt)

            # Render rows of 2 columns based on actual display_n
            for row_start in range(0, display_n, 2):
                cols = st.columns(2, gap='small')
                for col_idx in range(2):
                    idx = row_start + col_idx
                    if idx >= display_n:
                        with cols[col_idx]:
                            st.empty()
                        continue
                    item = items[idx]
                    ticker = item.get('ticker')
                    vol = item.get('volatility')
                    with cols[col_idx]:
                        st.subheader(f"{idx+1}. {ticker} — vol={vol:.4f}")
                        df = backend_data.get(ticker)
                        # normalize/check df before plotting
                        df = _normalize_klines_df(df, min_length=50)
                        if df is None or (hasattr(df, 'empty') and df.empty):
                            # Backend could not provide data (maybe prefetch miss or Upbit limit).
                            # Show a user-friendly message and skip direct Upbit calls from the UI to avoid causing a 429.
                            st.info('차트 데이터를 아직 불러올 수 없습니다. 잠시 후 다시 시도하세요. (백엔드에서 데이터 수집 중일 수 있음)')
                        else:
                            # Create a more compact chart for the grid
                            fig = plot_candles_with_indicators(df, ticker, ma_windows=[ma1, ma2], rsi_period=int(rsi_p))
                            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=20))
                            st.plotly_chart(fig, width='stretch')
                    # slight pause to avoid tight loop/rate limits
                    time.sleep(0.08)
            # show count info
            st.caption(f"표시된 종목: {display_n} / 요청한 TopN: {requested_n}")
        except Exception as e:
            st.error(str(e))


def render_watcher_page(cfg: Dict[str, Any]):
    st.title('이벤트 감시자 관리')
    st.write('웹소켓 대신 폴링(간단) 방식으로 체크를 수행합니다. 서버의 /watcher/start, /watcher/stop를 사용합니다.')

    watch_market = st.text_input('감시할 마켓', value=cfg.get('market', 'KRW-BTC'))
    watch_interval = st.number_input('체크 간격(초)', min_value=0.2, value=1.0)
    cb_vol_k = st.slider('Volatility K', min_value=0.0, max_value=1.0, value=float(cfg.get('strategy_params', {}).get('VolatilityBreakout', {}).get('k_value', 0.5)))
    cb_vol_mul = st.number_input('Volume Spike Multiplier', min_value=1, value=3)

    colx, coly = st.columns(2)
    with colx:
        if st.button('Watcher 시작'):
            payload = {
                'market': watch_market,
                'interval': float(watch_interval),
                'callbacks': [
                    {'type': 'volatility_breakout', 'k': float(cb_vol_k)},
                    {'type': 'volume_spike', 'multiplier': int(cb_vol_mul)}
                ]
            }
            try:
                resp = api_request('post', '/watcher/start', json=payload, timeout=10)
                st.success('Watcher 시작 요청 전송됨')
            except Exception as e:
                st.error(str(e))
    with coly:
        if st.button('Watcher 중지'):
            try:
                resp = api_request('post', '/watcher/stop', timeout=10)
                st.success('Watcher 중지 요청 전송됨')
            except Exception as e:
                st.error(str(e))


def render_about_page():
    # kept as an unused fallback; keep minimal content
    st.title('도움말 / 정보')
    st.write('이 UI는 여러 페이지로 구성됩니다. 서버가 로컬에서 실행 중이어야 합니다.')


# --- Sidebar navigation (pure Streamlit) ---
def _ensure_page_state():
    if 'page' not in st.session_state:
        st.session_state['page'] = '설정 편집'


def _rerun_app() -> None:
    """Trigger a Streamlit rerun, compatible with new and old APIs."""
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn:
        try:
            rerun_fn()
        except Exception:
            # ignore if rerun not permitted in some environments
            pass


def _sidebar_menu():
     _ensure_page_state()
     NAV_OPTIONS = ['종목 스크리닝', '이벤트 감시자 관리', '원화잔고 및 포지션 분석', '설정 편집']
     sb = st.sidebar

     # Small header block (kept minimal). This parallels the user's example but uses
     # only tiny localized CSS. Unsafe HTML is limited to this small header.
     sb.markdown(
         "Upbit Trader",
     )

     # Ensure a nav state that mirrors page
     if 'nav_menu' not in st.session_state:
         st.session_state['nav_menu'] = st.session_state.get('page', '설정 편집')

     selected_tab = st.session_state['nav_menu']

     # Render menu buttons inside a container to keep layout stable
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
             # Older Streamlit versions may not support `type` or `width`.
             # Fallback to basic button (width parameter may not be supported).
             try:
                 clicked = nav_container.button(nav_option, key=f"nav_btn_{nav_option.replace(' ', '_')}", width='stretch')
             except Exception:
                 clicked = nav_container.button(nav_option, key=f"nav_btn_{nav_option.replace(' ', '_')}" )
         if clicked:
             pending_selection = nav_option

     if pending_selection and pending_selection != selected_tab:
         st.session_state['nav_menu'] = pending_selection
         st.session_state['page'] = pending_selection
         # trigger a rerun so UI reflects selection immediately
         _rerun_app()


_sidebar_menu()

# dispatch to pages based on session_state
page = st.session_state.get('page', '설정 편집')
if page == '종목 스크리닝':
    render_screening_page()
elif page == '이벤트 감시자 관리':
    render_watcher_page(cfg)
elif page == '원화잔고 및 포지션 분석':
    # Simple positions/status view — attempt to call backend /status if available
    def render_positions_page():
        st.title('원화잔고 및 포지션 분석')
        try:
            resp = requests.get(f"{API_BASE.rstrip('/')}/status", timeout=5)
            if resp.status_code == 200:
                st.json(resp.json())
            else:
                st.info('백엔드 /status 응답이 없습니다.')
        except Exception:
            st.info('백엔드 상태를 불러올 수 없습니다. 필요한 API가 동작중인지 확인하세요.')

    render_positions_page()
elif page == '설정 편집':
    render_config_page(cfg)
else:
    render_about_page()

st.divider()
st.caption('주의: 이 UI는 인증이 없습니다. 로컬 개발 환경에서만 사용하세요.')
