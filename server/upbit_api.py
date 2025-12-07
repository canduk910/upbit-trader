import jwt
import uuid
import hashlib
from urllib.parse import urlencode
import requests
import time
from typing import Optional
from server.logger import log
try:
    import pyupbit
    _HAS_PYUPBIT = True
except Exception:
    pyupbit = None
    _HAS_PYUPBIT = False

# pandas and pytz may be available via pyupbit dependencies; import safely
try:
    import pandas as pd
    import pytz
    _HAS_PANDAS = True
except Exception:
    pd = None
    pytz = None
    _HAS_PANDAS = False


class UpbitAPI:
    """
    업비트 API 연동 클래스
    JWT 인증 및 주요 API 호출 기능
    """

    def __init__(self, access_key=None, secret_key=None):
        self.access_key = access_key
        self.secret_key = secret_key
        self.server_url = "https://api.upbit.com"

    def _generate_auth_token(self, query_params=None):
        """
        API 요청을 위한 JWT 인증 토큰 생성 (키가 없으면 None 반환)
        """
        if not self.access_key or not self.secret_key:
            return None

        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
        }

        # 쿼리 파라미터가 있는 경우, 페이로드에 해시값 추가
        if query_params:
            query_string = urlencode(query_params, doseq=True).encode("utf-8")
            m = hashlib.sha512()
            m.update(query_string)
            query_hash = m.hexdigest()
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'

        # JWT 생성
        jwt_token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        authorize_token = f"Bearer {jwt_token}"
        return authorize_token

    def _send_request(self, method, endpoint, params=None, data=None, headers=None):
        """
        업비트 API에 요청 전송
        """
        url = self.server_url + endpoint

        # 바디 및 쿼리 설정
        query = params if method in ['GET', 'DELETE'] else None
        body = data if method in ['POST', 'PUT'] else None

        auth_token = self._generate_auth_token(query or body)

        if headers is None:
            headers = {}
        if auth_token:
            headers['Authorization'] = auth_token

        # Implement retry on 429 and some transient network errors
        max_retries = 5
        backoff_base = 0.5
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                if method == 'GET':
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                elif method == 'POST':
                    headers['Content-Type'] = 'application/json'
                    response = requests.post(url, headers=headers, json=data, timeout=10)
                elif method == 'DELETE':
                    response = requests.delete(url, headers=headers, params=params, timeout=10)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # If rate-limited, backoff and retry
                if response.status_code == 429:
                    wait = backoff_base * (2 ** (attempt - 1))
                    jitter = random.uniform(0, 0.5)
                    total_wait = wait + jitter
                    log.warning(f'Upbit API rate limit hit (429). retry {attempt}/{max_retries} after {total_wait:.2f}s')
                    time.sleep(total_wait)
                    last_exc = requests.exceptions.HTTPError('429 Too Many Requests')
                    continue

                response.raise_for_status()
                # small pause to avoid tight loops
                time.sleep(0.12)
                return response.json()
            except requests.exceptions.HTTPError as http_err:
                last_exc = http_err
                status = getattr(getattr(http_err, 'response', None), 'status_code', None)
                # if not rate-limited, don't retry further
                if status != 429:
                    txt = getattr(http_err, 'response', None)
                    extra = ''
                    try:
                        if txt is not None:
                            extra = f" - {txt.text}"
                    except Exception:
                        extra = ''
                    log.error(f"HTTP error occurred: {http_err}{extra}")
                    break
                # else loop will retry
            except requests.exceptions.RequestException as req_e:
                last_exc = req_e
                # transient network error -> backoff and retry
                wait = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                log.warning(f'Network error during Upbit API call: {req_e}. retry {attempt}/{max_retries} after {wait:.2f}s')
                time.sleep(wait)
                continue
            except Exception as e:
                last_exc = e
                log.error(f"An error occurred: {e}")
                break

        # Retries exhausted or non-retryable error
        return None

    # --- Public API Methods ---

    def get_balances(self):
        """
        전체 계좌 잔고 조회 (인증 필요)
        """
        endpoint = "/v1/accounts"
        return self._send_request('GET', endpoint)

    def get_balance(self, ticker="KRW"):
        """
        특정 화폐/코인의 잔고 조회
        """
        balances = self.get_balances()
        if balances:
            for b in balances:
                if b.get('currency') == ticker:
                    try:
                        return float(b.get('balance', 0))
                    except Exception:
                        return 0.0
        return 0.0

    def get_klines(self, market, timeframe="minute1", count=200):
        """
        캔들(시세) 조회: timeframe 문자열을 Upbit 엔드포인트로 매핑하여 호출합니다.
        지원 예: 'minute1', 'minute3', 'minute5', 'minute15', 'minute60', 'day'
        """
        # 매핑
        tf = timeframe.lower()
        if tf.startswith('minute'):
            # minuteN -> /v1/candles/minutes/N
            try:
                n = int(tf.replace('minute', ''))
                endpoint = f"/v1/candles/minutes/{n}"
            except Exception:
                endpoint = "/v1/candles/minutes/1"
        elif tf in ('day', 'days'):
            endpoint = "/v1/candles/days"
        elif tf in ('week', 'weeks'):
            endpoint = "/v1/candles/weeks"
        elif tf in ('month', 'months'):
            endpoint = "/v1/candles/months"
        else:
            # fallback
            endpoint = f"/v1/candles/{tf}"

        # Try pyupbit if available for convenience and reliability
        tf = timeframe.lower()
        if _HAS_PYUPBIT:
            try:
                # pyupbit uses periods like "minute" string: use the numeric mapping
                if tf.startswith('minute'):
                    n = int(tf.replace('minute',''))
                    interval_str = f"minute{n}"
                    df = pyupbit.get_ohlcv(market, interval=interval_str, count=count)
                elif tf in ('day','days'):
                    df = pyupbit.get_ohlcv(market, interval='day', count=count)
                elif tf in ('week','weeks'):
                    df = pyupbit.get_ohlcv(market, interval='week', count=count)
                elif tf in ('month','months'):
                    df = pyupbit.get_ohlcv(market, interval='month', count=count)
                else:
                    # fallback to requests-based endpoint
                    df = None
                if df is not None:
                    records = []
                    # convert index to KST-aware ISO strings when possible
                    for idx, row in df.iterrows():
                        ts_str = None
                        try:
                            if _HAS_PANDAS and isinstance(idx, pd.Timestamp):
                                # if naive, assume UTC and localize; then convert to Asia/Seoul
                                if idx.tzinfo is None:
                                    idx_utc = idx.tz_localize('UTC')
                                else:
                                    idx_utc = idx.tz_convert('UTC') if idx.tzinfo else idx.tz_localize('UTC')
                                if pytz:
                                    kst = pytz.timezone('Asia/Seoul')
                                    idx_kst = idx_utc.tz_convert(kst)
                                    ts_str = idx_kst.isoformat()
                                else:
                                    ts_str = idx_utc.isoformat()
                            else:
                                # fallback: str(idx)
                                ts_str = str(idx)
                        except Exception:
                            ts_str = str(idx)
                        records.append({'candle_date_time_kst': ts_str, 'opening_price': row['open'], 'high_price': row['high'], 'low_price': row['low'], 'trade_price': row['close'], 'candle_acc_trade_volume': row['volume']})
                    return records
            except Exception as e:
                log.warning(f'pyupbit get_klines failed, falling back to HTTP: {e}')

        params = {'market': market, 'count': count}
        return self._send_request('GET', endpoint, params=params)

    def get_orderbook(self, markets):
        if isinstance(markets, (list, tuple)):
            market_param = ','.join(markets)
        else:
            market_param = markets
        data = self._send_request('GET', '/v1/orderbook', params={'markets': market_param})
        if data and isinstance(data, list):
            return data
        return None

    def place_order(self, market: str, side: str, ord_type: str = 'price', price: Optional[float] = None, volume: Optional[float] = None):
        payload = {
            'market': market,
            'side': side,
            'ord_type': ord_type,
        }
        if ord_type == 'price':
            if price is None:
                raise ValueError('price is required for ord_type="price"')
            payload['price'] = price
        elif ord_type == 'market':
            if volume is None:
                raise ValueError("volume is required for market sell (ord_type='market')")
            payload["volume"] = str(volume)
        elif ord_type == 'limit':
            if price is None or volume is None:
                raise ValueError('price and volume are required for ord_type="limit"')
            payload['price'] = price
            payload['volume'] = volume
        else:
            raise ValueError(f"Unsupported ord_type: {ord_type}")

        return self._send_request('POST', '/v1/orders', data=payload)
