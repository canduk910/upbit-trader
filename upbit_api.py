import jwt
import uuid
import hashlib
from urllib.parse import urlencode
import requests
import time
from logger import log


class UpbitAPI:
    """
    업비트 API 연동 클래스
    JWT 인증 및 주요 API 호출 기능
    """

    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key
        self.server_url = "https://api.upbit.com"

    def _generate_auth_token(self, query_params=None):
        """
        API 요청을 위한 JWT 인증 토큰 생성
        """
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

        # 쿼리 파라미터 설정 (GET, DELETE 요청 시)
        query = None
        if method in ['GET', 'DELETE'] and params:
            query = params

        # 바디 파라미터 설정 (POST, PUT 요청 시)
        body = None
        if method in ['POST', 'PUT'] and data:
            body = data

        # GET/DELETE는 query를, POST/PUT은 body를 기준으로 토큰 생성
        auth_token = self._generate_auth_token(query or body)

        if headers is None:
            headers = {}
        headers['Authorization'] = auth_token

        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, json=data)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # API 속도 제한 (초당 10회, 분당 600회) - 간단한 딜레이
            time.sleep(0.15)  # 150ms 대기 (초당 약 6.6회)

            response.raise_for_status()  # 4xx, 5xx 에러 발생 시 예외 처리
            return response.json()

        except requests.exceptions.HTTPError as http_err:
            log.error(f"HTTP error occurred: {http_err} - {response.text}")
        except Exception as e:
            log.error(f"An error occurred: {e}")
        return None

    # --- Public API Methods ---

    def get_balances(self):
        """
        전체 계좌 잔고 조회
        """
        endpoint = "/v1/accounts"
        return self._send_request('GET', endpoint)

    def get_balance(self, ticker="KRW"):
        """
        특정 화폐/코인의 잔고 조회
        :param ticker: 'KRW', 'BTC', 'ETH' 등
        :return: (float) 사용 가능한 잔고
        """
        balances = self.get_balances()
        if balances:
            for b in balances:
                if b['currency'] == ticker:
                    return float(b['balance'])
        return 0.0

    def get_klines(self, market, timeframe="minute1", count=200):
        """
        캔들(시세) 조회
        :param market: 마켓 코드 (예: KRW-BTC)
        :param timeframe: 캔들 단위 (예: minute1, minute5, day)
        :param count: 조회할 캔들 수 (최대 200)
        """
        endpoint = f"/v1/candles/{timeframe}"
        params = {
            "market": market,
            "count": count
        }
        # 업비트 캔들은 오래된 순 -> 최신 순으로 정렬되어 반환됨
        return self._send_request('GET', endpoint, params=params)

    def place_order(self, market, side, ord_type, volume=None, price=None):
        """
        주문 실행
        :param market: 마켓 코드 (예: KRW-BTC)
        :param side: 'bid' (매수), 'ask' (매도)
        :param ord_type: 'limit' (지정가), 'price' (시장가 매수), 'market' (시장가 매도)
        :param volume: (지정가, 시장가 매도 시) 주문 수량
        :param price: (지정가, 시장가 매수 시) 주문 가격 또는 주문 총액(KRW)
        """
        endpoint = "/v1/orders"
        data = {
            "market": market,
            "side": side,
            "ord_type": ord_type,
        }

        if ord_type == 'limit':
            if volume is None or price is None:
                raise ValueError("limit order requires 'volume' and 'price'")
            data['volume'] = str(volume)
            data['price'] = str(price)

        elif ord_type == 'price':  # 시장가 매수 (KRW 기준)
            if side != 'bid' or price is None:
                raise ValueError("market buy (price) order requires 'side=bid' and 'price' (total KRW)")
            data['price'] = str(price)

        elif ord_type == 'market':  # 시장가 매도 (코인 수량 기준)
            if side != 'ask' or volume is None:
                raise ValueError("market sell (market) order requires 'side=ask' and 'volume' (coin amount)")
            data['volume'] = str(volume)
        else:
            raise ValueError(f"Invalid ord_type: {ord_type}")

        log.info(f"Placing order: {data}")
        return self._send_request('POST', endpoint, data=data)

    def get_order_status(self, uuid):
        """
        개별 주문 상태 조회
        :param uuid: 주문 UUID
        """
        endpoint = "/v1/order"
        params = {"uuid": uuid}
        return self._send_request('GET', endpoint, params=params)

    def cancel_order(self, uuid):
        """
        주문 취소
        :param uuid: 주문 UUID
        """
        endpoint = "/v1/order"
        params = {"uuid": uuid}
        return self._send_request('DELETE', endpoint, params=params)