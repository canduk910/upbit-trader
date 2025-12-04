import json
from typing import Any, Dict, List, Optional
import time
import uuid

import jwt

from server.config import get_setting
from server.ws_listener_base import BaseWebsocketListener


class PrivateWebsocketListener(BaseWebsocketListener):
    def __init__(self, redis_client: Optional[Any] = None):
        super().__init__(
            redis_client=redis_client,
            log_name='UpbitWSPrivate',
            log_file='ws_listener_private.log',
            ws_url='wss://api.upbit.com/websocket/v1/private',
        )
        self._jwt_token: Optional[str] = None

    def _generate_token(self) -> Optional[str]:
        access_key = get_setting('UPBIT_ACCESS_KEY')
        secret_key = get_setting('UPBIT_SECRET_KEY')
        if not access_key or not secret_key:
            self.logger.warning('Private websocket token missing in configuration')
            return None
        payload = {
            'access_key': access_key,
            'nonce': str(uuid.uuid4()),
        }
        try:
            token = jwt.encode(payload, secret_key, algorithm='HS256')
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            return token
        except Exception as exc:
            self.logger.warning(f'Failed to generate websocket token: {exc}')
            return None

    def _pre_run(self) -> bool:
        self._jwt_token = self._generate_token()
        if not self._jwt_token:
            self.logger.warning('Private websocket has no JWT yet; retrying soon')
            return False
        return True

    def subscription_payload(self) -> str:
        message = [
            {'ticket': 'upbit-ws-private'},
            {'type': 'myOrder', 'codes': self._targets},
        ]
        return json.dumps(message)

    def auth_headers(self) -> List[str]:
        if not self._jwt_token:
            return []
        return [
            f'Authorization: Bearer {self._jwt_token}',
        ]

    def on_payload(self, payload: Dict[str, Any]) -> None:
        if payload.get('type') != 'myOrder':
            return
        self._record_exec_history(payload)
