import json
import threading
import time
import websocket  # websocket-client
from typing import Any, Dict, List, Optional
from server.ws_listener_base import BaseWebsocketListener


class PublicWebsocketlistener(BaseWebsocketListener):
    def __init__(self, redis_client: Optional[Any] = None):
        super().__init__(redis_client=redis_client, log_name='UpbitWSPublic', log_file='ws_listener_public.log')

    def subscription_payload(self) -> str:
        codes = self._targets
        message = [
            {"ticket": "upbit-ws-public"},
            {"type": "ticker", "codes": codes, "isOnlyRealtime": False},
        ]
        return json.dumps(message)

    def auth_headers(self) -> List[str]:
        return []

    def on_payload(self, payload: Dict[str, Any]) -> None:
        if payload.get('type') != 'ticker':
            raise ValueError('Unexpected payload type for public listener')
        self._aggregate_candle(payload)


def main():
    listener = PublicWebsocketlistener()
    listener.run()


if __name__ == '__main__':
    main()
