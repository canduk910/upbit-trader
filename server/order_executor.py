from dataclasses import dataclass, field
import math
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

import server.config as config
from server.history import order_history_store
from server.logger import log
from server.upbit_api import UpbitAPI


@dataclass
class OrderRequest:
    action: str
    symbol: str
    amount_krw: float = 0.0
    volume: float = 0.0
    reason: str = ""
    callback: Optional[Callable[[bool, Dict[str, Any]], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    trade_amount: Optional[float] = None
    target_quantity: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    market_factor: Optional[float] = None
    risk_amount: Optional[float] = None


class OrderExecutor:
    def __init__(self, api: UpbitAPI, min_order_amount: float = config.MIN_ORDER_AMOUNT):
        self.api = api
        self.min_order_amount = float(min_order_amount)
        self._queue: queue.Queue[Optional[OrderRequest]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()
        log.info("OrderExecutor thread started.")

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=3)
        log.info("OrderExecutor thread stopped.")

    def submit(self, request: OrderRequest) -> None:
        if request.action not in ("BUY", "SELL"):
            log.warning(f"Unsupported order action: {request.action}")
            return
        self._queue.put(request)
        log.info(f"OrderRequest queued: {request.action} {request.symbol}")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._queue.get(timeout=1)
            except queue.Empty:
                continue
            if request is None:
                break
            result = self._process_request(request)
            self._queue.task_done()
            if request.callback:
                try:
                    request.callback(result.get("success", False), result)
                except Exception as exc:
                    log.warning(f"Order callback failed: {exc}")

    def _process_request(self, request: OrderRequest) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "ts": time.time(),
            "action": request.action,
            "symbol": request.symbol,
            "amount_krw": request.amount_krw,
            "volume": request.volume,
            "reason": request.reason,
            "metadata": request.metadata,
            "trade_amount": request.trade_amount,
            "target_quantity": request.target_quantity,
            "entry_price": request.entry_price,
            "stop_loss_price": request.stop_loss_price,
            "take_profit_price": request.take_profit_price,
            "market_factor": request.market_factor,
            "risk_amount": request.risk_amount,
        }

        if request.action == "BUY":
            payload = self._handle_buy(request)
        else:
            payload = self._handle_sell(request)

        entry["success"] = payload.get("success", False)
        entry["details"] = payload.get("message")
        entry["response"] = payload.get("response")
        order_history_store.record(entry)
        return payload

    def _handle_buy(self, request: OrderRequest) -> Dict[str, Any]:
        if request.amount_krw < self.min_order_amount:
            msg = f"주문금액({request.amount_krw:.0f} KRW)이 최소 주문금액({self.min_order_amount:.0f} KRW)보다 작습니다."
            log.warning(msg)
            return {"success": False, "message": msg}

        available_krw = self.api.get_balance("KRW") or 0.0
        if available_krw < request.amount_krw:
            msg = f"원화 잔액 부족. 필요: {request.amount_krw:.0f}, 현재: {available_krw:.0f}"
            log.warning(msg)
            return {"success": False, "message": msg}

        if request.entry_price:
            log.info(
                "[Bracket] entry=%.0f stop=%.0f take=%.0f qty=%.6f risk=%.0f factor=%.2f"
                % (
                    request.entry_price,
                    request.stop_loss_price or 0,
                    request.take_profit_price or 0,
                    request.target_quantity or 0,
                    request.risk_amount or 0,
                    request.market_factor or 0,
                )
            )

        try:
            response = self.api.place_order(request.symbol, "bid", ord_type="price", price=request.amount_krw)
            log.info(f"BUY order executed: {request.symbol} {request.amount_krw:.0f} KRW")
            return {"success": True, "message": "BUY order executed", "response": response}
        except Exception as exc:
            msg = f"BUY 주문 실패: {exc}"
            log.error(msg, exc_info=True)
            return {"success": False, "message": msg}

    def _handle_sell(self, request: OrderRequest) -> Dict[str, Any]:
        coin_symbol = request.symbol.split("-")[-1]
        total_volume = self.api.get_balance(coin_symbol)

        if not total_volume or math.isclose(total_volume, 0.0):
            msg = f"{coin_symbol} 잔고가 없습니다."
            log.warning(msg)
            return {"success": False, "message": msg}

        # 매도 비율 결정 (기본값: 30% 매도, 나머지 70% 보유)
        sell_ratio = float(request.metadata.get('sell_ratio', 0.3)) if request.metadata else 0.3
        sell_ratio = max(0.1, min(1.0, sell_ratio))  # 10%~100% 범위로 제한

        volume = request.volume
        if not volume or math.isclose(volume, 0.0):
            volume = total_volume * sell_ratio

        # 최소 거래 가능 수량 체크 (보통 0.0001 이상)
        if volume < 0.0001:
            msg = f"매도 수량이 너무 적습니다: {volume:.8f}"
            log.warning(msg)
            return {"success": False, "message": msg}

        try:
            # 진입가 조회 (이전 매수 기록에서)
            entry_price = request.entry_price or request.metadata.get('entry_price', 0.0)

            response = self.api.place_order(request.symbol, "ask", ord_type="market", volume=volume)

            # 매도가 계산 (응답에서 평균 체결가 추출)
            avg_price = 0.0
            if isinstance(response, dict):
                avg_price = float(response.get('avg_price', 0) or response.get('price', 0) or 0)

            # 손익 계산
            profit_loss = 0.0
            profit_loss_pct = 0.0
            if entry_price > 0 and avg_price > 0:
                profit_loss = (avg_price - entry_price) * volume
                profit_loss_pct = ((avg_price / entry_price) - 1) * 100

            log.info(
                f"SELL order executed: {request.symbol} {volume:.8f} units "
                f"(전체의 {sell_ratio*100:.0f}%, 남은 수량: {total_volume - volume:.8f})"
            )

            if profit_loss != 0:
                log.info(
                    f"매매 손익: {profit_loss:+.2f} KRW ({profit_loss_pct:+.2f}%) "
                    f"[진입가: {entry_price:.0f}, 매도가: {avg_price:.0f}]"
                )

            return {
                "success": True,
                "message": "SELL order executed",
                "response": response,
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
                "entry_price": entry_price,
                "exit_price": avg_price,
                "volume": volume,
                "total_volume": total_volume,
                "sell_ratio": sell_ratio,
            }
        except Exception as exc:
            msg = f"SELL 주문 실패: {exc}"
            log.error(msg, exc_info=True)
            return {"success": False, "message": msg}
