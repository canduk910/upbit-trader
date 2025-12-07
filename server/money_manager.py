from server.logger import log

#  Kelly Criterion 기반 자금 관리자
# Kelly Criterion: f* = (p*b - (1-p)) / b
# f* = 최적 베팅 비율
# p = 승률 (0 < p < 1)
# b = 평균 이익 / 평균 손실 (payoff ratio, b > 0)
# 참고: https://en.wikipedia.org/wiki/Kelly_criterion
# 예시: 승률 60%, 손익비 1.5인 전략의 경우
# f* = (0.6*1.5 - 0.4) / 1.5 = 0.4 (즉, 자산의 40%를 투자)
# 단, 실제 투자에서는 변동성을 고려하여 절반 켈리(Half Kelly) 등을 권장합니다.

class KellyCriterionManager:

    # 켈리 기준 자금 관리자 초기화
    def __init__(self, win_rate=0.4, payoff_ratio=2.0, fraction=0.5, risk_per_trade=0.02):

        #:param win_rate: 전략의 승률 (0 < win_rate < 1)
        #:param payoff_ratio: 평균 이익 / 평균 손실 (payoff_ratio > 0)
        #:param fraction: 계산된 켈리 비율에 적용할 비중 (e.g., 0.5 for Half Kelly)
        #:param risk_per_trade: 1회 트레이딩 당 계좌 대비 최대 손실 허용 비율 (기본 2%)
        if not (0 < win_rate < 1):
            raise ValueError("Win rate must be between 0 and 1.")
        if payoff_ratio <= 0:
            raise ValueError("Payoff ratio must be greater than 0.")
        if not (0 < fraction <= 1):
            raise ValueError("Kelly fraction must be between 0 and 1.")

        # 멤버 변수 설정
        self.win_rate = win_rate
        self.payoff_ratio = payoff_ratio
        self.fraction = fraction
        self.risk_per_trade = risk_per_trade

        # 켈리 공식: f = (p*b - (1-p)) / b
        # p = win_rate, b = payoff_ratio
        kelly_f = ((self.win_rate * self.payoff_ratio) - (1 - self.win_rate)) / self.payoff_ratio

        # 음수일 경우 0으로 설정하여 손실 방지
        self.kelly_fraction = max(0, kelly_f) * self.fraction  # 음수일 경우 0으로

        log.info("Kelly Criterion Initialized.")
        log.info(f"  - Win Rate: {self.win_rate:.2%}")
        log.info(f"  - Payoff Ratio: {self.payoff_ratio:.2f}")
        log.info(f"  - Applied Kelly Fraction (f): {self.kelly_fraction:.2%}")
        log.info(f"  - Risk per Trade: {self.risk_per_trade:.2%}")

    # 총 자산 대비 투자 금액 계산
    # 총 자산의 일정 비율을 켈리 공식에 따라 투자 금액으로 산정
    # e.g., 총 자산 1,000,000 KRW, 켈리 비율 20% -> 200,000 KRW 투자
    # 반환값: 투자할 금액 (float)
    #:param total_balance: 총 자산 (e.g., KRW 잔고)
    #:return: (float) 투자할 금액
    # 계산된 투자 금액 반환
    # 총 자산이 0이거나 켈리 비율이 0인 경우 0 반환
    # 로그에 계산된 투자 금액 출력
    # e.g., "Calculated trade amount: 200,000 KRW (Balance: 1,000,000 * Kelly: 20.00%)"
    # 반환값: 투자할 금액 (float)
    def calculate_trade_amount(self, total_balance):
        if self.kelly_fraction <= 0:
            log.warning("Kelly fraction is 0 or negative. No investment is advised.")
            return 0

        # 총 자산이 0인 경우 0 반환
        if total_balance <= 0:
            log.warning("Total balance is 0 or negative. No investment can be made.")
            return 0

        trade_amount = total_balance * self.kelly_fraction
        log.info(f"Calculated trade amount: {trade_amount:,.0f} KRW (Balance: {total_balance:,.0f} * Kelly: {self.kelly_fraction:.2%})")
        return trade_amount

    # 매수 진입 시 적절한 투자 금액과 수량 계산
    # 리스크 관리와 켈리 기준을 모두 고려하여 최종 투자 금액 산정
    #   [로직 설명]
    #    1. 리스크 기반 계산 (Survival): 손절 나갔을 때 계좌의 R(2%)만 잃도록 수량 설정.
    #       - 공식: 허용 손실금액 / (매수가 - 손절가)
    #    2. 켈리 기반 계산 (Growth): 켈리 비중만큼 투자.
    #    3. 최종 결정: 두 값 중 더 '안전한(작은)' 값을 선택하고, 시장 상황(market_factor)을 곱함.
    #
    #    # input
    #    :param total_balance: 현재 계좌 총액 (KRW)
    #    :param entry_price: 진입 예정 가격
    #    :param stop_loss_price: 손절 예정 가격
    #    :param market_factor: 시장 상황 점수 (0.0 ~ 1.0, 1.0=상승장 풀배팅, 0.5=보수적, 0.0=매매금지)

    #    # return: {
    #        'trade_amount': 투자할 총 금액 (KRW),
    #        'quantity': 구매할 코인 수량,
    #        'risk_amount': 예상 손실 금액,
    #        'reason': 계산 근거
    #    }
    def get_position_size(self, total_balance, entry_price, stop_loss_price, market_factor=1.0):
        if total_balance <= 0 or entry_price <= 0:
            return {'trade_amount': 0, 'quantity': 0, 'reason': "Invalid Balance or Price"}

        # 1. 1회 최대 허용 손실금액 (R) 계산
        # 예: 1억 계좌, 2% 리스크 -> 200만원 (엑셀의 '2% 금액')
        max_risk_amount = total_balance * self.risk_per_trade

        # 2. 주당 리스크 (Stop Loss Gap)
        # 예: 1000원에 사서 920원에 손절 -> 주당 80원 리스크
        risk_per_share = entry_price - stop_loss_price

        if risk_per_share <= 0:
            log.warning("Stop loss price is higher than entry price (Long Position). Adjusting Logic.")
            risk_per_share = entry_price * 0.01  # 방어 코드

        stop_loss_percent = risk_per_share / entry_price

        # 3. 리스크 기반 최대 진입 수량 (Position Sizing based on Risk)
        # 공식: 내가 감당할 총 손실 / 주당 손실
        quantity_by_risk = max_risk_amount / risk_per_share
        amount_by_risk = quantity_by_risk * entry_price

        # 4. 켈리 기반 투자 금액 (Position Sizing based on Kelly)
        amount_by_kelly = total_balance * self.kelly_fraction

        # 5. 최종 금액 선정 (안전한 쪽 선택) & 시장 팩터 반영
        # 켈리가 너무 공격적이면 리스크 관리 룰을 따르고,
        # 리스크 룰이 너무 크면 켈리(자금성장 최적화)를 따름.
        safe_factor = max(0.0, min(1.0, market_factor if market_factor is not None else 1.0))
        final_amount = min(amount_by_risk, amount_by_kelly) * safe_factor

        # 최소 주문 금액(예: 업비트 5000원) 처리 등은 봇 로직에서 하겠지만 여기선 0 처리
        final_amount = max(0, final_amount)
        final_quantity = final_amount / entry_price

        # 로그 출력
        log.info(f"[Money Mgmt] Balance: {total_balance:,.0f} KRW | Market Factor: {market_factor}")
        log.info(f" -> Risk Limit (R={self.risk_per_trade:.1%}): Max Loss {max_risk_amount:,.0f} KRW")
        log.info(f" -> Stop Loss: {stop_loss_percent:.2%} (Gap: {risk_per_share:,.0f} KRW)")
        log.info(f" -> Sizing (Risk): {amount_by_risk:,.0f} KRW vs (Kelly): {amount_by_kelly:,.0f} KRW")
        log.info(f" -> Final Check: Invest {final_amount:,.0f} KRW (Qty: {final_quantity:.4f})")

        return {
            'trade_amount': final_amount,
            'quantity': final_quantity,
            'risk_amount': final_quantity * risk_per_share,  # 실제 예상 손실액
            'stop_loss_pct': stop_loss_percent
        }
