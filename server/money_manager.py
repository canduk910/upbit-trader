from server.logger import log


class KellyCriterionManager:
    """
    켈리 공식을 사용하여 투자 비중을 계산하는 자금 관리자.
    """

    def __init__(self, win_rate, payoff_ratio, fraction=1.0):
        """
        :param win_rate: 전략의 승률 (0 < win_rate < 1)
        :param payoff_ratio: 평균 이익 / 평균 손실 (payoff_ratio > 0)
        :param fraction: 계산된 켈리 비율에 적용할 비중 (e.g., 0.5 for Half Kelly)
        """
        if not (0 < win_rate < 1):
            raise ValueError("Win rate must be between 0 and 1.")
        if payoff_ratio <= 0:
            raise ValueError("Payoff ratio must be greater than 0.")
        if not (0 < fraction <= 1):
            raise ValueError("Kelly fraction must be between 0 and 1.")

        self.win_rate = win_rate
        self.payoff_ratio = payoff_ratio
        self.fraction = fraction

        # 켈리 공식: f = (p*b - (1-p)) / b
        # p = win_rate, b = payoff_ratio
        kelly_f = ((self.win_rate * self.payoff_ratio) - (1 - self.win_rate)) / self.payoff_ratio

        self.kelly_fraction = max(0, kelly_f) * self.fraction  # 음수일 경우 0으로

        log.info("Kelly Criterion Initialized.")
        log.info(f"  - Win Rate: {self.win_rate:.2%}")
        log.info(f"  - Payoff Ratio: {self.payoff_ratio:.2f}")
        log.info(f"  - Applied Kelly Fraction (f): {self.kelly_fraction:.2%}")

    def calculate_trade_amount(self, total_balance):
        """
        총 자산 대비 켈리 공식에 따른 투자 금액을 계산합니다.
        :param total_balance: 총 자산 (e.g., KRW 잔고)
        :return: (float) 투자할 금액
        """
        if self.kelly_fraction <= 0:
            log.warning("Kelly fraction is 0 or negative. No investment is advised.")
            return 0

        trade_amount = total_balance * self.kelly_fraction
        log.info(f"Calculated trade amount: {trade_amount:,.0f} KRW (Balance: {total_balance:,.0f} * Kelly: {self.kelly_fraction:.2%})")
        return trade_amount
