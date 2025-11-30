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
    def __init__(self, win_rate, payoff_ratio, fraction=1.0):

        #:param win_rate: 전략의 승률 (0 < win_rate < 1)
        #:param payoff_ratio: 평균 이익 / 평균 손실 (payoff_ratio > 0)
        #:param fraction: 계산된 켈리 비율에 적용할 비중 (e.g., 0.5 for Half Kelly)
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

        # 켈리 공식: f = (p*b - (1-p)) / b
        # p = win_rate, b = payoff_ratio
        kelly_f = ((self.win_rate * self.payoff_ratio) - (1 - self.win_rate)) / self.payoff_ratio

        # 음수일 경우 0으로 설정하여 손실 방지
        self.kelly_fraction = max(0, kelly_f) * self.fraction  # 음수일 경우 0으로

        log.info("Kelly Criterion Initialized.")
        log.info(f"  - Win Rate: {self.win_rate:.2%}")
        log.info(f"  - Payoff Ratio: {self.payoff_ratio:.2f}")
        log.info(f"  - Applied Kelly Fraction (f): {self.kelly_fraction:.2%}")

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
