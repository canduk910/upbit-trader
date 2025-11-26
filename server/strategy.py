import pandas as pd
import ta
from logger import log


class Strategy:
    """전략 클래스의 기본 인터페이스"""

    def generate_signals(self, klines_df):
        """
        매매 신호를 생성하는 메서드.
        :param klines_df: 캔들 데이터프레임
        :return: (str) 'BUY', 'SELL', 'HOLD'
        """
        raise NotImplementedError("generate_signals() must be implemented in subclass.")


class RSIStrategy(Strategy):
    """
    RSI(상대강도지수) 기반 매매 전략
    - 과매도 구간(OVERSOLD) 진입 시 매수
    - 과매수 구간(OVERBOUGHT) 진입 시 매도
    """

    def __init__(self, period, oversold_threshold, overbought_threshold):
        self.period = period
        self.oversold = oversold_threshold
        self.overbought = overbought_threshold

        if not (0 < self.oversold < 100 and 0 < self.overbought < 100 and self.oversold < self.overbought):
            raise ValueError("RSI thresholds are invalid.")

    def _calculate_rsi(self, klines_df):
        """
        RSI 지표 계산
        :param klines_df: 업비트 캔들 데이터프레임
        :return: (pd.Series) RSI 지표
        """
        try:
            # 'trade_price' (종가) 기준으로 RSI 계산
            rsi_indicator = ta.momentum.RSIIndicator(close=klines_df['trade_price'], window=self.period)
            return rsi_indicator.rsi()
        except Exception as e:
            log.error(f"Error calculating RSI: {e}")
            return None

    def generate_signals(self, klines_df):
        """
        매매 신호 생성
        :param klines_df: 캔들 데이터프레임
        :return: (str) 'BUY', 'SELL', 'HOLD'
        """
        klines_df['rsi'] = self._calculate_rsi(klines_df)

        if klines_df['rsi'] is None or klines_df.empty:
            log.warning("RSI calculation failed or empty data. Holding.")
            return 'HOLD'

        # 마지막 캔들(현재)의 RSI 값
        last_rsi = klines_df['rsi'].iloc[-1]
        # 직전 캔들의 RSI 값 (진입/이탈 확인용)
        prev_rsi = klines_df['rsi'].iloc[-2]

        log.debug(f"Current RSI: {last_rsi:.2f} (Previous: {prev_rsi:.2f})")

        # 과매도 구간 진입 시 매수 신호
        if last_rsi > self.oversold and prev_rsi <= self.oversold:
            log.info(f"RSI crossed UP oversold line ({self.oversold}). Signal: BUY")
            return 'BUY'

        # 과매수 구간 진입 시 매도 신호
        if last_rsi < self.overbought and prev_rsi >= self.overbought:
            log.info(f"RSI crossed DOWN overbought line ({self.overbought}). Signal: SELL")
            return 'SELL'

        # 그 외에는 홀드
        return 'HOLD'


class VolatilityBreakoutStrategy(Strategy):
    """
    변동성 돌파 전략 (래리 윌리엄스)
    - 당일 시가 + (전일 고가 - 전일 저가) * k 만큼 상승 시 매수
    - 주로 일봉 데이터와 함께 사용됩니다.
    """

    def __init__(self, k=0.5):
        self.k = k
        if not 0 < k < 1:
            log.warning(f"k value {k} is unusual. It's typically between 0.4 and 0.6.")

    def generate_signals(self, klines_df):
        """
        매수 신호 생성 (매도는 별도 전략 필요)
        :param klines_df: 일봉 캔들 데이터프레임
        :return: (str) 'BUY', 'HOLD'
        """
        if len(klines_df) < 2:
            log.warning("Volatility breakout strategy requires at least 2 days of data.")
            return 'HOLD'

        try:
            prev_day = klines_df.iloc[-2]
            today = klines_df.iloc[-1]

            # 변동폭 = 전일 고가 - 전일 저가
            volatility_range = prev_day['high_price'] - prev_day['low_price']
            # 매수 목표가 = 당일 시가 + 변동폭 * k
            target_price = today['opening_price'] + volatility_range * self.k

            log.debug(f"Volatility Breakout Target Price: {target_price:.2f} (Today's High: {today['high_price']:.2f})")

            # 당일 고가가 목표가를 돌파하면 매수 신호
            if today['high_price'] >= target_price:
                log.info(f"Price broke the target price ({target_price:.2f}). Signal: BUY")
                return 'BUY'

        except Exception as e:
            log.error(f"Error in VolatilityBreakoutStrategy: {e}")
            return 'HOLD'

        # 본 전략은 주로 진입 시점 포착에 사용되므로, 매도 신호는 생성하지 않음
        return 'HOLD'


class DualMomentumStrategy(Strategy):
    """
    듀얼 모멘텀 전략 (절대 모멘텀만 구현)
    - 특정 기간의 수익률이 0보다 크면 '매수', 그렇지 않으면 '매도'(현금화)
    - 완전한 듀얼 모멘텀은 여러 자산 비교(상대 모멘텀)가 필요합니다.
    """

    def __init__(self, window=12):
        self.window = window
        log.info(f"Dual Momentum (Absolute) initialized with window: {self.window} periods.")

    def generate_signals(self, klines_df):
        """
        절대 모멘텀 기반 매매 신호 생성
        :param klines_df: 캔들 데이터프레임 (주봉 또는 월봉 권장)
        :return: (str) 'BUY', 'SELL', 'HOLD'
        """
        if len(klines_df) < self.window + 1:
            log.warning(f"Dual momentum strategy requires at least {self.window + 1} data points.")
            return 'HOLD'

        try:
            # N 기간 전 종가 대비 현재 종가의 수익률 계산
            momentum = (klines_df['trade_price'].iloc[-1] / klines_df['trade_price'].iloc[-self.window - 1]) - 1
            log.debug(f"Absolute Momentum ({self.window} periods): {momentum:.2%}")

            # 모멘텀이 0보다 크면 추세 지속으로 보고 매수/유지
            if momentum > 0:
                log.info(f"Positive momentum ({momentum:.2%}). Signal: BUY")
                return 'BUY'
            # 모멘텀이 0보다 작으면 추세 이탈로 보고 매도
            else:
                log.info(f"Negative momentum ({momentum:.2%}). Signal: SELL")
                return 'SELL'

        except Exception as e:
            log.error(f"Error in DualMomentumStrategy: {e}")
            return 'HOLD'

        return 'HOLD'


"""
--- 전략 관련 참고사항 ---
- 켈리 공식 (Kelly Criterion):
  - 매매 '시점'을 결정하는 전략이 아니라, 특정 거래에 얼마만큼의 '자금을 할당'할지
    결정하는 수학적인 자금 관리 기법입니다.
  - 승률과 손익비 데이터를 기반으로 장기 수익을 극대화하는 최적의 베팅 비율을 계산합니다.
  - RSI, 변동성 돌파 등 다른 신호 생성 전략과 '함께' 사용될 수 있습니다.
"""
