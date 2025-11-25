import pandas as pd
import ta
from logger import log


class RSIStrategy:
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