import time
import pandas as pd
import ta
import config
from upbit_api import UpbitAPI
from strategy import RSIStrategy
from ai_analyst import EnsembleAnalyzer
from logger import log


class TradingBot:
    def __init__(self):
        log.info("========== [Upbit Auto Trading Bot Started] ==========")
        log.info(f"Market: {config.MARKET}, Strategy: RSI + AI Ensemble ({config.ENSEMBLE_STRATEGY})")

        # 1. 설정 로드
        self.market = config.MARKET
        self.trade_amount_krw = config.TRADE_AMOUNT_KRW
        self.timeframe = config.TIMEFRAME
        self.candle_count = config.CANDLE_COUNT
        self.loop_interval = config.LOOP_INTERVAL_SEC

        # 2. 모듈 초기화
        try:
            self.api = UpbitAPI(config.UPBIT_ACCESS_KEY, config.UPBIT_SECRET_KEY)

            # 1차 필터: RSI 전략
            self.strategy = RSIStrategy(
                period=config.RSI_PERIOD,
                oversold_threshold=config.RSI_OVERSOLD,
                overbought_threshold=config.RSI_OVERBOUGHT
            )

            # 2차 필터: AI 앙상블 (OpenAI + Gemini)
            self.ai = EnsembleAnalyzer()

        except Exception as e:
            log.error(f"Initialization Failed: {e}")
            raise e

        # 3. 초기 자산 상태 확인
        self.in_position = self.check_initial_position()
        log.info(f"Initial Position Status: {'HOLDING (매도 대기)' if self.in_position else 'NO POSITION (매수 대기)'}")

    def check_initial_position(self):
        """
        시작 시 보유 코인이 있는지 확인하여 상태를 동기화
        """
        try:
            coin_ticker = self.market.split('-')[1]  # 'KRW-BTC' -> 'BTC'
            balance = self.api.get_balance(ticker=coin_ticker)

            # 현재가 조회 (가치 계산용)
            current_price_data = self.api.get_klines(self.market, "minute1", 1)
            if not current_price_data:
                return False

            current_price = float(current_price_data[0]['trade_price'])

            # 평가금액이 5,000원 이상이면 보유 중으로 간주
            if balance * current_price > 5000:
                log.info(
                    f"Found existing holding: {balance} {coin_ticker} (approx. {balance * current_price:,.0f} KRW)")
                return True
            return False
        except Exception as e:
            log.error(f"Error checking initial position: {e}")
            return False

    def process_data_for_ai(self, klines):
        """
        AI에게 제공하기 위해 보조지표를 추가하여 데이터프레임을 풍부하게 만듦
        """
        df = pd.DataFrame(klines)
        df = df[['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price',
                 'candle_acc_trade_volume']]
        df = df.astype({'opening_price': float, 'high_price': float, 'low_price': float, 'trade_price': float,
                        'candle_acc_trade_volume': float})
        df = df.rename(
            columns={'candle_date_time_kst': 'time', 'opening_price': 'open', 'high_price': 'high', 'low_price': 'low',
                     'trade_price': 'close', 'candle_acc_trade_volume': 'volume'})
        df = df.sort_index(ascending=True).reset_index(drop=True)

        # 보조지표 추가 (AI 분석용)
        # 1. RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # 2. 볼린저 밴드
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()

        # 3. 이동평균선 (골든크로스/데드크로스 판단용)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_60'] = ta.trend.sma_indicator(df['close'], window=60)

        # 4. MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        return df

    def run(self):
        log.info("Bot main loop started. Monitoring market...")

        while True:
            try:
                # 1. 시세 데이터 조회
                klines = self.api.get_klines(self.market, self.timeframe, self.candle_count)
                if not klines:
                    log.warning("Empty klines data. Retrying...")
                    time.sleep(1)
                    continue

                # 2. 데이터 가공 (Strategy용 간단 DF)
                # Strategy 클래스는 내부적으로 필요한 컬럼을 찾아 씁니다.
                raw_df = pd.DataFrame(klines)

                # 3. [1차 필터] 기술적 지표 분석
                # RSIStrategy는 'trade_price'를 사용합니다.
                technical_signal = self.strategy.generate_signals(raw_df)

                final_decision = 'HOLD'

                # 4. [2차 필터] 기술적 신호가 있을 때만 AI에게 질문 (비용 절감)
                if technical_signal == 'BUY' and not self.in_position:
                    log.info(f"🚀 Technical Signal [BUY] detected! Asking AI Ensemble for confirmation...")

                    # AI 분석용 데이터 준비 (보조지표 포함)
                    ai_df = self.process_data_for_ai(klines)

                    # AI 앙상블 호출
                    ai_decision = self.ai.analyze(ai_df, "no_position")

                    if ai_decision == 'BUY':
                        final_decision = 'BUY'
                        log.info("✅ AI Ensemble APPROVED the BUY signal.")
                    else:
                        log.info(f"❌ AI Ensemble REJECTED the BUY signal (AI said: {ai_decision}). Holding.")

                elif technical_signal == 'SELL' and self.in_position:
                    log.info(f"📉 Technical Signal [SELL] detected! Asking AI Ensemble for confirmation...")

                    ai_df = self.process_data_for_ai(klines)
                    ai_decision = self.ai.analyze(ai_df, "in_position")

                    if ai_decision == 'SELL':
                        final_decision = 'SELL'
                        log.info("✅ AI Ensemble APPROVED the SELL signal.")
                    else:
                        log.info(f"❌ AI Ensemble REJECTED the SELL signal (AI said: {ai_decision}). Holding.")

                # 5. 최종 주문 실행
                self.execute_trade(final_decision)

                # 루프 대기
                time.sleep(self.loop_interval)

            except KeyboardInterrupt:
                log.info("Trading Bot stopped by user.")
                break
            except Exception as e:
                log.error(f"Critical error in main loop: {e}", exc_info=True)
                time.sleep(5)

    def execute_trade(self, decision):
        """
        주문 실행 및 상태 업데이트
        """
        if decision == 'BUY':
            # KRW 잔고 확인
            krw_balance = self.api.get_balance("KRW")
            if krw_balance >= self.trade_amount_krw:
                log.info(f"Attempting Market Buy: {self.trade_amount_krw} KRW")
                result = self.api.place_order(self.market, 'bid', 'price', price=self.trade_amount_krw)

                if result and 'uuid' in result:
                    self.in_position = True
                    log.info(f"*** BUY ORDER COMPLETE *** UUID: {result.get('uuid')}")
                else:
                    log.error(f"Buy Order Failed: {result}")
            else:
                log.warning(f"Insufficient KRW Balance: {krw_balance} < {self.trade_amount_krw}")

        elif decision == 'SELL':
            # 코인 잔고 확인
            coin_ticker = self.market.split('-')[1]
            balance = self.api.get_balance(coin_ticker)

            # 최소 거래 수량 체크는 API에서 에러를 뱉어주므로 일단 진행
            if balance > 0:
                log.info(f"Attempting Market Sell: {balance} {coin_ticker}")
                result = self.api.place_order(self.market, 'ask', 'market', volume=balance)

                if result and 'uuid' in result:
                    self.in_position = False
                    log.info(f"*** SELL ORDER COMPLETE *** UUID: {result.get('uuid')}")
                else:
                    log.error(f"Sell Order Failed: {result}")
            else:
                log.warning("No coin balance to sell.")

        # HOLD인 경우 로그 생략 (너무 시끄러움)


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()