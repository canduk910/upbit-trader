import os
import time
import pandas as pd
import ta
from server.upbit_api import UpbitAPI
from server.strategy import RSIStrategy, VolatilityBreakoutStrategy, DualMomentumStrategy
from server.money_manager import KellyCriterionManager
import server.config as config
from server.logger import log

# ai analyzer may be optional; import if available
try:
    from server.ai_analyst import EnsembleAnalyzer
except Exception:
    EnsembleAnalyzer = None


class TradingBot:
    def __init__(self):
        log.info("========== [Upbit Auto Trading Bot Started] ==========")
        log.info(f"Market: {config.MARKET}, Strategy: {config.STRATEGY_NAME} + AI Ensemble ({config.ENSEMBLE_STRATEGY})")

        # runtime config path for change detection
        self._config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runtime', 'config.json')
        self._config_mtime = self._get_config_mtime()

        # 1. 설정 로드
        self._load_config_values()

        # 2. 모듈 초기화
        try:
            self.api = UpbitAPI(config.UPBIT_ACCESS_KEY, config.UPBIT_SECRET_KEY)
        except Exception as e:
            log.error(f"Failed to initialize UpbitAPI: {e}")
            raise

        # initialize strategy/money manager/ai
        self._reinit_components()

        # 3. 초기 자산 상태 확인
        self.in_position = self.check_initial_position()
        log.info(f"Initial Position Status: {'HOLDING (매도 대기)' if self.in_position else 'NO POSITION (매수 대기)'}")

    def _get_config_mtime(self):
        try:
            return os.path.getmtime(self._config_path)
        except Exception:
            return None

    def _load_config_values(self):
        # copy some frequently used values to bot instance
        self.market = config.MARKET
        self.timeframe = config.TIMEFRAME
        self.candle_count = config.CANDLE_COUNT
        self.loop_interval = config.LOOP_INTERVAL_SEC
        self.trade_amount_krw = config.TRADE_AMOUNT_KRW

    def _reinit_components(self):
        # re-create strategy and money manager based on current config
        try:
            strategy_name = config.STRATEGY_NAME.lower()
            log.info(f"Initializing strategy: {strategy_name}")
            if strategy_name == 'rsi':
                self.strategy = RSIStrategy(
                    period=config.RSI_PERIOD,
                    oversold_threshold=config.RSI_OVERSOLD,
                    overbought_threshold=config.RSI_OVERBOUGHT,
                )
            elif strategy_name == 'volatilitybreakout':
                self.strategy = VolatilityBreakoutStrategy(k=config.VB_K_VALUE)
            elif strategy_name == 'dualmomentum':
                self.strategy = DualMomentumStrategy(window=config.DM_WINDOW)
            else:
                raise ValueError(f"Unknown strategy: {config.STRATEGY_NAME}")

            # money manager
            if config.USE_KELLY_CRITERION:
                self.money_manager = KellyCriterionManager(
                    win_rate=config.KELLY_WIN_RATE,
                    payoff_ratio=config.KELLY_PAYOFF_RATIO,
                    fraction=config.KELLY_FRACTION,
                )
            else:
                self.money_manager = None

            # AI ensemble
            if EnsembleAnalyzer is not None:
                try:
                    self.ai = EnsembleAnalyzer()
                except Exception as e:
                    log.warning(f"Failed to init EnsembleAnalyzer: {e}")
                    self.ai = None
            else:
                self.ai = None

            # update local cached settings
            self._load_config_values()

            log.info("Components initialized/reinitialized from config.")
        except Exception as e:
            log.error(f"Error initializing components: {e}")
            raise

    def _detect_and_reload_config(self):
        """파일 변경을 감지하면 config 모듈을 reload하고 컴포넌트를 재초기화."""
        try:
            current_mtime = self._get_config_mtime()
            if current_mtime and self._config_mtime and current_mtime != self._config_mtime:
                log.info("Runtime config.json changed. Reloading configuration...")
                config.reload_config()
                self._reinit_components()
                self._config_mtime = current_mtime
            elif current_mtime and not self._config_mtime:
                # initial mtime set
                self._config_mtime = current_mtime
        except Exception as e:
            log.error(f"Error during config reload detection: {e}")

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
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_60'] = ta.trend.sma_indicator(df['close'], window=60)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        return df

    def run(self):
        log.info("Bot main loop started. Monitoring market...")

        while True:
            try:
                # detect runtime config changes and reload if needed
                self._detect_and_reload_config()

                # 1. 시세 데이터 조회
                klines = self.api.get_klines(self.market, self.timeframe, self.candle_count)
                if not klines:
                    log.warning("Empty klines data. Retrying...")
                    time.sleep(1)
                    continue

                raw_df = pd.DataFrame(klines)

                technical_signal = self.strategy.generate_signals(raw_df)

                final_decision = 'HOLD'

                if technical_signal == 'BUY' and not self.in_position:
                    log.info(f"🚀 Technical Signal [BUY] detected! Asking AI Ensemble for confirmation...")
                    ai_df = self.process_data_for_ai(klines)
                    ai_decision = None
                    if self.ai:
                        try:
                            ai_decision = self.ai.analyze(ai_df, "no_position")
                        except Exception as e:
                            log.warning(f"AI analysis failed: {e}")

                    if ai_decision == 'BUY' or (self.ai is None and technical_signal == 'BUY'):
                        final_decision = 'BUY'
                        log.info("✅ Decision: BUY")
                    else:
                        log.info(f"❌ AI Ensemble REJECTED the BUY signal (AI said: {ai_decision}). Holding.")

                elif technical_signal == 'SELL' and self.in_position:
                    log.info(f"📉 Technical Signal [SELL] detected! Asking AI Ensemble for confirmation...")
                    ai_df = self.process_data_for_ai(klines)
                    ai_decision = None
                    if self.ai:
                        try:
                            ai_decision = self.ai.analyze(ai_df, "in_position")
                        except Exception as e:
                            log.warning(f"AI analysis failed: {e}")

                    if ai_decision == 'SELL' or (self.ai is None and technical_signal == 'SELL'):
                        final_decision = 'SELL'
                        log.info("✅ Decision: SELL")
                    else:
                        log.info(f"❌ AI Ensemble REJECTED the SELL signal (AI said: {ai_decision}). Holding.")

                self.execute_trade(final_decision)

                time.sleep(self.loop_interval)

            except KeyboardInterrupt:
                log.info("Trading Bot stopped by user.")
                break
            except Exception as e:
                log.error(f"Critical error in main loop: {e}", exc_info=True)
                time.sleep(5)

    def execute_trade(self, decision):
        if decision == 'BUY':
            # calculate trade amount (use kelly if configured)
            krw_balance = self.api.get_balance("KRW")
            trade_amount = self.trade_amount_krw
            if hasattr(self, 'money_manager') and self.money_manager:
                trade_amount = self.money_manager.calculate_trade_amount(krw_balance)

            if trade_amount < config.MIN_ORDER_AMOUNT:
                log.warning(f"Trade amount ({trade_amount:,.0f} KRW) is below the minimum order amount ({config.MIN_ORDER_AMOUNT:,.0f} KRW). Skipping buy order.")
                return

            if krw_balance < trade_amount:
                log.warning(f"Insufficient balance. Required: {trade_amount:,.0f} KRW, Available: {krw_balance:,.0f} KRW")
                return

            log.info(f"Attempting to place a BUY order for {trade_amount:,.0f} KRW.")
            result = self.api.place_order(self.market, 'bid', price=trade_amount, ord_type='price')
            if result:
                log.info(f"Successfully placed BUY order: {result}")
                self.in_position = True

        elif decision == 'SELL':
            coin_symbol = self.market.split('-')[1]
            balance_coin = self.api.get_balance(coin_symbol)

            if balance_coin is None or balance_coin <= 0:
                log.warning(f"No {coin_symbol} balance to sell.")
                return

            log.info(f"Attempting to place a SELL order for {balance_coin} {coin_symbol}.")
            result = self.api.place_order(self.market, 'ask', volume=balance_coin, ord_type='market')
            if result:
                log.info(f"Successfully placed SELL order: {result}")
                self.in_position = False


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()