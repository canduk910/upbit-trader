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

try:
    import pyupbit
except Exception:
    pyupbit = None

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

    # 파일 수정 시간을 가져오는 헬퍼 메서드
    def _get_config_mtime(self):
        try:
            return os.path.getmtime(self._config_path)
        except Exception:
            return None

    # 설정 값을 로드하여 인스턴스 변수에 저장
    def _load_config_values(self):
        # copy some frequently used values to bot instance
        self.market = config.MARKET
        self.timeframe = config.TIMEFRAME
        self.candle_count = config.CANDLE_COUNT
        self.loop_interval = config.LOOP_INTERVAL_SEC
        self.trade_amount_krw = config.TRADE_AMOUNT_KRW

    # 설정 변경 시 컴포넌트 재초기화
    # 전략, 자금 관리자, AI 분석기 등을 재설정
    # 호출 시점: 초기화 시 및 설정 변경 감지 시
    # 각 컴포넌트는 config 모듈의 최신 설정값을 사용하여 초기화
    def _reinit_components(self):
        # 전략 초기화 및 재초기화
        try:
            strategy_name = config.STRATEGY_NAME.lower()
            log.info(f"Initializing strategy: {strategy_name}")

            # RSI 전략
            if strategy_name == 'rsi':
                self.strategy = RSIStrategy(
                    period=config.RSI_PERIOD,
                    oversold_threshold=config.RSI_OVERSOLD,
                    overbought_threshold=config.RSI_OVERBOUGHT,
                )
            # 볼래틸리티 돌파 전략
            elif strategy_name == 'volatilitybreakout':
                self.strategy = VolatilityBreakoutStrategy(k=config.VB_K_VALUE)
            # 듀얼 모멘텀 전략
            elif strategy_name == 'dualmomentum':
                self.strategy = DualMomentumStrategy(window=config.DM_WINDOW)
            else:
                raise ValueError(f"Unknown strategy: {config.STRATEGY_NAME}")

            # 자금 관리자 초기화
            if config.USE_KELLY_CRITERION:
                self.money_manager = KellyCriterionManager(
                    win_rate=config.KELLY_WIN_RATE,
                    payoff_ratio=config.KELLY_PAYOFF_RATIO,
                    fraction=config.KELLY_FRACTION,
                )
            else:
                self.money_manager = None

            # AI 분석기 초기화
            if EnsembleAnalyzer is not None:
                try:
                    self.ai = EnsembleAnalyzer() # config 기반 초기화
                except Exception as e:
                    log.warning(f"Failed to init EnsembleAnalyzer: {e}")
                    self.ai = None
            else:
                self.ai = None

            # 설정값 재로드
            self._load_config_values()

            log.info("Components initialized/reinitialized from config.")
        except Exception as e:
            log.error(f"Error initializing components: {e}")
            raise

    # 설정 파일 변경 감지 및 재로드 메서드
    # 파일 변경을 감지하면 config 모듈을 reload하고 컴포넌트를 재초기화
    def _detect_and_reload_config(self):
        try:
            # check file mtime to detect changes
            current_mtime = self._get_config_mtime()
            # if changed, reload config and reinit components
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

    # 초기 포지션 상태 확인 및 동기화
    # 보유 코인이 있으면 True, 없으면 False 반환
    # 확인 후 self.in_position 초기화에 사용
    # 보유 코인이 있으면 매도 대기 상태로 간주
    # 없으면 매수 대기 상태로 간주
    # 초기화 시 한 번만 호출
    def check_initial_position(self):
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

    # AI 분석용 데이터 전처리 메서드
    # 기술지표 계산 및 DataFrame 반환
    # klines: UpbitAPI에서 조회한 캔들 데이터 리스트
    # 반환: 기술지표가 추가된 pandas DataFrame
    # klines 데이터에서 필요한 컬럼만 추출하고 타입 변환
    # RSI, 볼린저 밴드, SMA, MACD 등 주요 지표 계산
    # 반환된 DataFrame은 build_trading_context에서 사용
    # 호출 시점: build_trading_context 내부
    # klines 데이터는 UpbitAPI.get_klines() 결과
    # 반환된 DataFrame은 시간순으로 정렬되어야 함
    def process_data_for_ai(self, klines):
        df = pd.DataFrame(klines)
        # 필요한 컬럼만 추출
        df = df[['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price',
                 'candle_acc_trade_volume']]
        # 타입 변환
        df = df.astype({'opening_price': float, 'high_price': float, 'low_price': float, 'trade_price': float,
                        'candle_acc_trade_volume': float})
        # 컬럼명 변경
        df = df.rename(
            columns={'candle_date_time_kst': 'time', 'opening_price': 'open', 'high_price': 'high',
                     'low_price': 'low', 'trade_price': 'close', 'candle_acc_trade_volume': 'volume'})
        # 기술지표 계산
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14, fillna=True).rsi() # RSI
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2, fillna=True) # 볼린저 밴드
        df['bb_upper'] = bb.bollinger_hband() # 상단 밴드
        df['bb_lower'] = bb.bollinger_lband() # 하단 밴드
        df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20, fillna=True).sma_indicator() # 20일 이동평균
        df['sma_60'] = ta.trend.SMAIndicator(close=df['close'], window=60, fillna=True).sma_indicator() # 60일 이동평균
        macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True) # MACD
        df['macd'] = macd.macd() # MACD 값
        df['macd_signal'] = macd.macd_signal() # MACD 시그널 값
        return df

    # TradingContext JSON(dict) 구성 메서드
    # klines: UpbitAPI에서 조회한 캔들 데이터 리스트
    # 반환: TradingContext 구조의 dict
    # 1) 기술지표용 DataFrame 생성 (process_data_for_ai 재사용)
    # 2) 계좌/포지션 정보 조회 및 구성
    # 3) 오더북 조회 (pyupbit 사용, 실패해도 무시)
    # 4) 미체결 주문 조회 (현재는 빈 리스트)
    # 5) 타임프레임 정보 구성 (self.timeframe 기준)
    # 반환된 dict는 AI 분석기에서 사용
    def build_trading_context(self, klines):
        """Upbit 시세/계좌 정보를 기반으로 TradingContext JSON(dict)를 구성한다."""
        # 1) 기술지표용 DataFrame 생성 (기존 로직 재사용)
        df = self.process_data_for_ai(klines)

        if df is None or df.empty:
            raise ValueError('No data for building TradingContext')

        df = df.sort_values('time').reset_index(drop=True)
        last_row = df.iloc[-1]
        symbol = self.market  # 예: 'KRW-BTC'
        quote, base = symbol.split('-')

        # 일간 등락률/거래대금은 단일 타임프레임 기준의 근사값으로 계산
        if len(df) > 1:
            prev_close = float(df['close'].iloc[-2])
            day_change_pct = float((last_row['close'] / prev_close - 1.0) * 100.0) if prev_close else 0.0
        else:
            day_change_pct = 0.0
        day_volume_krw = float((df['close'] * df['volume']).sum())

        # 2) 계좌/포지션 정보
        total_equity_krw = 0.0
        available_krw = 0.0
        positions = []
        this_position = None

        try:
            balances = self.api.get_balances() or []
        except Exception as e:
            log.warning(f'Failed to fetch balances for AI context: {e}')
            balances = []

        # 각 화폐별 잔고 조회
        for b in balances:
            currency = b.get('currency') # 화폐 코드, e.g., 'KRW', 'BTC', 'ETH'
            balance = float(b.get('balance', 0) or 0) # 보유 수량
            locked = float(b.get('locked', 0) or 0) # 주문 중 묶여있는 수량
            avg_buy_price = float(b.get('avg_buy_price', 0) or 0) # 평균 매수가

            # KRW 잔고는 별도 처리
            if currency == 'KRW':
                available_krw = balance
                total_equity_krw += balance
                continue

            # KRW 마켓 기준으로만 평가 (필요 시 BTC/USDT 마켓 확장 가능)
            if balance <= 0 and locked <= 0:
                continue

            market_symbol = f'KRW-{currency}'
            # 현재 심볼의 현재가는 df 기준 마지막 종가 사용
            if market_symbol == symbol:
                current_price = float(last_row['close'])
            else:
                current_price = avg_buy_price  # 다른 코인은 보수적으로 평단가 기준

            # 평가금액 계산
            notional = (balance + locked) * current_price
            # 전체 평가금액에 합산
            total_equity_krw += notional

            # 포지션 정보 구성
            pos = {
                'symbol': market_symbol,        # 예: 'KRW-BTC'
                'side': 'LONG',                 # 업비트는 현물만 지원하므로 항상 LONG
                'size': balance + locked,       # 보유 수량 + 주문 중 묶여있는 수량
                'entry_price': avg_buy_price,   # 평균 매수가
                'avg_price': avg_buy_price,     # 평균 매수가
                'unrealized_pnl': None,         # 미실현 손익 (업비트 API에서 제공하지 않음)
                'leverage': 1.0,                # 레버리지 (현물은 1배)
                'notional_krw': notional,       # 평가금액 (KRW)
            }
            positions.append(pos) # 전체 포지션 리스트에 추가

            if market_symbol == symbol:
                this_position = pos # 현재 심볼의 포지션 정보 저장

        # 3) 오더북 (pyupbit가 있을 때만 조회, 실패해도 무시)
        orderbook = None
        if 'pyupbit' in globals() and pyupbit is not None:
            try:
                ob_list = pyupbit.get_orderbook(tickers=symbol) # 리스트 반환
                if ob_list:
                    ob = ob_list[0]
                    units = ob.get('orderbook_units', [])[:5]  # 상위 5호가까지만
                    bids = [{'price': u['bid_price'], 'size': u['bid_size']} for u in units] # 매수호가
                    asks = [{'price': u['ask_price'], 'size': u['ask_size']} for u in units] # 매도호가
                    orderbook = {
                        'timestamp': ob.get('timestamp'),
                        'bids': bids,
                        'asks': asks,
                    }
            except Exception as e:
                log.warning(f'Failed to fetch orderbook for AI context: {e}')

        # 4) 미체결 주문 (현재 UpbitAPI에 없으므로 일단 빈 리스트)
        open_orders = []

        # 5) 타임프레임 정보 구성 (현재는 self.timeframe 하나만 사용)
        tf_key = self.timeframe
        # 마지막 캔들 정보
        last_candle = {
            'time': str(last_row['time']),
            'open': float(last_row['open']),
            'high': float(last_row['high']),
            'low': float(last_row['low']),
            'close': float(last_row['close']),
            'volume': float(last_row['volume']),
        }
        #  주요 기술지표
        indicators = {
            'close': float(last_row['close']),
            'rsi': float(last_row['rsi']) if not pd.isna(last_row['rsi']) else None,
            'bb_upper': float(last_row['bb_upper']) if not pd.isna(last_row['bb_upper']) else None,
            'bb_lower': float(last_row['bb_lower']) if not pd.isna(last_row['bb_lower']) else None,
            'sma_20': float(last_row['sma_20']) if not pd.isna(last_row['sma_20']) else None,
            'sma_60': float(last_row['sma_60']) if not pd.isna(last_row['sma_60']) else None,
            'macd': float(last_row['macd']) if not pd.isna(last_row['macd']) else None,
            'macd_signal': float(last_row['macd_signal']) if not pd.isna(last_row['macd_signal']) else None,
            'recent_closes': df['close'].tail(60).tolist(),
        }
        # 타임프레임별 데이터 구조
        timeframes = {
            tf_key: {
                'last_candle': last_candle,
                'indicators': indicators,
            }
        }

        # TradingContext dict 구성
        # 최종 반환 구조
        trading_context = {
            'meta': {
                'exchange': 'UPBIT',
                'market_type': 'SPOT',
                'symbol': symbol,
                'quote_currency': quote,
                'generated_at_kst': str(last_row['time']),
                'ai_hint': {
                    'strategy': getattr(config, 'STRATEGY_NAME', 'UNKNOWN'),
                    'loop_interval_sec': self.loop_interval,
                },
            },
            'constraints': {
                'min_order_krw': float(getattr(config, 'MIN_ORDER_AMOUNT', 5000)),
                'per_trade_max_krw': float(self.trade_amount_krw),
                'allow_short': False,
                'use_leverage': False,
            },
            'account': {
                'total_equity_krw': float(total_equity_krw),
                'available_krw': float(available_krw),
                'positions': positions,
                'open_orders': open_orders,
            },
            'markets': [
                {
                    'symbol': symbol,
                    'base': base,
                    'quote': quote,
                    'day_change_pct': day_change_pct,
                    'day_volume_krw': day_volume_krw,
                    'timeframes': timeframes,
                    'orderbook': orderbook,
                    'position': this_position,
                }
            ],
        }

        return trading_context

    # 메인 루프
    # 시세 조회 -> 전략 신호 생성 -> AI 확인 -> 매매 실행
    # 무한 루프, 예외 처리 포함
    # KeyboardInterrupt 시 종료
    # 각 단계별 로그 출력
    # 1. 시세 데이터 조회
    # 2. 전략 신호 생성
    # 3. AI 분석 (선택적)
    # 4. 매매 결정 및 실행
    # 5. 설정 변경 감지 및 재로드
    # 6. 루프 대기
    # 반복
    def run(self):
        log.info("Bot main loop started. Monitoring market...")

        # 메인 루프
        while True:
            try:
                # detect runtime config changes and reload if needed
                self._detect_and_reload_config() # 설정 변경 감지 및 재로드

                # 1. 시세 데이터 조회
                klines = self.api.get_klines(self.market, self.timeframe, self.candle_count)
                if not klines:
                    log.warning("Empty klines data. Retrying...")
                    time.sleep(1)
                    continue

                raw_df = pd.DataFrame(klines) # 원본 DataFrame 생성

                # 2. 전략 신호 생성
                technical_signal = self.strategy.generate_signals(raw_df)

                final_decision = 'HOLD' # 기본 결정은 HOLD

                # 3. AI 분석 및 최종 매매 결정
                # 매수 신호 처리
                if technical_signal == 'BUY' and not self.in_position:
                    log.info(f"🚀 Technical Signal [BUY] detected! Asking AI Ensemble for confirmation...")
                    trading_context = self.build_trading_context(klines) # TradingContext 구성
                    ai_decision = None
                    if self.ai:
                        try:
                            ai_decision = self.ai.analyze(trading_context) # AI 분석
                        except Exception as e:
                            log.warning(f"AI analysis failed: {e}")

                    # AI가 BUY 권고하거나 AI가 없을 때 기술신호만으로 BUY 결정
                    if ai_decision == 'BUY' or (self.ai is None and technical_signal == 'BUY'):
                        final_decision = 'BUY'
                        log.info("✅ Decision: BUY")
                    else:
                        log.info(f"❌ AI Ensemble REJECTED the BUY signal (AI said: {ai_decision}). Holding.")

                # 매도 신호 처리
                elif technical_signal == 'SELL' and self.in_position:
                    log.info(f"📉 Technical Signal [SELL] detected! Asking AI Ensemble for confirmation...")
                    trading_context = self.build_trading_context(klines)
                    ai_decision = None
                    if self.ai:
                        try:
                            ai_decision = self.ai.analyze(trading_context)
                        except Exception as e:
                            log.warning(f"AI analysis failed: {e}")

                    if ai_decision == 'SELL' or (self.ai is None and technical_signal == 'SELL'):
                        final_decision = 'SELL'
                        log.info("✅ Decision: SELL")
                    else:
                        log.info(f"❌ AI Ensemble REJECTED the SELL signal (AI said: {ai_decision}). Holding.")

                # 4. 매매 실행
                self.execute_trade(final_decision)

                # 5. 루프 대기
                time.sleep(self.loop_interval)

            except KeyboardInterrupt:
                log.info("Trading Bot stopped by user.")
                break
            except Exception as e:
                log.error(f"Critical error in main loop: {e}", exc_info=True)
                time.sleep(5)

    # 매매 실행 메서드
    # decision: 'BUY', 'SELL', 'HOLD'
    # 매수/매도 주문 실행 및 로그 출력
    # self.in_position 상태 업데이트
    # 매수 시 KRW 잔고 기준으로 주문
    # 매도 시 보유 코인 전량 시장가 매도
    # 최소 주문 금액 체크 및 잔고 부족 시 경고
    # 매수/매도 성공 시 로그 출력
    # 호출 시점: run() 메서드 내에서 매매 결정 후 호출
    # 매매 실행 담당
    def execute_trade(self, decision):
        # 매수 처리
        if decision == 'BUY':
            # calculate trade amount (use kelly if configured)
            krw_balance = self.api.get_balance("KRW") # KRW 잔고 조회
            trade_amount = self.trade_amount_krw # default trade amount

            # Kelly 기준으로 매수 금액 조정
            if hasattr(self, 'money_manager') and self.money_manager:
                trade_amount = self.money_manager.calculate_trade_amount(krw_balance) # 켈리 기준 매수 금액 계산

            # 최소 주문 금액 체크
            if trade_amount < config.MIN_ORDER_AMOUNT:
                log.warning(f"Trade amount ({trade_amount:,.0f} KRW) is below the minimum order amount ({config.MIN_ORDER_AMOUNT:,.0f} KRW). Skipping buy order.")
                return

            # 잔고 부족 체크
            if krw_balance < trade_amount:
                log.warning(f"Insufficient balance. Required: {trade_amount:,.0f} KRW, Available: {krw_balance:,.0f} KRW")
                return

            # 매수 주문 실행
            log.info(f"Attempting to place a BUY order for {trade_amount:,.0f} KRW.")
            result = self.api.place_order(self.market, 'bid', price=trade_amount, ord_type='price')

            # 주문 성공 시 상태 업데이트
            if result:
                log.info(f"Successfully placed BUY order: {result}")
                self.in_position = True

        # 매도 처리
        elif decision == 'SELL':
            coin_symbol = self.market.split('-')[1] # 'KRW-BTC' -> 'BTC'
            balance_coin = self.api.get_balance(coin_symbol) # 보유 코인 잔고 조회

            # 보유 코인 체크
            if balance_coin is None or balance_coin <= 0:
                log.warning(f"No {coin_symbol} balance to sell.")
                return

            # 매도 주문 실행
            log.info(f"Attempting to place a SELL order for {balance_coin} {coin_symbol}.")
            result = self.api.place_order(self.market, 'ask', volume=balance_coin, ord_type='market')
            if result:
                log.info(f"Successfully placed SELL order: {result}")
                self.in_position = False


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()