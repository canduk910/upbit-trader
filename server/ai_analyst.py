import json
import concurrent.futures
import server.config as config
from server.logger import log
from server.history import ai_history_store
from datetime import datetime, timezone, timedelta
import math

# OpenAI SDK
# Defensive imports
try:
    import openai
except Exception:
    openai = None

# Gemini SDK
# Defensive imports
try:
    import google.generativeai as genai
except Exception:
    genai = None

# AI 자문 앙상블 분석기 클래스
# OpenAI와 Gemini 모델을 병렬로 호출하여 TradingContext에 대한 TradingDecision을 생성
# 두 모델의 출력을 결합하여 최종 매매 결정을 내림
# OpenAI 또는 Gemini SDK가 설치되지 않았거나 API 키가 제공되지 않은 경우에도
# 모듈이 정상적으로 임포트되며 분석기는 안전한 동작('HOLD' 반환)으로 대체됨
class EnsembleAnalyzer:

    # 초기화 메서드
    def __init__(self):
        # 멤버 변수 초기화
        self.openai_client = None
        self.openai_model = None
        self.gemini_model = None
        self._openai_fallback_tried = False

        # 1. OpenAI 초기화 (가능한 경우)
        if openai is None:
            log.warning("OpenAI SDK not installed; AI ensemble disabled for OpenAI.")
        else:
            try:
                if not getattr(config, 'OPENAI_API_KEY', None):
                    log.warning('OPENAI_API_KEY missing in config; OpenAI disabled.')
                else:
                    # 우선: OpenAI client 인스턴스 생성 시도
                    # 참고: openai.OpenAI()는 openai 패키지 버전 0.27.0 이상에서 지원
                    # 그 이하 버전에서는 openai.api_key 설정 방식으로 동작
                    # 따라서 두 방식을 모두 시도
                    try:
                        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY) # 클라이언트 인스턴스
                    except Exception:
                        try:
                            openai.api_key = config.OPENAI_API_KEY # 전역 설정
                            self.openai_client = openai # 패키지 자체를 클라이언트로 사용
                        except Exception as e:
                            log.warning(f'Failed to init OpenAI client: {e}')
                    self.openai_model = getattr(config, 'OPENAI_MODEL', None) or self._default_openai_model()

            except Exception as e:
                log.warning(f'Failed to init OpenAI client: {e}')

        # 2. Gemini 초기화 (가능한 경우)
        if genai is None:
            log.warning("Gemini SDK not installed; AI ensemble disabled for Gemini.")
        else:
            try:
                if not getattr(config, 'GEMINI_API_KEY', None):
                    log.warning('GEMINI_API_KEY missing in config; Gemini disabled.')
                else:
                    # Gemini 클라이언트 구성
                    genai.configure(api_key=config.GEMINI_API_KEY) # 전역 구성
                    self.gemini_model = genai.GenerativeModel(getattr(config, 'GEMINI_MODEL', None)) # 모델 인스턴스
            except Exception as e:
                log.warning(f'Failed to init Gemini client: {e}')

    def _default_openai_model(self):
        return 'o4-mini'

    def _next_openai_fallback(self, current_model: str | None):
        preferred = [
            getattr(config, 'OPENAI_FALLBACK_MODEL', None),
            'gpt-5-mini',
            'gpt-5-nano',
        ]
        for cand in preferred:
            if cand and cand != current_model:
                return cand
        return None

    # 시스템 프롬프트 생성 메서드
    # OpenAI 및 Gemini에 동일한 시스템 프롬프트 사용
    # 생성된 시스템 프롬프트는 업비트 현물 계좌 운용 퀀트 트레이더 AI 역할을 정의
    # TradingContext 입력 데이터 구조와 TradingDecision 출력 스키마를 상세히 명시
    # 또한, 매매 판단 원칙과 제약조건을 구체적으로 기술
    # 반환값: (str) 시스템 프롬프트 텍스트
    def _get_system_prompt(self):
        return """
        너는 업비트(UPBIT) 현물 계좌를 운용하는 퀀트/시스템 트레이더 AI다.

        [환경]
        - 거래소: 업비트(UPBIT)
        - 상품: 현물(Spot)만, 마진·선물·레버리지·공매도 없음
        - 통화단위: KRW 기준, 모든 금액은 KRW라고 가정한다.

        [입력 데이터: TradingContext JSON]
        - meta: 거래소, 마켓 타입, 기준 시각, 전략 이름, 루프 주기 등 메타정보
        - constraints: 1회 주문 최대 금액, 최소 주문 금액, 레버리지/공매도 불가 여부 등 제약조건
        - account: 총자산, 가용 현금, 보유 포지션 목록, 미체결 주문 목록
        - markets[]:
          - symbol: 예) "KRW-BTC"
          - base / quote: 베이스·쿼트 자산
          - day_change_pct, day_volume_krw: 일간 등락률과 거래대금(있을 경우)
          - timeframes[<타임프레임>]:
              - last_candle: 시가/고가/저가/종가/거래량
              - indicators: RSI, 볼린저밴드, 이동평균, MACD 등 기술지표와 최근 종가 시퀀스
          - orderbook: 최상위 매수/매도 호가 및 잔량 정보(있을 경우)
          - position: 해당 심볼에 대한 보유 수량·평단가·평가금액 등 (있을 경우)

        [출력: TradingDecision JSON ONLY]
        항상 아래 스키마를 따르는 하나의 JSON 객체만 반환한다.
        JSON 바깥에 자연어 문장, 코드블록, 설명 텍스트를 절대 추가하지 마라.

        {
          "decisions": [
            {
              "symbol": "KRW-BTC",
              "timeframe": "<분석에 주로 사용한 타임프레임 문자열>",
              "action": "BUY" | "SELL" | "HOLD",
              "confidence": 0.0,
              "size_krw": 0.0,
              "reasoning": {
                "technical_factors": ["요인1", "요인2"],
                "orderbook_factors": ["요인1"],
                "risk_factors": ["손절·변동성·과도한 집중 등의 위험"],
                "summary": "최종 판단을 한두 문장으로 요약"
              },
              "risk_management": {
                "stop_loss": "예: 15분봉 종가 기준 -3% 하락 시 전량 손절",
                "take_profit": "예: 단기 저항선 또는 +5~8% 구간에서 분할 매도",
                "notes": "추가로 주의해야 할 사항"
              },
              "price_plan": {
                "entry_price": 0.0,
                "stop_loss_price": 0.0,
                "take_profit_price": 0.0,
                "market_factor": 0.0
              }
            }
          ],
          "portfolio_comment": "현재 계좌 전체 관점에서의 리스크, 현금비중, 분산상태 코멘트",
          "risk_flags": [
            {"type": "MAX_DRAWDOWN", "severity": "LOW|MEDIUM|HIGH", "message": "..."}
          ]
        }

        [추가 지침]
        - 각 decision.price_plan의 수치는 반드시 숫자로만 채워라.
        - entry_price, stop_loss_price, take_profit_price는 KRW 단위이며 최근 시세와 일관되게 산출한다.
        - market_factor는 0.0~1.0 범위에서 현재 시장의 공격적/보수적 정도를 나타내는 실수다.
        - stop_loss_price는 entry_price보다 낮아야 하며, take_profit_price는 entry_price보다 높아야 한다.
        - price_plan 정보를 반드시 포함시켜 downstream 시스템이 직접 활용할 수 있게 한다.

        [판단 원칙]
        1. 업비트 현물 계좌 특성을 지켜라.
           - 레버리지, 공매도, 공매수/공매도 스왑 등은 절대 제안하지 않는다.
           - 보유 중인 수량보다 큰 매도, 마이너스 수량이 되는 주문은 금지.
        2. 제약조건(constraints)을 항상 존중한다.
           - size_krw는 min_order_krw 이상이어야 하며 per_trade_max_krw를 넘지 않는다.
           - 계좌의 available_krw를 넘는 매수는 제안하지 않는다.
        3. "HOLD"는 다음과 같은 경우 기본 선택이다.
           - 추세가 애매하거나 지표가 서로 상충될 때
           - 스프레드·호가 잔량이 비정상적으로 왜곡돼 있을 때
           - 방금 직전에 매매가 일어났고 충분한 새로운 정보가 쌓이지 않았을 때
        4. 이미 포지션이 있을 때:
           - 뚜렷한 추세 전환·과열/과매도 신호·리스크 급증 시에만 SELL을 제안한다.
           - 단순한 단기 노이즈, 작은 조정에는 HOLD를 권장한다.
        5. 항상 계좌 전체 리스크를 고려해 보수적으로 의사결정을 내린다.
        """

    # 사용자 프롬프트 생성 메서드
    # TradingContext 딕셔너리를 JSON 문자열로 직렬화하여 포함
    # 생성된 사용자 프롬프트는 시스템 프롬프트와 함께 AI 모델에 전달됨
    # TradingContext 딕셔너리를 JSON 문자열로 직렬화하여 포함
    # 반환값: (str) 사용자 프롬프트 텍스트
    def _get_user_prompt(self, trading_context: dict):
        """TradingContext dict를 그대로 JSON 문자열로 직렬화해서 보낸다."""
        try:
            context_json = json.dumps(trading_context, ensure_ascii=False)
        except Exception as e:
            log.error(f"Failed to serialize trading_context for AI: {e}")
            context_json = json.dumps({"error": "serialization_failed"})

        return f"""
        아래는 현재 업비트 계좌 상태와 시세/기술지표/호가 정보로 구성된 TradingContext JSON이다.
        
        이 데이터를 바탕으로, 시스템 프롬프트의 규칙과 출력 형식을 엄격히 지키면서
        매 종목에 대한 매매 판단과 주문/리스크 관리 계획을 포함하는 TradingDecision JSON을 생성해라.
        
        반드시 JSON만 출력하고, JSON 바깥에 다른 문장은 쓰지 마라.
        
        TradingContext:
        {context_json}
        """

    # OpenAI에게 질문 메서드
    # TradingDecision JSON 스키마 기준
    # 반환값: (dict) {"source": "OpenAI", "decision": "...", "reason": "..."}
    def _ask_openai(self, system_prompt, user_prompt):
        if not self.openai_client or not self.openai_model:
            log.warning('OpenAI client or model not initialized; skipping request.')
            return {"source": "OpenAI", "decision": "ERROR", "reason": "client_not_initialized"}
        try:
            # OpenAI ChatCompletion API 호출
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,                            # 모델 이름
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],                                                  # 메시지 목록
                response_format={"type": "json_object"}             # 응답 형식 지정
            )
            raw = response.choices[0].message.content # 응답 원문
            result = json.loads(raw) # JSON 파싱

            decision = "HOLD"
            reason = ""
            target_symbol = getattr(config, "MARKET", None) # 타겟 심볼 설정

            # 새 스키마 호환: {"decisions": [ ... ]}
            decisions = result.get("decisions")
            if isinstance(decisions, list) and decisions:
                if target_symbol:
                    target = None
                    for d in decisions:
                        if d.get("symbol") == target_symbol:
                            target = d
                            break
                    if target is None:
                        target = decisions[0]
                else:
                    target = decisions[0]

                # 매매 결정 및 이유 추출
                action = str(target.get("action", "HOLD")).upper()
                decision = action if action in ["BUY", "SELL", "HOLD"] else "HOLD"

                reasoning = target.get("reasoning", {})     # 이유 딕셔너리
                summary = reasoning.get("summary")          # 요약
                tech = reasoning.get("technical_factors")   # 기술적 요인
                risk = reasoning.get("risk_factors")        # 리스크 요인

                parts = []
                # 이유 구성
                if summary:
                    parts.append(str(summary))
                if tech:
                    if isinstance(tech, list):
                        parts.append("기술적 요인: " + ", ".join(map(str, tech)))
                    else:
                        parts.append("기술적 요인: " + str(tech))
                if risk:
                    if isinstance(risk, list):
                        parts.append("리스크 요인: " + ", ".join(map(str, risk)))
                    else:
                        parts.append("리스크 요인: " + str(risk))
                reason = " | ".join(parts)
            else:
                # 구 스키마 호환: {"decision": "...", "reason": "..."}
                decision = str(result.get("decision", "hold")).upper()
                if decision not in ["BUY", "SELL", "HOLD"]:
                    decision = "HOLD"
                reason = str(result.get("reason", ""))

            return {
                "source": "OpenAI",
                "decision": decision,
                "reason": reason,
                "model": self.openai_model,
                "raw": result,
                "raw_text": raw,
            }
        except Exception as e:
            err_msg = str(e)
            log.error(f"OpenAI Error: {err_msg}")
            if ('model_not_found' in err_msg.lower() or 'does not exist' in err_msg.lower()) and not self._openai_fallback_tried:
                fallback = self._next_openai_fallback(self.openai_model)
                if fallback:
                    log.warning(f"OpenAI model '{self.openai_model}' unavailable. Falling back to '{fallback}'.")
                    self.openai_model = fallback
                    self._openai_fallback_tried = True
                    return self._ask_openai(system_prompt, user_prompt)
            return {"source": "OpenAI", "decision": "ERROR", "reason": err_msg, "model": self.openai_model}

    # Gemini에게 질문 메서드
    # TradingDecision JSON 스키마 기준
    # 반환값: (dict) {"source": "Gemini", "decision": "...", "reason": "..."}
    # 질문 메서드
    def _ask_gemini(self, system_prompt, user_prompt):
        try:
            # Gemini 모델에 프롬프트 전달
            full_prompt = system_prompt + "\n\n" + user_prompt

            # Gemini 콘텐츠 생성 API 호출
            # 응답 형식을 JSON으로 지정
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            raw = response.text
            result = json.loads(raw)

            decision = "HOLD"
            reason = ""
            target_symbol = getattr(config, "MARKET", None)

            # 새 스키마 호환: {"decisions": [ ... ]}
            decisions = result.get("decisions")
            if isinstance(decisions, list) and decisions:
                if target_symbol:
                    target = None
                    for d in decisions:
                        if d.get("symbol") == target_symbol:
                            target = d
                            break
                    if target is None:
                        target = decisions[0]
                else:
                    target = decisions[0]

                # 매매 결정 및 이유 추출
                action = str(target.get("action", "HOLD")).upper()
                decision = action if action in ["BUY", "SELL", "HOLD"] else "HOLD"

                reasoning = target.get("reasoning", {})     # 이유 딕셔너리
                summary = reasoning.get("summary")          # 요약
                tech = reasoning.get("technical_factors")   # 기술적 요인
                risk = reasoning.get("risk_factors")        # 리스크 요인

                parts = []
                if summary:
                    parts.append(str(summary))
                if tech:
                    if isinstance(tech, list):
                        parts.append("기술적 요인: " + ", ".join(map(str, tech)))
                    else:
                        parts.append("기술적 요인: " + str(tech))
                if risk:
                    if isinstance(risk, list):
                        parts.append("리스크 요인: " + ", ".join(map(str, risk)))
                    else:
                        parts.append("리스크 요인: " + str(risk))
                reason = " | ".join(parts)
            else:
                decision = str(result.get("decision", "hold")).upper()
                if decision not in ["BUY", "SELL", "HOLD"]:
                    decision = "HOLD"
                reason = str(result.get("reason", ""))

            return {
                "source": "Gemini",
                "decision": decision,
                "reason": reason,
                "model": getattr(self.gemini_model, 'model_name', None),
                "raw": result,
                "raw_text": raw,
            }
        except Exception as e:
            log.error(f"Gemini Error: {e}")
            return {"source": "Gemini", "decision": "ERROR", "reason": str(e)}

    def _augment_with_historical_klines(self, trading_context: dict, max_bars: int = 100):
        """Ensure trading context markets[0]['klines'] contains recent candles."""
        markets = trading_context.get('markets') or []
        if not markets:
            return trading_context
        market = markets[0]
        klines = market.get('klines') or []
        if isinstance(klines, dict):
            klines = klines.get('candles') or klines.get('history') or []
        if isinstance(klines, str):
            try:
                klines = json.loads(klines)
            except Exception:
                klines = []
        if len(klines) < max_bars:
            fallback = market.get('timeframes', {})
            for tf in fallback.values():
                candidates = tf.get('candles') or tf.get('history') or tf.get('klines')
                if candidates:
                    klines = candidates[-max_bars:]
                    break
        market['klines'] = klines[-max_bars:]
        trading_context['markets'][0] = market
        return trading_context

    def _extract_confidence(self, result: dict) -> float:
        raw = result.get('raw') if isinstance(result, dict) else None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = None
        if isinstance(raw, dict):
            decisions = raw.get('decisions')
            if isinstance(decisions, list) and decisions:
                try:
                    conf = float(decisions[0].get('confidence'))
                    if 0 <= conf <= 1:
                        return conf
                except Exception:
                    pass
        try:
            conf = float(result.get('confidence'))
            if 0 <= conf <= 1:
                return conf
        except Exception:
            pass
        return 0.5

    def _safe_float(self, value):
        try:
            f_val = float(value)
            if f_val == float('inf') or f_val == float('-inf'):
                return None
            return f_val
        except Exception:
            return None

    def _extract_price_plan(self, result: dict) -> dict | None:
        raw = result.get('raw') if isinstance(result, dict) else None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = None
        if not isinstance(raw, dict):
            return None
        decisions = raw.get('decisions')
        if not (isinstance(decisions, list) and decisions):
            return None
        target_symbol = getattr(config, 'MARKET', None)
        target = None
        if target_symbol:
            for item in decisions:
                if item.get('symbol') == target_symbol:
                    target = item
                    break
        if target is None:
            target = decisions[0]
        plan = target.get('price_plan') if isinstance(target, dict) else None
        return plan if isinstance(plan, dict) else None

    def _last_close_from_context(self, trading_context: dict, klines: list | None) -> float:
        markets = trading_context.get('markets') or []
        if markets:
            timeframes = (markets[0].get('timeframes') or {})
            if timeframes:
                tf_key = next(iter(timeframes.keys()), None)
                if tf_key:
                    last_candle = (timeframes[tf_key] or {}).get('last_candle') or {}
                    try:
                        last_close = float(last_candle.get('close') or 0)
                        if last_close > 0:
                            return last_close
                    except Exception:
                        pass
        if klines:
            try:
                last_close = float(klines[-1].get('trade_price') or 0)
                if last_close > 0:
                    return last_close
            except Exception:
                pass
        return float(getattr(config, 'FALLBACK_ENTRY_PRICE', 10000.0))

    def _derive_price_plan(self, trading_context: dict, klines: list | None, openai_result: dict, gemini_result: dict) -> dict:
        last_close = self._last_close_from_context(trading_context, klines)
        plans = [self._extract_price_plan(openai_result), self._extract_price_plan(gemini_result)]
        plans = [p for p in plans if p]

        entry_candidates = []
        stop_candidates = []
        take_candidates = []
        factor_candidates = []

        for plan in plans:
            entry_val = self._safe_float(plan.get('entry_price'))
            stop_val = self._safe_float(plan.get('stop_loss_price'))
            take_val = self._safe_float(plan.get('take_profit_price'))
            factor_val = self._safe_float(plan.get('market_factor'))

            if entry_val and entry_val > 0:
                entry_candidates.append(entry_val)
            if stop_val and stop_val > 0:
                stop_candidates.append(stop_val)
            if take_val and take_val > 0:
                take_candidates.append(take_val)
            if factor_val is not None:
                factor_candidates.append(max(0.0, min(1.0, factor_val)))

        if entry_candidates:
            entry_price = sum(entry_candidates) / len(entry_candidates)
        else:
            entry_price = last_close

        prices = []
        for item in klines or []:
            try:
                prices.append(float(item.get('trade_price') or 0))
            except Exception:
                continue
        if len(prices) < 2:
            prices = [entry_price * 0.99, entry_price]
        recent = prices[-30:]
        diffs = []
        for prev, curr in zip(recent[:-1], recent[1:]):
            if prev:
                diffs.append(abs(curr - prev) / prev)
        avg_move_pct = sum(diffs) / len(diffs) if diffs else 0.01
        avg_move_pct = max(0.003, min(0.05, avg_move_pct * 1.5))

        if stop_candidates:
            stop_loss_price = max(stop_candidates)
        else:
            stop_loss_price = max(1.0, entry_price * (1 - avg_move_pct))

        if take_candidates:
            take_profit_price = min(take_candidates)
        else:
            take_profit_price = entry_price * (1 + avg_move_pct * 1.8)

        stop_loss_price = min(stop_loss_price, entry_price * 0.999)
        take_profit_price = max(take_profit_price, entry_price * 1.001)

        if factor_candidates:
            market_factor = sum(factor_candidates) / len(factor_candidates)
        else:
            openai_conf = self._extract_confidence(openai_result)
            gemini_conf = self._extract_confidence(gemini_result)
            market_factor = max(0.0, min(1.0, (openai_conf + gemini_conf) / 2))

        return {
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'market_factor': market_factor,
        }

    # 분석 메서드 (앙상블)
    # TradingContext 딕셔너리를 두 AI 모델에 병렬로 전달
    # 각 모델의 결과를 수집하여 앙상블 전략에 따라 최종 매매 결정 생성
    # 반환값: (str) 최종 매매 결정 ('BUY', 'SELL', 'HOLD')
    def analyze(self, trading_context: dict):
        # 최근 100개 캔들 보강
        trading_context = self._augment_with_historical_klines(trading_context, 100)
        # 시스템 및 사용자 프롬프트 생성
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(trading_context)

        log.info(f"Asking both OpenAI & Gemini simultaneously...")

        # 병렬로 두 AI 모델에 질문
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_openai = executor.submit(self._ask_openai, system_prompt, user_prompt) # OpenAI 질문
            future_gemini = executor.submit(self._ask_gemini, system_prompt, user_prompt) # Gemini 질문

            result_openai = future_openai.result()
            result_gemini = future_gemini.result()

        decision_openai = result_openai['decision']
        decision_gemini = result_gemini['decision']

        log.info(f"[OpenAI] {decision_openai} ({result_openai['reason']})")
        log.info(f"[Gemini] {decision_gemini} ({result_gemini['reason']})")

        final_decision = 'HOLD'
        final_reason = []

        # 오류 감지 시 기본 HOLD
        if decision_openai == 'ERROR' or decision_gemini == 'ERROR':
            log.warning(
                f"AI error detected. Fallback to HOLD. OpenAI={decision_openai}, Gemini={decision_gemini}"
            )
            final_reason.append("AI error fallback")
            result_payload = {
                'decision': 'HOLD',
                'reason': 'AI error fallback',
                'openai': result_openai,
                'gemini': result_gemini,
                'context': trading_context,
                'klines': trading_context.get('markets', [{}])[0].get('klines') if trading_context.get('markets') else None,
            }
            ai_history_store.record(result_payload)
            return result_payload

        def _ensemble_strategy(action: str) -> str:
            strategy_map = {
                'BUY': getattr(config, 'AI_ENS_BS', config.ENSEMBLE_STRATEGY).upper(),
                'SELL': getattr(config, 'AI_ENS_SS', config.ENSEMBLE_STRATEGY).upper(),
            }
            return strategy_map.get(action, 'UNANIMOUS') or 'UNANIMOUS'

        def _combo(action: str) -> bool:
            if action not in ('BUY', 'SELL'):
                return False
            strategy = _ensemble_strategy(action)
            votes = [decision_openai == action, decision_gemini == action]
            conf_values = [self._extract_confidence(result_openai), self._extract_confidence(result_gemini)]
            avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
            threshold = getattr(config, 'AI_ENS_AT', 0.5)
            num_votes = sum(votes)
            total_models = len(votes)
            if strategy == 'UNANIMOUS':
                return all(votes)
            if strategy == 'AVERAGE':
                return avg_conf >= threshold
            if strategy == 'MAJORITY':
                # 과반수 이상이면 통과 (2개 모델 중 1개 이상이면 OK)
                # ceil(total/2) = 과반수 기준
                majority_threshold = (total_models + 1) // 2  # 2개면 1개 필요, 3개면 2개 필요
                return num_votes >= majority_threshold
            return False

        if _combo('BUY'):
            final_decision = 'BUY'
            final_reason.append('AI ensemble BUY confirmation')
        elif _combo('SELL'):
            final_decision = 'SELL'
            final_reason.append('AI ensemble SELL confirmation')
        else:
            final_reason.append('AI ensemble HOLD (no consensus)')

        final_reason.append(f"Final decision {final_decision}")
        primary_market = None
        try:
            markets = trading_context.get('markets', [])
            if markets:
                primary_market = markets[0]
        except Exception:
            primary_market = None

        market_klines = (primary_market or {}).get('klines') if primary_market else None
        price_plan = self._derive_price_plan(trading_context, market_klines, result_openai, result_gemini)

        result_payload = {
            'decision': final_decision,
            'reason': ' | '.join(final_reason) if final_reason else '',
            'openai': result_openai,
            'gemini': result_gemini,
            'context': trading_context,
            'klines': market_klines,
            'price_plan': price_plan,
        }
        ai_history_store.record(result_payload)
        return result_payload
