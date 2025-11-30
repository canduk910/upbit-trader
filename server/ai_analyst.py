"""AI Ensemble analyzer (OpenAI + Gemini).

This module is defensive: if OpenAI or Gemini SDKs are not installed or API keys
are not provided, the module still imports successfully and the analyzer will
fall back to a safe behavior (returning 'HOLD').
"""

# -- Python imports
import json
import concurrent.futures
import server.config as config
from server.logger import log

# Try imports but don't fail module import if libraries are missing
try:
    import openai
except Exception:
    openai = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


class EnsembleAnalyzer:
    def __init__(self):
        # Initialize flags and clients defensively
        self.openai_client = None
        self.openai_model = None
        self.gemini_model = None

        # 1. OpenAI 초기화 (가능한 경우)
        if openai is None:
            log.warning("OpenAI SDK not installed; AI ensemble disabled for OpenAI.")
        else:
            try:
                if not getattr(config, 'OPENAI_API_KEY', None):
                    log.warning('OPENAI_API_KEY missing in config; OpenAI disabled.')
                else:
                    # Depending on openai package version, the client init may differ.
                    # Try common initialization patterns.
                    try:
                        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
                    except Exception:
                        # fallback: set api key on module
                        try:
                            openai.api_key = config.OPENAI_API_KEY
                            self.openai_client = openai
                        except Exception as e:
                            log.warning(f'Failed to init OpenAI client: {e}')
                    self.openai_model = getattr(config, 'OPENAI_MODEL', None)

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
                    genai.configure(api_key=config.GEMINI_API_KEY)
                    self.gemini_model = genai.GenerativeModel(getattr(config, 'GEMINI_MODEL', None))
            except Exception as e:
                log.warning(f'Failed to init Gemini client: {e}')

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
              }
            }
          ],
          "portfolio_comment": "현재 계좌 전체 관점에서의 리스크, 현금비중, 분산상태 코멘트",
          "risk_flags": [
            {"type": "MAX_DRAWDOWN", "severity": "LOW|MEDIUM|HIGH", "message": "..."}
          ]
        }

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

    def _ask_openai(self, system_prompt, user_prompt):
        """OpenAI에게 질문 (TradingDecision JSON 스키마 기준)"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)

            decision = "HOLD"
            reason = ""
            target_symbol = getattr(config, "MARKET", None)

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

                action = str(target.get("action", "HOLD")).upper()
                decision = action if action in ["BUY", "SELL", "HOLD"] else "HOLD"

                reasoning = target.get("reasoning", {})
                summary = reasoning.get("summary")
                tech = reasoning.get("technical_factors")
                risk = reasoning.get("risk_factors")

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
                # 구 스키마 호환: {"decision": "...", "reason": "..."}
                decision = str(result.get("decision", "hold")).upper()
                if decision not in ["BUY", "SELL", "HOLD"]:
                    decision = "HOLD"
                reason = str(result.get("reason", ""))

            return {"source": "OpenAI", "decision": decision, "reason": reason}
        except Exception as e:
            log.error(f"OpenAI Error: {e}")
            return {"source": "OpenAI", "decision": "ERROR", "reason": str(e)}


    def _ask_gemini(self, system_prompt, user_prompt):
        """Gemini에게 질문 (TradingDecision JSON 스키마 기준)"""
        try:
            full_prompt = system_prompt + "\n\n" + user_prompt

            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            raw = response.text
            result = json.loads(raw)

            decision = "HOLD"
            reason = ""
            target_symbol = getattr(config, "MARKET", None)

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

                action = str(target.get("action", "HOLD")).upper()
                decision = action if action in ["BUY", "SELL", "HOLD"] else "HOLD"

                reasoning = target.get("reasoning", {})
                summary = reasoning.get("summary")
                tech = reasoning.get("technical_factors")
                risk = reasoning.get("risk_factors")

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

            return {"source": "Gemini", "decision": decision, "reason": reason}
        except Exception as e:
            log.error(f"Gemini Error: {e}")
            return {"source": "Gemini", "decision": "ERROR", "reason": str(e)}


    def analyze(self, trading_context: dict):
        """
        두 AI에게 TradingContext를 전달하고 결과를 종합(Ensemble)
        """
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(trading_context)

        log.info(f"Asking both OpenAI & Gemini simultaneously...")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_openai = executor.submit(self._ask_openai, system_prompt, user_prompt)
            future_gemini = executor.submit(self._ask_gemini, system_prompt, user_prompt)

            result_openai = future_openai.result()
            result_gemini = future_gemini.result()

        decision_openai = result_openai['decision']
        decision_gemini = result_gemini['decision']

        log.info(f"[OpenAI] {decision_openai} ({result_openai['reason']})")
        log.info(f"[Gemini] {decision_gemini} ({result_gemini['reason']})")

        final_decision = 'HOLD'

        if decision_openai == 'ERROR' or decision_gemini == 'ERROR':
            log.warning(
                f"AI error detected. Fallback to HOLD. OpenAI={decision_openai}, Gemini={decision_gemini}"
            )
            return 'HOLD'

        if config.ENSEMBLE_STRATEGY == 'MAJORITY':
            if decision_openai == decision_gemini and decision_openai in ['BUY', 'SELL']:
                final_decision = decision_openai
                log.info(f"==> Consensus Reached: {final_decision}")
            else:
                log.info(
                    f"==> Disagreement (OpenAI:{decision_openai} vs Gemini:{decision_gemini}). Result: HOLD"
                )

        elif config.ENSEMBLE_STRATEGY == 'ANY':
            if 'BUY' in [decision_openai, decision_gemini]:
                final_decision = 'BUY'
            elif 'SELL' in [decision_openai, decision_gemini]:
                final_decision = 'SELL'

        return final_decision

