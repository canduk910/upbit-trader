import json
import concurrent.futures
import server.config as config
from server.logger import log

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
                    # Gemini 클라이언트 구성
                    genai.configure(api_key=config.GEMINI_API_KEY) # 전역 구성
                    self.gemini_model = genai.GenerativeModel(getattr(config, 'GEMINI_MODEL', None)) # 모델 인스턴스
            except Exception as e:
                log.warning(f'Failed to init Gemini client: {e}')

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

            return {"source": "OpenAI", "decision": decision, "reason": reason}
        except Exception as e:
            log.error(f"OpenAI Error: {e}")
            return {"source": "OpenAI", "decision": "ERROR", "reason": str(e)}

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

            return {"source": "Gemini", "decision": decision, "reason": reason}
        except Exception as e:
            log.error(f"Gemini Error: {e}")
            return {"source": "Gemini", "decision": "ERROR", "reason": str(e)}

    # 분석 메서드 (앙상블)
    # TradingContext 딕셔너리를 두 AI 모델에 병렬로 전달
    # 각 모델의 결과를 수집하여 앙상블 전략에 따라 최종 매매 결정 생성
    # 반환값: (str) 최종 매매 결정 ('BUY', 'SELL', 'HOLD')
    def analyze(self, trading_context: dict):
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

        # 오류 감지 시 기본 HOLD
        if decision_openai == 'ERROR' or decision_gemini == 'ERROR':
            log.warning(
                f"AI error detected. Fallback to HOLD. OpenAI={decision_openai}, Gemini={decision_gemini}"
            )
            return 'HOLD'

        # 앙상블 전략 적용
        # 1. 다수결 원칙 전략
        # 양쪽 모델이 동일한 BUY 또는 SELL 신호를 낸 경우에만 해당 신호 채택
        if config.ENSEMBLE_STRATEGY == 'MAJORITY':
            if decision_openai == decision_gemini and decision_openai in ['BUY', 'SELL']:
                final_decision = decision_openai
                log.info(f"==> Consensus Reached: {final_decision}")
            else:
                log.info(
                    f"==> Disagreement (OpenAI:{decision_openai} vs Gemini:{decision_gemini}). Result: HOLD"
                )
        # 2. ANY 전략
        # BUY 또는 SELL 신호가 하나라도 있으면 해당 신호 채택
        elif config.ENSEMBLE_STRATEGY == 'ANY':
            if 'BUY' in [decision_openai, decision_gemini]:
                final_decision = 'BUY'
            elif 'SELL' in [decision_openai, decision_gemini]:
                final_decision = 'SELL'

        return final_decision

