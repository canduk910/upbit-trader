import openai
import google.generativeai as genai
import json
import concurrent.futures
import config
from logger import log


class EnsembleAnalyzer:
    def __init__(self):
        # 1. OpenAI 초기화
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.openai_model = config.OPENAI_MODEL

        # 2. Gemini 초기화
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)

    def _get_system_prompt(self):
        return """
        You are an expert crypto technical analyst. 
        Analyze the provided chart data (OHLCV + Indicators).

        Response Format (JSON ONLY):
        {"decision": "buy" or "sell" or "hold", "reason": "brief reason"}
        """

    def _get_user_prompt(self, df, current_status):
        recent_data = df.tail(15).to_json()
        return f"""
        Market: {config.MARKET}
        My Position: {current_status} (in_position=holding coin, no_position=holding cash)

        Data (JSON): {recent_data}

        Task:
        - Analyze Trends, RSI, Bollinger Bands.
        - If 'no_position' and strong uptrend signal: "buy"
        - If 'in_position' and trend reversal/downtrend: "sell"
        - Uncertain? "hold"
        """

    def _ask_openai(self, system_prompt, user_prompt):
        """OpenAI에게 질문"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return {"source": "OpenAI", "decision": result.get("decision", "hold").upper(),
                    "reason": result.get("reason", "")}
        except Exception as e:
            log.error(f"OpenAI Error: {e}")
            return {"source": "OpenAI", "decision": "ERROR", "reason": str(e)}

    def _ask_gemini(self, system_prompt, user_prompt):
        """Gemini에게 질문"""
        try:
            # Gemini는 시스템 프롬프트 설정 방식이 다르지만, 여기선 user prompt에 합쳐서 보냅니다.
            full_prompt = system_prompt + "\n\n" + user_prompt

            # JSON 모드 강제 (generation_config 사용)
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            return {"source": "Gemini", "decision": result.get("decision", "hold").upper(),
                    "reason": result.get("reason", "")}
        except Exception as e:
            log.error(f"Gemini Error: {e}")
            return {"source": "Gemini", "decision": "ERROR", "reason": str(e)}

    def analyze(self, df, current_status):
        """
        두 AI에게 동시에 질문하고 결과를 종합(Ensemble)
        """
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(df, current_status)

        log.info(f"Asking both OpenAI & Gemini simultaneously...")

        # 병렬 처리로 속도 향상
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_openai = executor.submit(self._ask_openai, system_prompt, user_prompt)
            future_gemini = executor.submit(self._ask_gemini, system_prompt, user_prompt)

            result_openai = future_openai.result()
            result_gemini = future_gemini.result()

        decision_openai = result_openai['decision']
        decision_gemini = result_gemini['decision']

        log.info(f"[OpenAI] {decision_openai} ({result_openai['reason']})")
        log.info(f"[Gemini] {decision_gemini} ({result_gemini['reason']})")

        # --- 투표 로직 (Ensemble Logic) ---
        final_decision = 'HOLD'

        # 에러가 하나라도 있으면 안전하게 HOLD (또는 에러 무시 로직 추가 가능)
        if decision_openai == 'ERROR' or decision_gemini == 'ERROR':
            log.warning("One of the AIs failed. Safety fallback: HOLD")
            return 'HOLD'

        if config.ENSEMBLE_STRATEGY == 'UNANIMOUS':
            # 만장일치: 둘 다 같은 의견이어야 함
            if decision_openai == decision_gemini and decision_openai in ['BUY', 'SELL']:
                final_decision = decision_openai
                log.info(f"==> Consensus Reached: {final_decision}")
            else:
                log.info(f"==> Disagreement (OpenAI:{decision_openai} vs Gemini:{decision_gemini}). Result: HOLD")

        elif config.ENSEMBLE_STRATEGY == 'ANY':
            # 공격적: 둘 중 하나라도 신호를 주면 진입
            if 'BUY' in [decision_openai, decision_gemini]:
                final_decision = 'BUY'
            elif 'SELL' in [decision_openai, decision_gemini]:
                final_decision = 'SELL'

        return final_decision