# services/report_generator_service.py (수정 전)
# from core.config import API_KEY # <-- 이 부분이 오류의 원인입니다.

# services/report_generator_service.py (수정 후)

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
# config.py에서 settings 객체를 임포트합니다.
from core.config import settings # 이 줄을 추가/수정합니다.

class ReportGeneratorService:
    def __init__(self, api_key: str):
        # 이제 api_key는 생성자를 통해 외부(main.py)에서 주입받습니다.
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.3)

    def generate_report(self, dream_text: str) -> dict:
        # 감정 분석 및 키워드 추출을 위한 시스템 프롬프트
        system_prompt = """
        You are an AI dream analyst. Analyze the user's dream text to identify core emotions and key elements.
        Provide:
        1. A list of dominant emotions with a score (0-1, 0 being low, 1 being high)
        2. A list of key keywords (nouns, verbs, adjectives relevant to the dream's core)
        3. A brief (2-3 sentences) overall analysis summary of the dream's emotional tone and potential themes.
        Respond in JSON format.
        Example JSON:
        {
          "emotions": [
            {"emotion": "fear", "score": 0.8},
            {"emotion": "anxiety", "score": 0.6}
          ],
          "keywords": ["dark forest", "chasing", "lost"],
          "analysis_summary": "The dream depicts a strong sense of fear and anxiety, with themes of being pursued and disoriented in a threatening environment."
        }
        """
        user_prompt_template = PromptTemplate.from_template(
            "User's dream description (Korean): {dream_text}"
        )
        chain = PromptTemplate.from_template(system_prompt + "\n" + user_prompt_template.template) | self.llm | StrOutputParser()

        try:
            raw_response = chain.invoke({"dream_text": dream_text})
            # LLM이 간혹 JSON 앞뒤에 불필요한 문자를 추가할 수 있으므로, JSON 부분만 추출 시도
            if raw_response.strip().startswith("```json") and raw_response.strip().endswith("```"):
                json_str = raw_response.strip()[7:-3].strip()
            else:
                json_str = raw_response.strip()

            report = json.loads(json_str)
            return report
        except Exception as e:
            print(f"Error generating report or parsing JSON: {e}\nRaw response: {raw_response}")
            return {
                "emotions": [{"emotion": "error", "score": 1.0}],
                "keywords": ["error"],
                "analysis_summary": f"리포트 생성 중 오류가 발생했습니다: {e}"
            }