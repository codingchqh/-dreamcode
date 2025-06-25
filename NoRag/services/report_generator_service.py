# services/report_generator_service.py

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json # JSON 파싱을 위해 추가

class ReportGeneratorService:
    """
    꿈 텍스트를 분석하여 감정, 키워드, 요약을 포함하는 리포트를 생성하는 클래스입니다.
    LangChain과 OpenAI LLM을 사용합니다.
    """
    def __init__(self, api_key: str):
        """
        ReportGeneratorService를 초기화합니다.
        :param api_key: OpenAI API 키
        """
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.3)

    def generate_report(self, dream_text: str) -> dict:
        """
        주어진 꿈 텍스트에 대한 감정 분석 리포트를 JSON 형식으로 생성합니다.
        :param dream_text: 분석할 꿈의 텍스트
        :return: 감정, 키워드, 분석 요약을 포함하는 딕셔너리
        """
        # 시스템 프롬프트: LLM에게 리포트 생성 지시 및 JSON 형식 예시 제공
        # 중요: 예시 JSON 내의 모든 중괄호는 {{ }}로 이스케이프해야 합니다!
        # ===> 여기에 'Output in Korean.' 지시와 한국어 예시를 추가합니다. <===
        system_prompt = """
        You are an AI dream analyst. Analyze the user's dream text to identify core emotions and key elements.
        Provide:
        1. A list of dominant emotions with a score (0-1, 0 being low, 1 being high). Output emotion names in Korean.
        2. A list of key keywords (nouns, verbs, adjectives relevant to the dream's core). Output keywords in Korean.
        3. A brief (2-3 sentences) overall analysis summary of the dream's emotional tone and potential themes. Output analysis summary in Korean.
        Respond strictly in JSON format. Do not include any other text or markdown outside the JSON block.
        
        Example JSON (Output in Korean):
        {{
          "emotions": [
            {{"emotion": "두려움", "score": 0.8}},
            {{"emotion": "불안", "score": 0.6}},
            {{"emotion": "무기력", "score": 0.9}}
          ],
          "keywords": ["어두운 숲", "추격", "길 잃음", "압도감"],
          "analysis_summary": "꿈은 압도적인 환경 속에서 추격당하고 길을 잃는 것과 같은 강한 두려움과 불안감을 나타냅니다. 이는 무기력함과 좌절감을 반영하며, 현실에서의 어려움을 상징할 수 있습니다."
        }}
        """
        # 사용자 프롬프트 템플릿
        user_prompt_template = PromptTemplate.from_template(
            "User's dream description (Korean): {dream_text}"
        )
        # 시스템 프롬프트와 사용자 프롬프트를 결합하여 최종 PromptTemplate 생성
        chain = PromptTemplate.from_template(system_prompt + "\n" + user_prompt_template.template) | self.llm | StrOutputParser()

        try:
            raw_response = chain.invoke({"dream_text": dream_text}) # dream_text 변수 전달
            
            # LLM 응답에서 JSON 부분만 안전하게 추출
            if raw_response.strip().startswith("```json") and raw_response.strip().endswith("```"):
                json_str = raw_response.strip()[7:-3].strip()
            else:
                json_str = raw_response.strip()

            # JSON 문자열을 파이썬 딕셔너리로 로드
            report = json.loads(json_str)
            return report
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM response: {e}\nRaw response: {raw_response}")
            return {
                "emotions": [{"emotion": "파싱_오류", "score": 1.0}], # 오류 시에도 한글로
                "keywords": ["JSON_오류"], # 오류 시에도 한글로
                "analysis_summary": f"리포트 JSON 형식 오류: {e}. 원본 응답: {raw_response[:min(len(raw_response), 200)]}..."
            }
        except Exception as e:
            print(f"Error generating report: {e}")
            return {
                "emotions": [{"emotion": "처리_오류", "score": 1.0}], # 오류 시에도 한글로
                "keywords": ["리포트_오류"], # 오류 시에도 한글로
                "analysis_summary": f"리포트 생성 중 알 수 없는 오류가 발생했습니다: {e}"
            }