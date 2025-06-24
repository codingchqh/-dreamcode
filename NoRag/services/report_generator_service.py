# services/report_generator_service.py (디버깅용 임시 코드)

from core.config import API_KEY
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
# [디버깅] 원래의 파서로 되돌리고, 오류 유형을 명시적으로 임포트합니다.
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from typing import List

llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY, temperature=0.2)

class Emotion(BaseModel):
    emotion: str = Field(description="감정의 이름 (예: 불안, 무력감, 희망)")
    score: int = Field(description="해당 감정의 강도를 0에서 100 사이의 점수로 표현")

class EmotionReport(BaseModel):
    emotions: List[Emotion] = Field(description="꿈에서 발견된 주요 감정들의 목록 (점수가 높은 순으로 3-4개)")
    keywords: List[str] = Field(description="이러한 감정들을 나타내는 꿈 속의 핵심 단어나 구절들")
    analysis_summary: str = Field(description="꿈의 전반적인 심리적 경향에 대한 한 문장 요약 (예: 우울/회피 경향의 키워드가 다수 발견됩니다.)")

report_parser = PydanticOutputParser(pydantic_object=EmotionReport)

report_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert dream analyst with a deep understanding of psychology. "
            "Your task is to analyze the user's dream description and create a structured emotion report. "
            "Identify the dominant emotions, assign an intensity score, extract relevant keywords, and provide a summary of the psychological tendency. "
            "All parts of the report should be in Korean. "
            "Strictly follow the provided JSON format instructions.\n\n{format_instructions}",
        ),
        ("human", "다음은 제가 꾼 꿈의 내용입니다. 분석해주세요.\n\n---\n\n{dream_text}"),
    ]
)

# [디버깅] 자동 재시도 기능을 끈 원래의 체인으로 변경
report_chain = report_generation_prompt | llm | report_parser

def generate_report(dream_text: str) -> EmotionReport:
    """
    꿈 텍스트를 분석하여 구조화된 감정 리포트를 생성합니다.
    """
    try:
        print("[DEBUG] 감정 분석 리포트 생성 시작...")
        report = report_chain.invoke({
            "dream_text": dream_text,
            "format_instructions": report_parser.get_format_instructions(),
        })
        print("[DEBUG] 감정 분석 리포트 생성 완료.")
        return report
    # [디버깅] 오류 발생 시, LLM의 원본 답변을 출력하는 로직 추가
    except OutputParserException as e:
        # 파싱에 실패한 LLM의 원본 답변을 가져옵니다.
        llm_output = e.llm_output
        print("--- [DEBUG] LLM의 원본 답변 (파싱 실패 원인) ---")
        print(llm_output)
        print("-------------------------------------------------")
        return EmotionReport(emotions=[], keywords=[], analysis_summary="리포트 생성에 실패했습니다. (파싱 오류)")
    except Exception as e:
        print(f"리포트 생성 중 알 수 없는 오류 발생: {e}")
        return EmotionReport(emotions=[], keywords=[], analysis_summary="리포트 생성에 실패했습니다. (알 수 없는 오류)")