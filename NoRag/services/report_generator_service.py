# services/report_generator_service.py

from core.config import API_KEY
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import List

# LangChain 모델 초기화
llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY, temperature=0.2)

# --- 1. LLM으로부터 받을 데이터의 구조를 정의합니다 ---
class Emotion(BaseModel):
    """꿈에 나타난 단일 감정에 대한 정보"""
    emotion: str = Field(description="감정의 이름 (예: 불안, 무력감, 희망)")
    score: int = Field(description="해당 감정의 강도를 0에서 100 사이의 점수로 표현")

class EmotionReport(BaseModel):
    """꿈의 내용에 대한 전체 감정 분석 리포트"""
    emotions: List[Emotion] = Field(description="꿈에서 발견된 주요 감정들의 목록 (점수가 높은 순으로 3-4개)")
    keywords: List[str] = Field(description="이러한 감정들을 나타내는 꿈 속의 핵심 단어나 구절들")
    analysis_summary: str = Field(description="꿈의 전반적인 심리적 경향에 대한 한 문장 요약 (예: 우울/회피 경향의 키워드가 다수 발견됩니다.)")

# --- 2. 이 구조에 맞춰 출력하도록 파서를 설정합니다 ---
report_parser = PydanticOutputParser(pydantic_object=EmotionReport)

# --- 3. LLM에게 역할을 부여하는 프롬프트 템플릿을 만듭니다 ---
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

# --- 4. 프롬프트, LLM, 파서를 하나의 체인으로 연결합니다 ---
report_chain = report_generation_prompt | llm | report_parser

def generate_report(dream_text: str) -> EmotionReport:
    """
    꿈 텍스트를 분석하여 구조화된 감정 리포트를 생성합니다.
    """
    try:
        return report_chain.invoke({
            "dream_text": dream_text,
            "format_instructions": report_parser.get_format_instructions(),
        })
    except Exception as e:
        print(f"리포트 생성 중 오류 발생: {e}")
        # 오류 발생 시, 빈 리포트 객체를 반환하거나 오류 처리를 할 수 있습니다.
        return EmotionReport(emotions=[], keywords=[], analysis_summary="리포트를 생성하는 데 실패했습니다.")