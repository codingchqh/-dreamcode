# services/report_generator_service.py (자동 재시도 기능 추가)

from core.config import API_KEY
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
# [수정됨] 자동 재시도 파서를 추가로 임포트합니다.
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from typing import List

# LangChain 모델 초기화
llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY, temperature=0.2)

# --- 1. LLM으로부터 받을 데이터의 구조를 정의합니다 ---
class Emotion(BaseModel):
    emotion: str = Field(description="감정의 이름 (예: 불안, 무력감, 희망)")
    score: int = Field(description="해당 감정의 강도를 0에서 100 사이의 점수로 표현")

class EmotionReport(BaseModel):
    emotions: List[Emotion] = Field(description="꿈에서 발견된 주요 감정들의 목록 (점수가 높은 순으로 3-4개)")
    keywords: List[str] = Field(description="이러한 감정들을 나타내는 꿈 속의 핵심 단어나 구절들")
    analysis_summary: str = Field(description="꿈의 전반적인 심리적 경향에 대한 한 문장 요약 (예: 우울/회피 경향의 키워드가 다수 발견됩니다.)")

# --- 2. 파서를 설정합니다 ---
# 2-1. 기본 Pydantic 파서 생성
report_parser = PydanticOutputParser(pydantic_object=EmotionReport)

# [수정됨] 2-2. 자동 재시도 파서로 기본 파서를 감싸줍니다.
retry_report_parser = RetryWithErrorOutputParser.from_llm(
    parser=report_parser,
    llm=llm # 오류 수정 요청 시 사용할 LLM
)

# --- 3. 프롬프트 템플릿을 만듭니다 ---
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

# --- 4. [수정됨] 최종 체인에 '자동 재시도 파서'를 연결합니다 ---
report_chain = report_generation_prompt | llm | retry_report_parser

def generate_report(dream_text: str) -> EmotionReport:
    """
    꿈 텍스트를 분석하여 구조화된 감정 리포트를 생성합니다.
    (오류 자동 재시도 기능 포함)
    """
    try:
        print("[DEBUG] 감정 분석 리포트 생성 시작...")
        # .invoke() 호출 시, Retry 파서를 사용하므로 format_instructions를 직접 전달해야 합니다.
        report = report_chain.invoke({
            "dream_text": dream_text,
            "format_instructions": report_parser.get_format_instructions(),
        })
        print("[DEBUG] 감정 분석 리포트 생성 완료.")
        return report
    except Exception as e:
        print(f"리포트 생성 중 최종 오류 발생: {e}")
        return EmotionReport(emotions=[], keywords=[], analysis_summary="리포트를 생성하는 데 실패했습니다. 잠시 후 다시 시도해주세요.")