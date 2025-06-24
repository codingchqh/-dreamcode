from core.config import API_KEY
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
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
retry_report_parser = RetryWithErrorOutputParser.from_llm(parser=report_parser, llm=llm)

report_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert dream analyst... (이하 프롬프트 내용은 이전과 동일)"
        ),
        ("human", "다음은 제가 꾼 꿈의 내용입니다. 분석해주세요.\n\n---\n\n{dream_text}"),
    ]
)
report_chain = report_generation_prompt | llm | retry_report_parser

def generate_report(dream_text: str) -> EmotionReport:
    try:
        print("[DEBUG] 감정 분석 리포트 생성 시작...")
        report = report_chain.invoke({
            "dream_text": dream_text,
            "format_instructions": report_parser.get_format_instructions(),
        })
        print("[DEBUG] 감정 분석 리포트 생성 완료.")
        return report
    except Exception as e:
        print(f"리포트 생성 중 최종 오류 발생: {e}")
        return EmotionReport(emotions=[], keywords=[], analysis_summary="리포트를 생성하는 데 실패했습니다. 잠시 후 다시 시도해주세요.")