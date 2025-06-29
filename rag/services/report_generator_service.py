# services/report_generator_service.py

import json
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

# --- Pydantic 모델 정의: LLM의 출력 형식을 강제하고 안정적으로 파싱하기 위함 ---
class Emotion(BaseModel):
    emotion: str = Field(description="감정의 명칭 (한국어)")
    score: float = Field(description="감정의 강도 (0.0에서 1.0 사이)")

class Report(BaseModel):
    emotions: List[Emotion] = Field(description="주요 감정 목록")
    keywords: List[str] = Field(description="꿈의 핵심 키워드 목록 (한국어)")
    analysis_summary: str = Field(description="전문 지식을 바탕으로 한 심층 분석 요약 (2-4 문장, 한국어)")


class ReportGeneratorService:
    """
    [RAG 통합 버전] 꿈 텍스트와 전문 지식을 함께 분석하여
    감정, 키워드, 심층 분석 요약을 포함하는 리포트를 생성하는 클래스입니다.
    """
    def __init__(self, api_key: str, retriever: Any):
        """
        ReportGeneratorService를 초기화합니다.
        :param api_key: OpenAI API 키
        :param retriever: 미리 학습된 FAISS retriever 객체
        """
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.3)
        self.retriever = retriever
        # Pydantic 모델을 기반으로 출력 파서를 생성합니다.
        self.parser = PydanticOutputParser(pydantic_object=Report)

    def _format_docs(self, docs: List[Dict]) -> str:
        """검색된 문서들을 하나의 문자열로 결합하는 내부 함수"""
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_report_with_rag(self, dream_text: str) -> dict:
        """
        주어진 꿈 텍스트에 대해 RAG를 활용한 심층 분석 리포트를 생성합니다.
        :param dream_text: 분석할 꿈의 텍스트
        :return: 감정, 키워드, 심층 분석 요약을 포함하는 딕셔너리
        """
        # RAG 기능을 포함한 새로운 프롬프트 템플릿
        rag_prompt_template = """
        You are an AI dream analyst who is an expert in IRT and dream symbolism.
        Your task is to analyze the user's dream by referring to the provided [Professional Knowledge].
        
        Based on BOTH the [User's Dream Text] and the [Professional Knowledge], generate a structured report.
        The 'analysis_summary' MUST be based on insights from the [Professional Knowledge].
        
        All parts of the report (emotions, keywords, summary) MUST be in Korean.

        {format_instructions}

        [Professional Knowledge]
        {context}

        [User's Dream Text]
        {dream_text}
        """

        prompt = ChatPromptTemplate.from_template(
            rag_prompt_template,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        # LCEL을 사용한 RAG 체인 구성
        chain = (
            {
                "context": self.retriever | self._format_docs, # retriever로 context 검색 및 포맷팅
                "dream_text": RunnablePassthrough() # 사용자 입력을 그대로 전달
            }
            | prompt
            | self.llm
            | self.parser
        )

        try:
            # RAG 체인을 실행하여 리포트(Pydantic 객체)를 생성
            report_object = chain.invoke(dream_text)
            # Pydantic 객체를 딕셔너리로 변환하여 반환
            return report_object.dict()
        except Exception as e:
            print(f"Error generating report with RAG: {e}")
            # RAG 실패 시, 기존 방식(LLM 단독)으로 보고서 생성 시도 (선택적)
            # 또는 간단한 오류 리포트 반환
            return {
                "emotions": [{"emotion": "오류", "score": 1.0}],
                "keywords": ["RAG_리포트_생성_오류"],
                "analysis_summary": f"RAG 리포트 생성 중 오류가 발생했습니다: {e}"
            }