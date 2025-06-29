# services/report_generator_service.py

import json
from typing import List, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

# Pydantic 모델 정의: LLM의 출력 형식을 강제하고 안정적으로 파싱하기 위함
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
    def __init__(self, api_key: str, retriever: Any = None):
        """
        ReportGeneratorService를 초기화합니다.
        :param api_key: OpenAI API 키
        :param retriever: (선택 사항) 미리 학습된 FAISS retriever 객체
        """
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.3)
        self.retriever = retriever
        self.parser = PydanticOutputParser(pydantic_object=Report)

    def _format_docs(self, docs: List[Any]) -> str:
        """검색된 문서들을 하나의 문자열로 결합하는 내부 함수"""
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_report_with_rag(self, dream_text: str) -> dict:
        """
        주어진 꿈 텍스트에 대해 RAG를 활용한 심층 분석 리포트를 생성합니다.
        :param dream_text: 분석할 꿈의 텍스트
        :return: 감정, 키워드, 심층 분석 요약을 포함하는 딕셔너리
        """
        if not self.retriever:
            raise ValueError("RAG 리포트를 생성하려면 retriever 객체가 필요합니다.")

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
        chain = (
            {"context": self.retriever | self._format_docs, "dream_text": RunnablePassthrough()}
            | prompt
            | self.llm
            | self.parser
        )
        try:
            report_object = chain.invoke(dream_text)
            return report_object.dict()
        except Exception as e:
            print(f"Error generating report with RAG: {e}")
            return {"emotions": [], "keywords": [], "analysis_summary": f"RAG 리포트 생성 중 오류가 발생했습니다: {e}"}

    def generate_report(self, dream_text: str) -> dict:
        """ (기존 함수) RAG 없이 LLM만으로 리포트를 생성합니다. """
        # (사용자님의 기존 코드를 여기에 그대로 유지하거나, 아래의 Pydantic 기반 코드로 대체할 수 있습니다.)
        # 지금은 RAG 버전을 사용할 것이므로 이 함수는 호출되지 않습니다.
        # 이 함수는 만약을 위한 백업으로 남겨둡니다.
        # ... (사용자님의 기존 generate_report 코드) ...
        # 간단한 오류 리포트를 반환하도록 수정
        return {"emotions": [], "keywords": [], "analysis_summary": "RAG 없는 기본 분석은 현재 비활성화되어 있습니다."}