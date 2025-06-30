import json # JSON 데이터 처리를 위한 json 모듈 임포트
from typing import List, Any # 타입 힌트를 위한 List, Any 임포트
from pydantic import BaseModel, Field # Pydantic을 이용한 데이터 모델 정의
from langchain_core.prompts import ChatPromptTemplate # 챗 프롬프트 템플릿 정의
from langchain_core.runnables import RunnablePassthrough # 입력값을 그대로 통과시키는 Runnable
from langchain_openai import ChatOpenAI # OpenAI 챗 모델 사용
from langchain.output_parsers import PydanticOutputParser # Pydantic 모델 기반 출력 파서

# Pydantic 모델 정의
# 감정 정보를 담는 모델
class Emotion(BaseModel):
    emotion: str = Field(description="감정의 명칭 (한국어)") # 감정 명칭
    score: float = Field(description="감정의 강도 (0.0에서 1.0 사이)") # 감정 강도

# 분석 리포트 전체 구조를 담는 모델
class Report(BaseModel):
    emotions: List[Emotion] = Field(description="주요 감정 목록") # 주요 감정 목록
    keywords: List[str] = Field(description="꿈의 핵심 키워드 목록 (한국어)") # 핵심 키워드 목록
    analysis_summary: str = Field(description="전문 지식을 바탕으로 한 심층 분석 요약 (2-4 문장, 한국어)") # 심층 분석 요약

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
        # OpenAI 챗 모델 초기화
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.3)
        # 검색기(retriever) 설정 (RAG 사용 시 필요)
        self.retriever = retriever
        # PydanticOutputParser를 사용하여 리포트 모델에 맞게 출력 파싱
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
        # retriever가 없으면 RAG 리포트 생성이 불가하므로 에러 발생
        if not self.retriever:
            raise ValueError("RAG 리포트를 생성하려면 retriever 객체가 필요합니다.")

        # RAG를 위한 프롬프트 템플릿 정의
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
        # 프롬프트 템플릿 생성 및 형식 지시어 적용
        prompt = ChatPromptTemplate.from_template(
            rag_prompt_template,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        # LangChain Expression Language (LCEL) 체인 구성
        chain = (
            {"context": self.retriever | self._format_docs, "dream_text": RunnablePassthrough()} # context는 retriever로 문서 검색 후 포맷, dream_text는 그대로 전달
            | prompt # 프롬프트 적용
            | self.llm # LLM 호출
            | self.parser # 파서로 출력 형식 변환
        )
        try:
            # 체인 실행 및 리포트 객체 반환
            report_object = chain.invoke(dream_text)
            return report_object.dict() # 리포트 객체를 딕셔너리로 변환하여 반환
        except Exception as e:
            # 오류 발생 시 에러 메시지 출력 및 빈 리포트 반환
            print(f"Error generating report with RAG: {e}")
            return {"emotions": [], "keywords": [], "analysis_summary": f"RAG 리포트 생성 중 오류가 발생했습니다: {e}"}

    def generate_report(self, dream_text: str) -> dict:
        """ (기존 함수) RAG 없이 LLM만으로 리포트를 생성합니다. """
        # 현재 RAG 버전을 사용하므로 이 함수는 비활성화됨
        return {"emotions": [], "keywords": [], "analysis_summary": "RAG 없는 기본 분석은 현재 비활성화되어 있습니다."}