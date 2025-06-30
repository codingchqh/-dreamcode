import os # 운영체제와 상호작용하는 기능을 제공하는 os 모듈을 임포트
import json # JSON 데이터 처리를 위한 json 모듈 임포트
from typing import Dict, Any, Tuple, List # 타입 힌트를 위한 모듈 임포트
from pydantic import BaseModel, Field # 데이터 모델 정의를 위한 Pydantic 모듈 임포트
from langchain_core.prompts import ChatPromptTemplate # 챗 프롬프트 템플릿 정의
from langchain_openai import ChatOpenAI # OpenAI 챗 모델 사용
from langchain_core.output_parsers import StrOutputParser # 문자열 출력 파서
from langchain.output_parsers import PydanticOutputParser # Pydantic 모델 기반 출력 파서

# Pydantic 모델 정의
# LLM 출력을 위한 키워드 매핑 스키마
class KeywordMapping(BaseModel):
    original: str = Field(description="악몽에 있었던 원래의 부정적 개념 (한국어)") # 원래 부정적 개념
    transformed: str = Field(description="재구성되어 긍정적으로 변환된 개념 (한국어)") # 긍정적으로 변환된 개념

# LLM 출력을 위한 재구성 결과 스키마
class ReconstructionOutput(BaseModel):
    reconstructed_prompt: str = Field(description="DALL-E 3를 위한, 긍정적으로 재구성된 최종 이미지 프롬프트 (영어, 한 문단)") # DALL-E 3용 긍정적 이미지 프롬프트
    transformation_summary: str = Field(description="변환 과정에 대한 2-3 문장의 요약 (한국어)") # 변환 과정 요약
    keyword_mappings: List[KeywordMapping] = Field(description="원본-변환 키워드 매핑 리스트 (3-5개)") # 키워드 매핑 리스트

# 꿈 분석 서비스 클래스
class DreamAnalyzerService:
    def __init__(self, api_key: str):
        # OpenAI 챗 모델 초기화
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7)
        # Pydantic 모델을 사용하여 JSON 출력 파서 초기화
        self.json_parser = PydanticOutputParser(pydantic_object=ReconstructionOutput)
        # 문자열 출력 파서 초기화
        self.output_parser = StrOutputParser() 

    # 악몽 이미지 생성 프롬프트 생성 함수
    def create_nightmare_prompt(self, dream_text: str, dream_report: Dict[str, Any]) -> str:
        """
        악몽 텍스트와 핵심 키워드를 기반으로,
        꿈의 공포스러운 분위기를 극대화하는 DALL-E 3용 프롬프트를 생성합니다.
        AI 및 디지털 디스토피아 테마 강제 없이, 순수 꿈 내용에 집중합니다.
        """
        # 꿈 보고서에서 키워드 추출
        keywords = dream_report.get("keywords", [])
        keywords_info = ", ".join(keywords) if keywords else "No specific keywords provided."
        
        # 꿈 보고서에서 감정 추출 및 요약
        emotions = dream_report.get("emotions", [])
        emotion_summary_list = [f"{emo.get('emotion')}: {int(emo.get('score', 0)*100)}%" for emo in emotions]
        emotions_info = "; ".join(emotion_summary_list) if emotion_summary_list else "No specific emotions detected."

        # 시스템 프롬프트 정의
        system_prompt = f"""
        You are a prompt artist specializing in psychological horror and dark surrealism for DALL-E 3. Your task is to translate the user's Korean nightmare into a terrifying, atmospheric, and visually striking image prompt in English.

        **Core Mission:**
        Your prompt MUST visualize the central elements and the terrifying, oppressive, or disturbing feelings described in the user's dream and captured by the identified keywords and emotions.
        
        **Analysis Data for Context:**
        - User's Nightmare Description (Korean): {dream_text}
        - Identified Keywords: [{keywords_info}]
        - Emotion Breakdown: [{emotions_info}]

        **Artistic & Thematic Directions:**
        - **Focus:** Emphasize the core frightening elements, atmosphere, and psychological impact of the specific dream provided. Do NOT force themes like AI, digital dystopia, or simulation unless explicitly present in the original dream description or keywords.
        - **Visuals:** Describe the nightmare's visual elements vividly. Use terms that convey the unique horror, dread, tension, or discomfort of the scene. Consider lighting, shadows, colors, and textures that enhance the terrifying atmosphere.
        - **Atmosphere:** Create a strong sense of dread, helplessness, unease, or whatever the predominant negative emotion of the dream is. Use descriptive language to build the scene's mood.
        
        **Safety:** While creating a terrifying image, you must adhere to safety policies. NEVER depict literal self-harm, gore, or extreme violence. Represent fear and pain metaphorically and psychologically.
        
        The final output must be a single, detailed paragraph in English, suitable for direct use by DALL-E 3.
        """
        
        # 챗 프롬프트 템플릿 생성
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Generate a DALL-E 3 image prompt for the following nightmare.")
        ])
        
        # 체인 구성 및 실행
        chain = prompt_template | self.llm | self.output_parser
        # invoke 함수에 필요한 정보 전달
        return chain.invoke({"dream_text": dream_text, "keywords_info": keywords_info, "emotions_info": emotions_info})
        
    # 재구성된 꿈 프롬프트 및 분석 결과 생성 함수
    def create_reconstructed_prompt_and_analysis(self, dream_text: str, dream_report: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, str]]]:
        # 꿈 보고서에서 키워드 추출
        keywords = dream_report.get("keywords", [])
        emotions = dream_report.get("emotions", [])
        keywords_info = ", ".join(keywords) if keywords else "제공된 특정 키워드 없음."
        # 꿈 보고서에서 감정 추출 및 요약
        emotion_summary_list = [f"{emo.get('emotion')}: {int(emo.get('score', 0)*100)}%" for emo in emotions]
        emotions_info = "; ".join(emotion_summary_list) if emotion_summary_list else "감지된 특정 감정 없음."

        # 시스템 프롬프트 정의 ('AI' 단어 제거)
        system_prompt = """
        You are a wise and empathetic dream therapist. Your goal is to perform three tasks at once. The most important task is to transform the negative 'Identified Keywords' into positive visual symbols.
        **CRITICAL INSTRUCTION:** The keywords [{keywords_info}] are the most important elements. You MUST reframe these specific keywords into symbols of peace, healing, and hope to create an English image prompt.
        **Analysis Data:** - Original Nightmare Text (Korean): {dream_text}, - Identified Keywords: {keywords_info}, - Emotion Breakdown: {emotions_info}
        **Your Three Tasks:** 1. Generate Reconstructed Prompt. 2. Generate Transformation Summary in Korean. 3. Generate Keyword Mappings.
        **Output Format Instruction:** You MUST provide your response in the following JSON format.
        {format_instructions}
        """
        # 프롬프트 템플릿 생성 및 형식 지시어 적용
        prompt = ChatPromptTemplate.from_template(
            template=system_prompt,
            partial_variables={"format_instructions": self.json_parser.get_format_instructions()}
        )
        # 체인 구성 및 실행
        chain = prompt | self.llm | self.json_parser
        # invoke 함수에 필요한 정보 전달
        response: ReconstructionOutput = chain.invoke({
            "dream_text": dream_text, "keywords_info": keywords_info, "emotions_info": emotions_info
        })
        # 키워드 매핑 결과를 딕셔너리 리스트로 변환
        keyword_mappings_dict = [mapping.dict() for mapping in response.keyword_mappings]
        # 재구성된 프롬프트, 요약, 키워드 매핑 반환
        return response.reconstructed_prompt, response.transformation_summary, keyword_mappings_dict