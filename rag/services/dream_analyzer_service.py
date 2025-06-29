# services/dream_analyzer_service.py (악몽 프롬프트 강화 버전)

import os
import json
from typing import Dict, Any, Tuple, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser

# Pydantic 모델 정의 (이전과 동일)
class KeywordMapping(BaseModel):
    original: str = Field(description="악몽에 있었던 원래의 부정적 개념 (한국어)")
    transformed: str = Field(description="재구성되어 긍정적으로 변환된 개념 (한국어)")
class ReconstructionOutput(BaseModel):
    reconstructed_prompt: str = Field(description="DALL-E 3를 위한, 긍정적으로 재구성된 최종 이미지 프롬프트 (영어, 한 문단)")
    transformation_summary: str = Field(description="변환 과정에 대한 2-3 문장의 요약 (한국어)")
    keyword_mappings: List[KeywordMapping] = Field(description="원본-변환 키워드 매핑 리스트 (3-5개)")


class DreamAnalyzerService:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7)
        self.json_parser = PydanticOutputParser(pydantic_object=ReconstructionOutput)
        self.output_parser = StrOutputParser() # StrOutputParser도 유지

    # ===> 여기가 핵심 변경 사항입니다! <===
    def create_nightmare_prompt(self, dream_text: str, dream_report: Dict[str, Any]) -> str:
        """
        악몽 텍스트와 핵심 키워드를 기반으로,
        꿈의 공포스러운 분위기를 극대화하는 DALL-E 3용 프롬프트를 생성합니다.
        """
        keywords = dream_report.get("keywords", [])
        keywords_info = ", ".join(keywords) if keywords else "No specific keywords provided."

        # 훨씬 더 강력하고 구체적인 지시를 담은 새 시스템 프롬프트
        system_prompt = f"""
        You are a prompt artist specializing in psychological horror and dark surrealism for DALL-E 3. Your task is to translate the user's Korean nightmare into a terrifying, atmospheric, and visually striking image prompt in English.

        **Core Mission:**
        Your prompt MUST visualize the feelings of paranoia, loss of control, and digital dystopia embedded in the user's dream. The identified keywords [{keywords_info}] are the central theme.

        **Artistic & Thematic Directions:**
        - **Theme:** The main theme is the horror of a simulated reality controlled by a cold, omnipotent AI.
        - **Visuals:** Emphasize digital corruption. Use terms like "glitching," "pixelating," "data streams," "fractured reality," "unstable simulation."
        - **Atmosphere:** Create a sense of dread and helplessness. Use phrases like "cold blue and sterile white light," "oppressive silence," "cold metallic textures," "a figure made of crumbling code."
        - **Safety:** While creating a terrifying image, you must adhere to safety policies. NEVER depict literal self-harm, gore, or violence. Represent fear and pain metaphorically.
        
        The final output must be a single, detailed paragraph in English.
        """
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "User's nightmare description (Korean): {dream_text}")
        ])
        
        chain = prompt_template | self.llm | self.output_parser
        return chain.invoke({"dream_text": dream_text})
        
    def create_reconstructed_prompt_and_analysis(self, dream_text: str, dream_report: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, str]]]:
        # (이 함수는 변경할 필요 없습니다. 그대로 두시면 됩니다.)
        keywords = dream_report.get("keywords", [])
        emotions = dream_report.get("emotions", [])
        keywords_info = ", ".join(keywords) if keywords else "제공된 특정 키워드 없음."
        emotion_summary_list = [f"{emo.get('emotion')}: {int(emo.get('score', 0)*100)}%" for emo in emotions]
        emotions_info = "; ".join(emotion_summary_list) if emotion_summary_list else "감지된 특정 감정 없음."

        system_prompt = """
        You are a wise and empathetic dream therapist AI. Your goal is to perform three tasks at once. The most important task is to transform the negative 'Identified Keywords' into positive visual symbols.
        **CRITICAL INSTRUCTION:** The keywords [{keywords_info}] are the most important elements. You MUST reframe these specific keywords into symbols of peace, healing, and hope to create an English image prompt.
        **Analysis Data:** - Original Nightmare Text (Korean): {dream_text}, - Identified Keywords: {keywords_info}, - Emotion Breakdown: {emotions_info}
        **Your Three Tasks:** 1. Generate Reconstructed Prompt. 2. Generate Transformation Summary in Korean. 3. Generate Keyword Mappings.
        **Output Format Instruction:** You MUST provide your response in the following JSON format.
        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_template(
            template=system_prompt,
            partial_variables={"format_instructions": self.json_parser.get_format_instructions()}
        )
        chain = prompt | self.llm | self.json_parser
        response: ReconstructionOutput = chain.invoke({
            "dream_text": dream_text, "keywords_info": keywords_info, "emotions_info": emotions_info
        })
        keyword_mappings_dict = [mapping.dict() for mapping in response.keyword_mappings]
        return response.reconstructed_prompt, response.transformation_summary, keyword_mappings_dict