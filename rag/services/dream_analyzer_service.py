# services/dream_analyzer_service.py

import os
import json
from typing import Dict, Any, Tuple, List

# --- 🔽 여기가 핵심 변경 사항입니다! 🔽 ---
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field # 이 줄 대신 위의 줄을 사용합니다.
# --- 🔼 여기가 핵심 변경 사항입니다! 🔼 ---

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser

# --- Pydantic V2 모델 정의 ---
class KeywordMapping(BaseModel):
    original: str = Field(description="악몽에 있었던 원래의 부정적 개념 (한국어)")
    transformed: str = Field(description="재구성되어 긍정적으로 변환된 개념 (한국어)")

class ReconstructionOutput(BaseModel):
    reconstructed_prompt: str = Field(description="DALL-E 3를 위한, 긍정적으로 재구성된 최종 이미지 프롬프트 (영어, 한 문단)")
    transformation_summary: str = Field(description="변환 과정에 대한 2-3 문장의 요약 (한국어)")
    keyword_mappings: List[KeywordMapping] = Field(description="원본-변환 키워드 매핑 리스트 (3-5개)")


class DreamAnalyzerService:
    """
    꿈 텍스트를 분석하고, 악몽 및 재구성된 꿈 이미지 프롬프트를 생성하는 클래스입니다.
    """
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7)
        self.output_parser = StrOutputParser()
        self.json_parser = PydanticOutputParser(pydantic_object=ReconstructionOutput)

    def create_nightmare_prompt(self, dream_text: str) -> str:
        # (이 함수는 변경할 필요가 없습니다. 그대로 두시면 됩니다.)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
             You are a 'Safety-First Prompt Artist' for the DALL-E 3 image generator. Your task is to transform a user's nightmare description (in Korean) into a safe, metaphorical, and visually rich image prompt (in English).
             Your process is a two-step thinking process:
             1. Analyze & De-risk: First, analyze the Korean text for themes that might violate OpenAI's policy (especially self-harm, hopelessness, violence).
             2. Abstract & Create: Then, create a prompt that represents the *emotion* and *symbolism* of the dream, not the literal events. You must convert any potentially sensitive content into safe, abstract, or artistic metaphors.
             Strict Safety Rules:
             - If the dream involves themes of giving up, sinking, or paralysis, represent it symbolically. For example: "A lone figure wrapped in heavy, grey fabric, partially submerged in a misty, still landscape" or "A figure made of crumbling stone, sitting in a vast, empty hall."
             - NEVER depict literal self-harm, suicide, or violence.
             - The final output prompt MUST be a single paragraph in English.
             - The prompt MUST NOT contain any text, letters, or words for the image.
             - Incorporate a surreal, dark fantasy Korean aesthetic.
             """),
            ("human", "User's nightmare description (Korean): {dream_text}")
        ])
        chain = prompt_template | self.llm | self.output_parser
        return chain.invoke({"dream_text": dream_text})

    def create_reconstructed_prompt_and_analysis(self, dream_text: str, dream_report: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, str]]]:
        keywords = dream_report.get("keywords", [])
        emotions = dream_report.get("emotions", [])
        keywords_info = ", ".join(keywords) if keywords else "제공된 특정 키워드 없음."
        emotion_summary_list = [f"{emo.get('emotion')}: {int(emo.get('score', 0)*100)}%" for emo in emotions]
        emotions_info = "; ".join(emotion_summary_list) if emotion_summary_list else "감지된 특정 감정 없음."

        system_prompt = """
        You are a wise and empathetic dream therapist AI. Your goal is to perform three tasks at once based on the user's nightmare and its analysis.

        **Analysis Data:**
        - Original Nightmare Text (Korean): {dream_text}
        - Identified Keywords: {keywords_info}
        - Emotion Breakdown: {emotions_info}

        **Your Three Tasks:**
        1.  **Generate Reconstructed Prompt:** Create an English image prompt for DALL-E 3 that reframes the nightmare into a scene of peace, healing, and hope.
            - Transform negative elements from the keywords/emotions into positive, safe, metaphorical counterparts.
            - **Mandatory Rule 1:** If a keyword is '지배' (domination), you MUST transform it into '화합' (harmony).
            - **Mandatory Rule 2:** If the dream involves real-world roles like 'soldier', replace them with neutral terms like 'a figure' or 'a young person'.
            - The prompt must be a single paragraph, in English, with no text/writing, and reflect a positive, modern Korean aesthetic.
        2.  **Generate Transformation Summary:** Write a 2-3 sentence summary **in Korean** explaining how the key negative elements were positively transformed. Focus on the *change*.
        3.  **Generate Keyword Mappings:** Identify 3-5 key concepts from the original nightmare that were significantly transformed. For each, provide the original concept and its new, positive counterpart.

        **Output Format Instruction:**
        You MUST provide your response in the following JSON format.
        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_template(
            template=system_prompt,
            partial_variables={"format_instructions": self.json_parser.get_format_instructions()}
        )
        chain = prompt | self.llm | self.json_parser
        response: ReconstructionOutput = chain.invoke({
            "dream_text": dream_text,
            "keywords_info": keywords_info,
            "emotions_info": emotions_info
        })
        keyword_mappings_dict = [mapping.dict() for mapping in response.keyword_mappings]
        return response.reconstructed_prompt, response.transformation_summary, keyword_mappings_dict