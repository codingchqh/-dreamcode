# dream_analyzer_service.py (수정된 내용)

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from typing import Dict, Any

# config.py 에서 API 키를 가져오는 것을 권장합니다. (실제 배포 시)
# from core.config import settings

class DreamAnalyzerService:
    def __init__(self, api_key: str): # api_key를 생성자에서 받도록 수정
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7) # 창의성을 위해 temperature 설정
        self.output_parser = StrOutputParser()

    def create_nightmare_prompt(self, dream_text: str) -> str:
        # 이 부분은 이전과 동일합니다.
        system_prompt = """
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
        """
        user_prompt_template = PromptTemplate.from_template(
            "User's nightmare description (Korean): {dream_text}"
        )
        chain = PromptTemplate.from_template(system_prompt + "\n" + user_prompt_template.template) | self.llm | self.output_parser
        return chain.invoke({"dream_text": dream_text})


    def create_reconstructed_prompt(self, dream_text: str, dream_report: Dict[str, Any]) -> str:
        """
        사용자의 악몽 텍스트와 감정 분석 리포트를 기반으로 긍정적으로 재구성된 이미지 프롬프트를 생성합니다.
        감정 분석 리포트의 키워드를 활용하여 재구성 방향을 구체화합니다.
        """
        # 감정 분석 리포트에서 키워드 추출
        keywords = dream_report.get("keywords", [])
        emotions = dream_report.get("emotions", []) # 감정 정보도 추가로 활용 가능
        
        # 키워드들을 프롬프트에 포함하기 위한 문자열로 변환
        keywords_str = ", ".join(keywords) if keywords else "No specific keywords provided."
        
        # 감정 정보를 더 상세하게 프롬프트에 추가 (선택 사항)
        emotion_summary = []
        for emo in emotions:
            emotion_summary.append(f"{emo.get('emotion')}: {int(emo.get('score', 0))}%")
        emotion_summary_str = "; ".join(emotion_summary) if emotion_summary else "No specific emotions detected."


        system_prompt = f"""
        You are a wise and empathetic dream therapist. Your goal is to reframe the user's nightmare into an image of peace, healing, and hope.
        
        CRITICAL RULE: If the dream contains sensitive real-world roles like 'soldier', replace them with neutral terms like 'a young person' or 'a figure'.

        Utilize the provided analysis data to guide the transformation:
        - Identified Keywords from the original nightmare: [{keywords_str}]
        - Emotion Breakdown: [{emotion_summary_str}]
        
        Based on these, transform any negative elements associated with the keywords or emotions into their positive, safe, and metaphorical counterparts.
        
        CONTEXT-AWARE KOREAN AESTHETIC: Reinterpret the scene within a positive, modern Korean context relevant to the original dream. Avoid stereotypes.
        
        The final output must be a single-paragraph, English image prompt that is safe and positive. It must NOT contain any text or writing.
        """

        user_prompt_template = PromptTemplate.from_template(
            "User's nightmare description (Korean): {dream_text}"
        )
        
        chain = PromptTemplate.from_template(system_prompt + "\n" + user_prompt_template.template) | self.llm | self.output_parser
        return chain.invoke({"dream_text": dream_text})