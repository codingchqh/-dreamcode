# dream_analyzer_service.py (수정된 내용)

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from typing import Dict, Any, Tuple, List # Tuple, List 임포트 추가
import json # JSON 파싱을 위해 추가

class DreamAnalyzerService:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7)
        self.output_parser = StrOutputParser()

    def create_nightmare_prompt(self, dream_text: str) -> str:
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


    def create_reconstructed_prompt(self, dream_text: str, dream_report: Dict[str, Any]) -> Tuple[str, str, List[Tuple[str, str]]]:
        """
        사용자의 악몽 텍스트와 감정 분석 리포트를 기반으로 긍정적으로 재구성된 이미지 프롬프트를 생성합니다.
        추가로 변환 요약 및 키워드 매핑 정보도 반환합니다.
        """
        keywords = dream_report.get("keywords", [])
        emotions = dream_report.get("emotions", [])
        
        keywords_str = ", ".join(keywords) if keywords else "No specific keywords provided."
        
        emotion_summary = []
        for emo in emotions:
            emotion_summary.append(f"{emo.get('emotion')}: {int(emo.get('score', 0))}%")
        emotion_summary_str = "; ".join(emotion_summary) if emotion_summary else "No specific emotions detected."

        # 1. 재구성 이미지 프롬프트 생성 (이전과 동일하게 키워드/감정 정보 포함)
        reconstruction_system_prompt = f"""
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
        reconstruction_chain = PromptTemplate.from_template(reconstruction_system_prompt + "\n" + user_prompt_template.template) | self.llm | self.output_parser
        reconstructed_prompt = reconstruction_chain.invoke({"dream_text": dream_text})

        # 2. 변환 요약 텍스트 생성
        summary_system_prompt = f"""
        You are an AI assistant specialized in summarizing dream transformations.
        Given the original nightmare text, the identified keywords, and the reconstructed image prompt,
        summarize in 2-3 concise sentences in Korean how the negative elements or keywords from the original nightmare
        were transformed into positive, healing, or hopeful themes in the reconstructed image prompt.
        Focus on the *change* and *positive reinterpretation*.
        
        Original Nightmare Keywords: [{keywords_str}]
        Reconstructed Image Prompt: "{reconstructed_prompt}"
        """
        summary_user_prompt_template = PromptTemplate.from_template(
            "Original nightmare description (Korean): {dream_text}"
        )
        summary_chain = PromptTemplate.from_template(summary_system_prompt + "\n" + summary_user_prompt_template.template) | self.llm | self.output_parser
        transformation_summary = summary_chain.invoke({"dream_text": dream_text})

        # 3. 키워드 매핑 생성 (강조를 위해, 원본 키워드가 프롬프트에서 어떻게 재해석되었는지 LLM에 질의)
        # 이 부분은 LLM의 환각 위험이 있어 간단하게 처리하거나, 수동 매핑 고려
        # 여기서는 LLM에게 프롬프트 내에서 재구성된 주요 개념을 추출하게 지시합니다.
        mapping_system_prompt = f"""
        Given the original nightmare keywords and the reconstructed image prompt, identify 3-5 key concepts from the original nightmare that were most significantly reinterpreted or transformed into positive elements in the reconstructed prompt.
        For each, provide the original concept (from the keywords) and its positively reinterpreted counterpart found in the reconstructed prompt.
        Respond in JSON format: [{{ "original": "original_concept", "transformed": "transformed_concept" }}, ...]
        
        Original Nightmare Keywords: [{keywords_str}]
        Reconstructed Image Prompt: "{reconstructed_prompt}"
        """
        mapping_chain = PromptTemplate.from_template(mapping_system_prompt) | self.llm | self.output_parser
        
        try:
            mapping_raw = mapping_chain.invoke({"dream_text": dream_text}) # dream_text는 사실 여기서는 큰 의미 없음
            keyword_mappings = json.loads(mapping_raw)
            # LLM이 잘못된 형식의 JSON을 줄 수도 있으므로 추가 검증
            if not isinstance(keyword_mappings, list) or not all(isinstance(item, dict) and "original" in item and "transformed" in item for item in keyword_mappings):
                keyword_mappings = [] # 유효하지 않으면 빈 리스트로 설정
        except json.JSONDecodeError:
            print(f"경고: 키워드 매핑 생성 중 JSON 파싱 오류: {mapping_raw}")
            keyword_mappings = [] # 오류 발생 시 빈 리스트로 설정
        except Exception as e:
            print(f"경고: 키워드 매핑 생성 중 예상치 못한 오류: {e}")
            keyword_mappings = []


        return reconstructed_prompt, transformation_summary, keyword_mappings