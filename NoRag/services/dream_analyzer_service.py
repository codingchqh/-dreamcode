# services/dream_analyzer_service.py

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from typing import Dict, Any, Tuple, List # Tuple, List 임포트 추가
import json # JSON 파싱을 위해 추가

class DreamAnalyzerService:
    """
    꿈 텍스트를 분석하고, 악몽 및 재구성된 꿈 이미지 프롬프트를 생성하는 클래스입니다.
    LangChain과 OpenAI LLM을 사용합니다.
    """
    def __init__(self, api_key: str):
        """
        DreamAnalyzerService를 초기화합니다.
        :param api_key: OpenAI API 키
        """
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7) # 창의성을 위해 temperature 설정
        self.output_parser = StrOutputParser()

    def create_nightmare_prompt(self, dream_text: str) -> str:
        """
        악몽 텍스트를 기반으로 DALL-E 3용 악몽 이미지 프롬프트를 생성합니다.
        :param dream_text: 사용자의 악몽 텍스트 (한국어)
        :return: 생성된 이미지 프롬프트 (영어)
        """
        # 시스템 프롬프트는 LLM에 대한 지시를 포함합니다.
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
        # 사용자 프롬프트는 dream_text 변수를 받습니다.
        user_prompt_template = PromptTemplate.from_template(
            "User's nightmare description (Korean): {dream_text}"
        )
        # 시스템 프롬프트와 사용자 프롬프트 템플릿을 연결하여 체인을 생성합니다.
        chain = PromptTemplate.from_template(system_prompt + "\n" + user_prompt_template.template) | self.llm | self.output_parser
        # 체인을 호출하여 프롬프트를 생성하고 결과를 반환합니다.
        return chain.invoke({"dream_text": dream_text})


    def create_reconstructed_prompt(self, dream_text: str, dream_report: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, str]]]:
        """
        사용자의 악몽 텍스트와 감정 분석 리포트를 기반으로 긍정적으로 재구성된 이미지 프롬프트,
        변환 요약 텍스트, 그리고 키워드 매핑 정보를 함께 생성합니다.
        
        :param dream_text: 원본 악몽 텍스트 (한국어)
        :param dream_report: 감정 분석 리포트 (딕셔너리)
        :return: (재구성된 프롬프트, 변환 요약 텍스트, 키워드 매핑 리스트)
        """
        keywords = dream_report.get("keywords", [])
        emotions = dream_report.get("emotions", [])
        
        # keywords_info와 emotions_info를 PromptTemplate에 전달할 변수로 만듭니다.
        keywords_info = ", ".join(keywords) if keywords else "No specific keywords provided."
        
        emotion_summary_list = []
        for emo in emotions:
            emotion_summary_list.append(f"{emo.get('emotion')}: {int(emo.get('score', 0))}%")
        emotions_info = "; ".join(emotion_summary_list) if emotion_summary_list else "No specific emotions detected."

        # 1. 재구성 이미지 프롬프트 생성
        # 이제 {keywords_info}와 {emotions_info}는 LangChain의 PromptTemplate 변수입니다.
        reconstruction_system_prompt = """
        You are a wise and empathetic dream therapist. Your goal is to reframe the user's nightmare into an image of peace, healing, and hope.
        
        CRITICAL RULE: If the dream contains sensitive real-world roles like 'soldier', replace them with neutral terms like 'a young person' or 'a figure'.

        Utilize the provided analysis data to guide the transformation:
        - Identified Keywords from the original nightmare: {keywords_info}
        - Emotion Breakdown: {emotions_info}
        
        Based on these, transform any negative elements associated with the keywords or emotions into their positive, safe, and metaphorical counterparts.
        
        CONTEXT-AWARE KOREAN AESTHETIC: Reinterpret the scene within a positive, modern Korean context relevant to the original dream. Avoid stereotypes.
        
        The final output must be a single-paragraph, English image prompt that is safe and positive. It must NOT contain any text or writing.
        """
        # Note: The `system_prompt` itself is now the template for LangChain
        # 우리는 system_prompt와 user_prompt를 하나의 PromptTemplate 문자열로 결합합니다.
        combined_template_for_reconstruction = reconstruction_system_prompt + "\nUser's nightmare description (Korean): {dream_text}"

        reconstruction_chain = PromptTemplate.from_template(combined_template_for_reconstruction) | self.llm | self.output_parser
        
        # invoke 호출 시 모든 필요한 변수를 전달합니다.
        reconstructed_prompt = reconstruction_chain.invoke({
            "dream_text": dream_text,
            "keywords_info": keywords_info, # 새로운 변수 추가
            "emotions_info": emotions_info  # 새로운 변수 추가
        })

        # 2. 변환 요약 텍스트 생성
        # 이 요약 프롬프트도 새로 정의된 변수들을 사용합니다.
        summary_system_prompt = """
        You are an AI assistant specialized in summarizing dream transformations.
        Given the original nightmare text, the identified keywords, and the reconstructed image prompt,
        summarize in 2-3 concise sentences in Korean how the negative elements or keywords from the original nightmare
        were transformed into positive, healing, or hopeful themes in the reconstructed image prompt.
        Focus on the *change* and *positive reinterpretation*.
        
        Original Nightmare Keywords: {keywords_info}
        Reconstructed Image Prompt: "{reconstructed_prompt}"
        """
        summary_combined_template = summary_system_prompt + "\nOriginal nightmare description (Korean): {dream_text}"
        summary_chain = PromptTemplate.from_template(summary_combined_template) | self.llm | self.output_parser
        
        transformation_summary = summary_chain.invoke({
            "dream_text": dream_text,
            "keywords_info": keywords_info,
            "reconstructed_prompt": reconstructed_prompt # reconstructed_prompt를 변수로 전달
        })

        # 3. 키워드 매핑 생성
        # JSON 예시 내의 중괄호는 PromptTemplate에 의해 변수로 오해되지 않도록 {{ }}로 이스케이프합니다.
        mapping_system_prompt = """
        Given the original nightmare keywords and the reconstructed image prompt, identify 3-5 key concepts from the original nightmare that were most significantly reinterpreted or transformed into positive elements in the reconstructed prompt.
        For each, provide the original concept (from the keywords) and its positively reinterpreted counterpart found in the reconstructed prompt.
        Respond strictly in JSON format. Example: [{{ "original": "original_concept", "transformed": "transformed_concept" }}, ...]
        
        Original Nightmare Keywords: {keywords_info}
        Reconstructed Image Prompt: "{reconstructed_prompt}"
        """
        mapping_chain = PromptTemplate.from_template(mapping_system_prompt) | self.llm | self.output_parser
        
        try:
            # invoke 호출 시 필요한 변수를 전달합니다.
            mapping_raw = mapping_chain.invoke({
                "keywords_info": keywords_info,
                "reconstructed_prompt": reconstructed_prompt
            })
            
            # JSON 파싱 로직 (이전과 동일)
            if raw_response := mapping_raw.strip():
                if raw_response.startswith("```json") and raw_response.endswith("```"):
                    json_str = raw_response[7:-3].strip()
                else:
                    json_str = raw_response
            else:
                json_str = ""

            if json_str:
                keyword_mappings = json.loads(json_str)
                if not isinstance(keyword_mappings, list) or not all(isinstance(item, dict) and "original" in item and "transformed" in item for item in keyword_mappings):
                    print(f"경고: 키워드 매핑 JSON 형식이 유효하지 않습니다. 예상치 못한 형식: {json_str}")
                    keyword_mappings = []
            else:
                keyword_mappings = []

        except json.JSONDecodeError as e:
            print(f"경고: 키워드 매핑 생성 중 JSON 파싱 오류: {e}\n원본 응답: {mapping_raw}")
            keyword_mappings = []
        except Exception as e:
            print(f"경고: 키워드 매핑 생성 중 예상치 못한 오류: {e}")
            keyword_mappings = []

        return reconstructed_prompt, transformation_summary, keyword_mappings