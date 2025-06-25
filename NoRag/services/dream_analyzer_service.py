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
        # reconstruction_chain에 전달되는 template은 하나의 문자열이어야 합니다.
        # f-string으로 system_prompt를 만들고, user_prompt_template.template를 뒤에 붙입니다.
        chain = PromptTemplate.from_template(system_prompt + "\n" + user_prompt_template.template) | self.llm | self.output_parser
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
        
        keywords_str = ", ".join(keywords) if keywords else "No specific keywords provided."
        
        emotion_summary = []
        for emo in emotions:
            # f-string으로 감정 요약 문자열을 만듭니다.
            emotion_summary.append(f"{emo.get('emotion')}: {int(emo.get('score', 0))}%")
        emotion_summary_str = "; ".join(emotion_summary) if emotion_summary else "No specific emotions detected."

        # 1. 재구성 이미지 프롬프트 생성
        # 중요: PromptTemplate에 전달될 때, 변수로 해석되지 않아야 할 중괄호는 {{ }}로 이스케이프해야 합니다.
        # 여기서 {keywords_str}과 {emotion_summary_str}은 이미 이 f-string에 의해 값이 채워진 문자열입니다.
        reconstruction_system_prompt = f"""
        You are a wise and empathetic dream therapist. Your goal is to reframe the user's nightmare into an image of peace, healing, and hope.
        
        CRITICAL RULE: If the dream contains sensitive real-world roles like 'soldier', replace them with neutral terms like 'a young person' or 'a figure'.

        Utilize the provided analysis data to guide the transformation:
        - Identified Keywords from the original nightmare: [{{keywords_str}}]
        - Emotion Breakdown: [{{emotion_summary_str}}]
        
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

        # 3. 키워드 매핑 생성
        # LLM에게 JSON 응답을 강제하기 위한 response_schema 사용이 더 좋지만,
        # StrOutputParser를 사용하므로 여기서는 텍스트로 요청하고 파싱 시도.
        # JSON 예시 내의 중괄호도 PromptTemplate에 의해 변수로 오해되지 않도록 {{ }}로 이스케이프합니다.
        mapping_system_prompt = f"""
        Given the original nightmare keywords and the reconstructed image prompt, identify 3-5 key concepts from the original nightmare that were most significantly reinterpreted or transformed into positive elements in the reconstructed prompt.
        For each, provide the original concept (from the keywords) and its positively reinterpreted counterpart found in the reconstructed prompt.
        Respond strictly in JSON format. Example: [{{{{ "original": "original_concept", "transformed": "transformed_concept" }}}}, ...]
        
        Original Nightmare Keywords: [{keywords_str}]
        Reconstructed Image Prompt: "{reconstructed_prompt}"
        """
        mapping_chain = PromptTemplate.from_template(mapping_system_prompt) | self.llm | self.output_parser
        
        try:
            # invoke 호출 시 필요한 변수가 없다면 빈 딕셔너리 {} 전달
            mapping_raw = mapping_chain.invoke({"dream_text": dream_text}) # dream_text를 context로 제공
            
            # Debugging print (개발 중일 때만 사용)
            # print(f"DEBUG: Raw Mapping Response: {mapping_raw}")

            # LLM이 간혹 JSON 앞뒤에 불필요한 문자를 추가할 수 있으므로, JSON 부분만 추출 시도
            if raw_response := mapping_raw.strip(): # 비어있지 않은지 확인
                if raw_response.startswith("```json") and raw_response.endswith("```"):
                    json_str = raw_response[7:-3].strip()
                else:
                    json_str = raw_response
            else:
                json_str = "" # 응답이 비어있으면 빈 문자열

            if json_str: # JSON 문자열이 있을 경우에만 파싱 시도
                keyword_mappings = json.loads(json_str)
                
                # LLM이 잘못된 형식의 JSON을 줄 수도 있으므로 추가 검증
                if not isinstance(keyword_mappings, list) or not all(isinstance(item, dict) and "original" in item and "transformed" in item for item in keyword_mappings):
                    print(f"경고: 키워드 매핑 JSON 형식이 유효하지 않습니다. 예상치 못한 형식: {json_str}")
                    keyword_mappings = [] # 유효하지 않으면 빈 리스트로 설정
            else:
                keyword_mappings = [] # 빈 응답이면 빈 리스트

        except json.JSONDecodeError as e:
            print(f"경고: 키워드 매핑 생성 중 JSON 파싱 오류: {e}\n원본 응답: {mapping_raw}")
            keyword_mappings = []
        except Exception as e:
            print(f"경고: 키워드 매핑 생성 중 예상치 못한 오류: {e}")
            keyword_mappings = []

        return reconstructed_prompt, transformation_summary, keyword_mappings