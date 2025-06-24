from openai import OpenAI
from core.config import API_KEY

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from typing import List

llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY, temperature=0)

class DreamKeywords(BaseModel):
    main_character: List[str] = Field(description="The main character(s) of the dream. e.g., 'I', 'a soldier', 'a friend'")
    setting: str = Field(description="The background or place of the dream. e.g., 'a dark forest', 'a military barrack', 'an old school'")
    key_objects: List[str] = Field(description="Important objects appearing in the dream. e.g., 'a gun', 'a spider', 'a broken clock'")
    action: str = Field(description="The core action happening in the dream. e.g., 'running away', 'searching for something'")
    atmosphere: str = Field(description="The overall mood of the dream. e.g., 'anxious', 'terrifying', 'urgent'")

keyword_parser = PydanticOutputParser(pydantic_object=DreamKeywords)
retry_parser = RetryWithErrorOutputParser.from_llm(parser=keyword_parser, llm=llm)

keyword_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert analyst who extracts key visual elements from a dream description. "
            "Analyze the user's dream text and extract the required information. "
            "Your output must be formatted as a JSON object that strictly follows the provided schema. "
            "All keywords must be in English.\n\n{format_instructions}",
        ),
        ("human", "{dream_text}"),
    ]
)

keyword_extraction_chain = keyword_extraction_prompt | llm | retry_parser

def create_nightmare_prompt(dream_text: str) -> str:
    """
    1. 꿈에서 키워드 추출 -> 2. 키워드를 직접 조합하여 단순한 프롬프트 생성
    """
    try:
        print("[DEBUG] 1단계: 꿈 내용 분석 및 키워드 추출 시작...")
        extracted_keywords: DreamKeywords = keyword_extraction_chain.invoke({
            "dream_text": dream_text,
            "format_instructions": keyword_parser.get_format_instructions(),
        })
        print(f"[DEBUG] 추출된 키워드: {extracted_keywords}")

        print("[DEBUG] 2단계: 키워드를 조합하여 단순 프롬프트 생성...")

        prompt_parts = []
        prompt_parts.extend(extracted_keywords.main_character)
        prompt_parts.append(extracted_keywords.setting)
        prompt_parts.extend(extracted_keywords.key_objects)
        prompt_parts.append(extracted_keywords.action)
        prompt_parts.append(extracted_keywords.atmosphere)
        
        style_keywords = "in a Korean setting, dark, atmospheric, surreal, photorealistic, cinematic lighting, psychological horror style"
        
        final_prompt = ", ".join(prompt_parts) + ", " + style_keywords
        
        return final_prompt

    except Exception as e:
        print(f"LangChain 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 생성하는 데 실패했습니다. 입력 내용을 확인해주세요."

def create_reconstructed_prompt(dream_text: str) -> str:
    """
    [최종 테스트] 군인/군부대 키워드를 중립적인 단어로 대체하여 DALL-E 필터를 통과하는지 확인합니다.
    """
    system_prompt = """
    You are a wise and empathetic dream therapist. Your goal is to reframe the user's nightmare into an image of peace, healing, and hope.

    **CRITICAL RULE: The user's dream may contain sensitive keywords like 'soldier' or 'military'. You MUST replace these keywords with neutral, non-military alternatives.**
    - Replace 'soldier' with 'a young person', 'a figure', or 'an individual'.
    - Replace 'military base' or 'barracks' with 'a peaceful retreat', 'a quiet minimalist room', or 'a communal cabin'.
    - Maintain the original emotions and narrative arc (e.g., from distress to peace), but transpose them into a non-military setting.

    **AESTHETIC:** The reinterpreted scene should have a serene, hopeful Korean aesthetic, using elements of nature, light, and modern minimalist design. Avoid overt traditional symbols unless relevant.

    The final output must be a single-paragraph, English image prompt that is safe, positive, and completely free of military-related terms. **It must NOT contain any text, letters, or writing.**
    """
    
    try:
        vanilla_client = OpenAI(api_key=API_KEY)
        response = vanilla_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dream_text}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"재구성 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 재구성하는 데 실패했습니다."