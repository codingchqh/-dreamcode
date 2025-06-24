from openai import OpenAI
from core.config import API_KEY

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from typing import List

llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY, temperature=0)

class DreamKeywords(BaseModel):
    """꿈의 내용에서 추출한 핵심 시각적 키워드"""
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
    [최종 수정] 1. 꿈에서 키워드 추출 -> 2. 키워드를 직접 조합하여 단순한 프롬프트 생성
    """
    try:
        print("[DEBUG] 1단계: 꿈 내용 분석 및 키워드 추출 시작...")
        extracted_keywords: DreamKeywords = keyword_extraction_chain.invoke({
            "dream_text": dream_text,
            "format_instructions": keyword_parser.get_format_instructions(),
        })
        print(f"[DEBUG] 추출된 키워드: {extracted_keywords}")

        # [최종 수정] 2단계: LLM을 다시 호출하는 대신, 추출된 키워드를 직접 조합
        print("[DEBUG] 2단계: 키워드를 조합하여 단순 프롬프트 생성...")

        # 키워드들을 리스트로 합친다
        prompt_parts = []
        prompt_parts.extend(extracted_keywords.main_character)
        prompt_parts.append(extracted_keywords.setting)
        prompt_parts.extend(extracted_keywords.key_objects)
        prompt_parts.append(extracted_keywords.action)
        prompt_parts.append(extracted_keywords.atmosphere)
        
        # 스타일 키워드를 추가한다
        style_keywords = "in a Korean setting, dark, atmospheric, surreal, photorealistic, cinematic lighting, psychological horror style"
        
        # 모든 키워드를 쉼표로 연결하여 최종 프롬프트를 만든다
        final_prompt = ", ".join(prompt_parts) + ", " + style_keywords
        
        return final_prompt

    except Exception as e:
        print(f"LangChain 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 생성하는 데 실패했습니다. 입력 내용을 확인해주세요."

def create_reconstructed_prompt(dream_text: str) -> str:
    """
    악몽 텍스트를 긍정적이고 희망적인 내용으로 재구성하여 새로운 이미지 프롬프트로 만듭니다.
    (이 함수는 정상 작동하므로 기존 방식을 유지합니다)
    """
    system_prompt = """
    You are a wise and empathetic dream therapist. Your goal is to reframe the user's nightmare into an image of peace, healing, and hope.

    **MOST IMPORTANT RULE: You must maintain the original characters and setting of the dream.** For example, if the dream is about a soldier, the reconstructed image must also feature a soldier. Do not replace them with unrelated subjects.

    **CONTEXT-AWARE KOREAN AESTHETIC:** The scene should be reinterpreted within a positive Korean context that is relevant to the original dream.
    - For a soldier's dream, this could mean showing them during a peaceful moment on a modern Korean base, being welcomed home in a modern Korean city, or finding tranquility in a beautiful Korean landscape like the DMZ filled with wildflowers instead of tension.
    - **Avoid stereotypes like hanboks or ancient palaces unless they were in the original dream.**

    **TRANSFORMATION GOAL:** Transform the negative narrative and emotions. Apply these principles while keeping the original subjects:
    1.  **Different Outcome:** The conflict is resolved, the threat is gone.
    2.  **Symbolism of Peace:** Replace symbols of danger with symbols of peace (e.g., a rifle is set down and a flower grows from its barrel).
    3.  **Change of Emotion:** The character's expression should be one of relief, peace, or hope, not fear.

    The final output must be a single-paragraph, English image prompt that is safe, positive, and directly related to the user's original dream context. **It must NOT contain any text, letters, or writing.**
    """
    
    try:
        # 이 부분은 OpenAI 직접 호출을 유지
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