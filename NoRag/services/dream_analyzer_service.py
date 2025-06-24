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
    try:
        # --- 1단계: 키워드 추출 ---
        print("[DEBUG] 1단계: 꿈 내용 분석 및 키워드 추출 시작...")
        extracted_keywords: DreamKeywords = keyword_extraction_chain.invoke({
            "dream_text": dream_text,
            "format_instructions": keyword_parser.get_format_instructions(),
        })
        print(f"[DEBUG] 추출된 키워드: {extracted_keywords}")

        # --- [최종 수정] 2단계: 키워드를 바탕으로 '짧은 문장' 프롬프트 생성 ---
        print("[DEBUG] 2단계: 키워드를 바탕으로 DALL-E 2를 위한 짧은 문장 프롬프트 생성 시작...")
        
        sentence_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a master of concise, powerful prompts for the DALL-E 2 image model.
            Your task is to combine the given dream keywords into a single, coherent English sentence.

            **CRITICAL CONSTRAINTS:**
            1.  The final sentence MUST be less than 150 words to stay within the 1000-character limit of the DALL-E 2 API.
            2.  Focus on combining the character, action, and setting into a clear narrative.
            3.  Append a few key artistic styles at the end of the sentence.
            4.  The final output must be ONLY the prompt sentence itself.
            """),
            ("human", """
            Here are the key elements from the dream:
            - Main Character(s): {main_character}
            - Setting: {setting}
            - Key Objects: {key_objects}
            - Core Action: {action}
            - Atmosphere: {atmosphere}
            
            Now, create one single descriptive sentence for DALL-E 2, ending with styles like 'in a Korean setting, dark, surreal, photorealistic, cinematic lighting'.
            """)
        ])

        creative_llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY, temperature=0.7)
        sentence_generation_chain = sentence_prompt_template | creative_llm
        final_prompt = sentence_generation_chain.invoke(extracted_keywords.dict())
        
        return final_prompt.content.strip()

    except Exception as e:
        print(f"LangChain 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 생성하는 데 실패했습니다. 입력 내용을 확인해주세요."

def create_reconstructed_prompt(dream_text: str) -> str:
    # 이 함수는 정상 작동하므로 그대로 둡니다.
    system_prompt = """
    You are a wise and empathetic dream therapist. Your goal is to reframe the user's nightmare into an image of peace, healing, and hope.
    **CRITICAL RULE: You must maintain the original characters and setting of the dream.** Replace sensitive keywords with neutral alternatives (e.g., 'soldier' becomes 'a young person').
    **CONTEXT-AWARE KOREAN AESTHETIC:** Reinterpret the scene within a positive, modern Korean context relevant to the original dream. Avoid stereotypes.
    The final output must be a single-paragraph, English image prompt that is safe and positive. **It must NOT contain any text or writing.**
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