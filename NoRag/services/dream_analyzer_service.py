# services/dream_analyzer_service.py (자동 재시도 기능이 포함된 최종 완성본)

from openai import OpenAI
from core.config import API_KEY

# LangChain 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
# '자동 재시도 파서'를 추가로 임포트합니다.
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from typing import List

# LangChain의 ChatOpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY, temperature=0)

# 1단계: 키워드 추출을 위한 데이터 구조 정의 (Pydantic 모델)
class DreamKeywords(BaseModel):
    """꿈의 내용에서 추출한 핵심 시각적 키워드"""
    main_character: List[str] = Field(description="꿈의 주인공. 예: 나, 군인, 친구")
    setting: str = Field(description="꿈의 배경이 되는 장소. 예: 어두운 숲, 군대 막사, 옛날 학교")
    key_objects: List[str] = Field(description="꿈에 등장하는 중요한 사물. 예: 총, 거미, 고장난 시계")
    action: str = Field(description="꿈에서 일어나는 핵심적인 행동. 예: 도망치고 있음, 무언가를 찾고 있음")
    atmosphere: str = Field(description="꿈의 전반적인 분위기. 예: 불안함, 공포스러움, 긴박함")

# 1-1. Pydantic 모델을 기반으로 기본 출력 파서 생성
keyword_parser = PydanticOutputParser(pydantic_object=DreamKeywords)

# 1-2. 자동 재시도 파서 생성
# LLM이 잘못된 형식의 출력을 생성하면, 이 파서가 오류를 수정하도록 다시 요청합니다.
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=keyword_parser,
    llm=llm  # 재시도 요청 시 사용할 LLM
)

# 1-3. 키워드 추출을 위한 프롬프트 템플릿 생성
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

# 1-4. 키워드 추출 체인에 '자동 재시도 파서'를 연결합니다.
keyword_extraction_chain = keyword_extraction_prompt | llm | retry_parser


def create_nightmare_prompt(dream_text: str) -> str:
    """
    [LangChain 적용]
    1. 꿈 텍스트에서 핵심 키워드를 추출합니다. (오류 자동 재시도 기능 포함)
    2. 추출된 키워드를 바탕으로 DALL-E용 이미지 프롬프트를 생성합니다.
    """
    try:
        # --- 1단계: 키워드 추출 실행 ---
        print("[DEBUG] 1단계: 꿈 내용 분석 및 키워드 추출 시작...")
        # .invoke()를 호출할 때, Pydantic 파서가 아니라 Retry 파서를 사용하므로 format_instructions를 직접 전달해야 합니다.
        extracted_keywords: DreamKeywords = keyword_extraction_chain.invoke({
            "dream_text": dream_text,
            "format_instructions": keyword_parser.get_format_instructions(),
        })
        print(f"[DEBUG] 추출된 키워드: {extracted_keywords}")

        # --- 2단계: 추출된 키워드를 바탕으로 이미지 프롬프트 생성 ---
        print("[DEBUG] 2단계: 키워드를 바탕으로 최종 프롬프트 조립 시작...")
        
        image_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a prompt artist specializing in dark, atmospheric, and surreal art, creating prompts for DALL-E.
            Your task is to synthesize the provided keywords from a dream into a powerful, single-paragraph image prompt.
            The final prompt must be a visually rich, cinematic, and photorealistic scene that captures the essence of the dream.
            
            **MOST IMPORTANT RULE: You must avoid OpenAI's safety filters.**
            - Instead of literal descriptions of prohibited content, use metaphor and symbolism.
            
            **ARTISTIC STYLE: The overall mood must have a Korean aesthetic.**
            - Incorporate elements of Korean settings or folklore relevant to the keywords.
            
            **CRITICAL INSTRUCTION: The final image must NOT contain any text, letters, or writing.**
            """),
            ("human", """
            Here are the key elements from the dream:
            - Main Character(s): {main_character}
            - Setting: {setting}
            - Key Objects: {key_objects}
            - Core Action: {action}
            - Atmosphere: {atmosphere}
            
            Now, create a single, cohesive image prompt based on these elements.
            """)
        ])

        creative_llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY, temperature=0.8)
        image_generation_chain = image_prompt_template | creative_llm

        # Pydantic 모델은 LangChain이 자동으로 딕셔너리로 변환하여 처리해줍니다.
        final_prompt = image_generation_chain.invoke(extracted_keywords)
        
        return final_prompt.content

    except Exception as e:
        print(f"LangChain 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 생성하는 데 실패했습니다. 입력 내용을 확인해주세요."

# '재구성 프롬프트' 함수는 LangChain을 사용하지 않으므로 그대로 둡니다.
def create_reconstructed_prompt(dream_text: str) -> str:
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
        vanilla_client = OpenAI(api_key=API_KEY)
        response = vanilla_client.completions.create(
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