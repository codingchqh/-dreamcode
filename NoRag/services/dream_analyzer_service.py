# services/dream_analyzer_service.py (All-in-One 안전 변환 프롬프트 적용)

from core.config import API_KEY
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChain LLM을 직접 사용합니다.
llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY, temperature=0.7)

# derisk_dream_text 함수는 이제 create_nightmare_prompt에 통합되므로 삭제합니다.

def create_nightmare_prompt(dream_text: str) -> str:
    """
    [All-in-One 최종 방식]
    단일 LLM 호출로 '위험 완화'와 'DALL-E 3용 프롬프트 생성'을 동시에 수행합니다.
    """
    system_prompt = """
    You are a 'Safety-First Prompt Artist' for the DALL-E 3 image generator. Your task is to transform a user's nightmare description (in Korean) into a safe, metaphorical, and visually rich image prompt (in English).

    **Your process is a two-step thinking process:**
    1.  **Analyze & De-risk:** First, analyze the Korean text for themes that might violate OpenAI's policy (especially self-harm, hopelessness, violence).
    2.  **Abstract & Create:** Then, create a prompt that represents the *emotion* and *symbolism* of the dream, not the literal events. You must convert any potentially sensitive content into safe, abstract, or artistic metaphors.

    **Strict Safety Rules:**
    - If the dream involves themes of giving up, sinking, or paralysis, represent it symbolically. For example: "A lone figure wrapped in heavy, grey fabric, partially submerged in a misty, still landscape" or "A figure made of crumbling stone, sitting in a vast, empty hall."
    - **NEVER** depict literal self-harm, suicide, or violence.
    - The final output prompt **MUST** be a single paragraph in English.
    - The prompt **MUST NOT** contain any text, letters, or words for the image.
    - Incorporate a surreal, dark fantasy Korean aesthetic.
    """
    
    nightmare_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", dream_text)
    ])

    nightmare_chain = nightmare_prompt_template | llm | StrOutputParser()

    print("[DEBUG] All-in-One 악몽 프롬프트 생성 시작...")
    try:
        final_prompt = nightmare_chain.invoke({})
        print(f"[DEBUG] 생성된 최종 프롬프트: {final_prompt}")
        return final_prompt
    except Exception as e:
        print(f"악몽 프롬프트 생성 중 오류 발생: {e}")
        return "악몽 프롬프트를 생성하는데 실패했습니다."


def create_reconstructed_prompt(dream_text: str) -> str:
    """
    재구성 프롬프트는 '위험 완화'가 필요 없으므로, 원본 텍스트를 바로 받아 처리합니다.
    (군인 등 민감 키워드만 중립적으로 바꾸도록 지시)
    """
    system_prompt = """
    You are a wise and empathetic dream therapist. Your goal is to reframe the user's nightmare into an image of peace, healing, and hope.
    **CRITICAL RULE:** If the dream contains sensitive real-world roles like 'soldier', replace them with neutral terms like 'a young person' or 'a figure'.
    **CONTEXT-AWARE KOREAN AESTHETIC:** Reinterpret the scene within a positive, modern Korean context relevant to the original dream. Avoid stereotypes.
    The final output must be a single-paragraph, English image prompt that is safe and positive. It must NOT contain any text or writing.
    """
    reconstructed_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", dream_text)
    ])

    reconstruction_chain = reconstructed_prompt_template | llm | StrOutputParser()
    
    print("[DEBUG] 재구성 프롬프트 생성 시작...")
    try:
        final_prompt = reconstruction_chain.invoke({})
        print(f"[DEBUG] 생성된 재구성 프롬프트: {final_prompt}")
        return final_prompt
    except Exception as e:
        print(f"재구성 프롬프트 생성 중 오류 발생: {e}")
        return "재구성 프롬프트를 생성하는데 실패했습니다."