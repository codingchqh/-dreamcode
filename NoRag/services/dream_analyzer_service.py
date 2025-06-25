from core.config import API_KEY
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY, temperature=0.3)

def derisk_dream_text(dream_text: str) -> str:
    """
    [신규] DALL-E 3 안전 필터를 통과하기 위해
    꿈 텍스트의 민감한 단어를 은유적/중립적으로 변환하는 체인.
    """
    derisk_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a content policy specialist. Your task is to rewrite the user's text to be safe for an AI image generator. "
             "You must replace words related to violence, gore, explicit content, or other sensitive topics with neutral, metaphorical, or symbolic alternatives. "
             "For example, 'a soldier with a gun' could become 'a uniformed figure with a metallic object'. 'Blood' could become 'crimson stains'. "
             "The core meaning and mood of the text should be preserved, but in a safe-to-generate manner. "
             "The output should be the rewritten text only, in Korean."),
            ("human", "다음 문장을 안전하게 바꿔줘: {text}")
        ]
    )
    
    derisk_chain = derisk_prompt | llm | StrOutputParser()
    
    print("[DEBUG] 원본 텍스트 위험 완화 작업 시작...")
    derisked_text = derisk_chain.invoke({"text": dream_text})
    print(f"[DEBUG] 위험 완화된 텍스트: {derisked_text}")
    
    return derisked_text

def create_nightmare_prompt(derisked_text: str) -> str:
    """
    위험이 완화된 텍스트를 바탕으로 DALL-E 3에 최적화된 상세한 프롬프트를 생성합니다.
    """
    system_prompt = """
    You are a prompt artist for DALL-E 3. Your task is to convert the user's de-risked dream description into a powerful, detailed, single-paragraph image prompt.
    The prompt must be in English.
    The final image must NOT contain any text or letters.
    Focus on creating a rich, atmospheric, and cinematic scene based on the provided text.
    Incorporate a Korean aesthetic as requested.
    """
    try:
        response = llm.invoke(
            [
                ("system", system_prompt),
                ("user", derisked_text)
            ]
        )
        return response.content
    except Exception as e:
        print(f"악몽 프롬프트 생성 중 오류 발생: {e}")
        return "악몽 프롬프트를 생성하는 데 실패했습니다."

def create_reconstructed_prompt(derisked_text: str) -> str:
    """
    위험이 완화된 텍스트를 바탕으로 긍정적인 재구성 프롬프트를 생성합니다.
    """
    system_prompt = """
    You are a wise and empathetic dream therapist. Your goal is to reframe the user's (already de-risked) nightmare into an image of peace and hope.
    Maintain the core, neutral subjects but transform the narrative into a positive one.
    The final output must be a single-paragraph, English image prompt.
    Incorporate a positive, modern Korean aesthetic.
    """
    try:
        response = llm.invoke(
            [
                ("system", system_prompt),
                ("user", derisked_text)
            ]
        )
        return response.content
    except Exception as e:
        print(f"재구성 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 재구성하는 데 실패했습니다."