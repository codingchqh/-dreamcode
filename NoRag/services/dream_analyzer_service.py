from openai import OpenAI
from core.config import API_KEY

# 이 파일은 더 이상 복잡한 LangChain 체인을 사용하지 않고, 안정성을 위해 직접 OpenAI API를 호출하는 방식으로 되돌립니다.
client = OpenAI(api_key=API_KEY)

def create_nightmare_prompt(dream_text: str) -> str:
    system_prompt = """
    You are a prompt artist for DALL-E 2. Your task is to convert a user's dream into a simple, effective, comma-separated keyword prompt.

    **Analysis:**
    1. Read the user's dream and identify the absolute core elements: Main Character, Setting, and core Mood.
    2. Combine these into a very simple phrase.
    3. Append artistic style keywords at the end.

    **Rules:**
    - The entire prompt must be under 1000 characters.
    - The prompt must be in English.
    - Do not describe a scene. List keywords.
    - To avoid safety filters, use neutral terms. Replace 'soldier' with 'a figure', 'gun' with 'a metallic object', etc.
    
    Example:
    User Dream: "군인이 숲에서 총을 들고 쫓기는 꿈"
    Your Output: "A figure in a dark forest, being chased, holding a metallic object, feeling of fear, korean dark fantasy, surreal, cinematic"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dream_text}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"악몽 프롬프트 생성 중 오류 발생: {e}")
        return "악몽 프롬프트를 생성하는 데 실패했습니다."

def create_reconstructed_prompt(dream_text: str) -> str:
    system_prompt = """
    You are a wise and empathetic dream therapist. Your goal is to reframe the user's nightmare into an image of peace, healing, and hope.
    **CRITICAL RULE:** Replace sensitive keywords with neutral alternatives (e.g., 'soldier' becomes 'a young person').
    **CONTEXT-AWARE KOREAN AESTHETIC:** Reinterpret the scene within a positive, modern Korean context.
    The final output must be a single-paragraph, English image prompt that is safe and positive.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dream_text}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"재구성 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 재구성하는 데 실패했습니다."