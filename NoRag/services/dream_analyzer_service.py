# services/dream_analyzer_service.py

from openai import OpenAI
from core.config import API_KEY

if not API_KEY:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")

client = OpenAI(api_key=API_KEY)

def create_nightmare_prompt(dream_text: str) -> str:
    """
    악몽 텍스트를 DALL-E가 이미지를 잘 생성할 수 있는 상세한 영어 프롬프트로 변환합니다.
    """
    system_prompt = """
    You are a prompt artist who specializes in creating vivid, detailed, and emotionally resonant prompts for an AI image generator (like DALL-E).
    Your task is to convert the user's dream description into a powerful, single-paragraph image prompt.
    The prompt must be in English.
    Focus on visual details, atmosphere, lighting, and key objects or figures.
    Translate the core feeling of the dream into a visually descriptive language.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dream_text}
            ],
            temperature=0.8,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"악몽 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 생성하는 데 실패했습니다."

def create_reconstructed_prompt(dream_text: str) -> str:
    """
    악몽 텍스트를 긍정적이고 희망적인 내용으로 재구성하여 새로운 이미지 프롬프트로 만듭니다.
    """
    system_prompt = """
    You are a gentle and creative dream therapist. Your goal is to help the user reframe their nightmare into a positive, empowering, or even whimsical experience.
    Read the user's nightmare and transform it into a new, positive story. Then, summarize that new story into a single-paragraph image prompt for an AI image generator (like DALL-E).
    The final output must be only the image prompt, and it must be in English.

    Apply these transformation principles:
    1.  **Different Outcome:** Change the ending to be hopeful or resolved.
    2.  **Symbolism of Strength:** Insert imagery that symbolizes survival, resilience, or control (e.g., a small light in the darkness, a blooming flower on a barren landscape).
    3.  **Threat Transformation:** Convert threatening figures or objects into non-threatening or even friendly ones (e.g., a monster becomes a fluffy, sleeping giant; a weapon turns into a bouquet of flowers).
    4.  **Change of Perspective:** Describe the scene from a safe or powerful viewpoint (e.g., watching the scene on a TV that can be turned off, or floating above it peacefully).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dream_text}
            ],
            temperature=1.0, # 창의성을 높이기 위해 온도를 약간 높임
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"재구성 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 재구성하는 데 실패했습니다."