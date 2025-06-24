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
    You are a prompt artist specializing in dark, atmospheric, and surreal art. Your task is to convert a user's nightmare into a powerful image prompt for DALL-E.

    **MOST IMPORTANT RULE: You must avoid OpenAI's safety filters.**
    - DO NOT use words related to explicit violence, blood, gore, self-harm, or sexual content.
    - Instead of literal descriptions, use metaphor, symbolism, and artistic abstraction (e.g., "crimson liquid" instead of "blood").

    **[NEW] ARTISTIC STYLE: The overall mood must have a Korean aesthetic.**
    - Incorporate elements of Korean settings (e.g., an ancient, moonlit palace; a dense, rainy alleyway in Seoul; a misty mountain temple) or folklore.
    - The final image should feel like a still from a Korean psychological horror film.

    **CRITICAL INSTRUCTION: The final image must NOT contain any text, letters, or writing.**

    Translate the core feeling of the dream into a visually rich, cinematic, and photorealistic scene. Focus on atmosphere, lighting, and composition.
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
    Read the user's nightmare, transform it into a new, positive story, and summarize it into a single-paragraph image prompt for DALL-E.
    The final output must be only the image prompt, and it must be in English.

    **[NEW] AESTHETIC TOUCH: Weave in a touch of Korean beauty or tranquility.**
    - Incorporate elements like traditional clothing (hanbok), architectural elements (hanok), serene nature (a tranquil bamboo forest, a calm temple garden), or gentle, warm lighting.

    **CRITICAL INSTRUCTION: The final image must NOT contain any text, letters, or writing.**

    Apply these transformation principles:
    1.  **Different Outcome:** Change the ending to be hopeful or resolved.
    2.  **Symbolism of Strength:** Insert imagery that symbolizes survival or control.
    3.  **Threat Transformation:** Convert threatening figures into non-threatening ones.
    4.  **Change of Perspective:** Describe the scene from a safe or powerful viewpoint.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dream_text}
            ],
            temperature=1.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"재구성 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 재구성하는 데 실패했습니다."