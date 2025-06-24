from openai import OpenAI
from core.config import API_KEY

if not API_KEY:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")

client = OpenAI(api_key=API_KEY)

def create_nightmare_prompt(dream_text: str) -> str:
    """
    악몽 텍스트를 DALL-E가 이미지를 잘 생성할 수 있는 상세한 영어 프롬프트로 변환합니다.
    """
    # 이 함수는 그대로 유지합니다.
    system_prompt = """
    You are a prompt artist specializing in dark, atmospheric, and surreal art. Your task is to convert a user's nightmare into a powerful image prompt for DALL-E.

    **MOST IMPORTANT RULE: You must avoid OpenAI's safety filters.**
    - DO NOT use words related to explicit violence, blood, gore, self-harm, or sexual content.
    - Instead of literal descriptions, use metaphor, symbolism, and artistic abstraction (e.g., "crimson liquid" instead of "blood").

    **ARTISTIC STYLE: The overall mood must have a Korean aesthetic.**
    - Incorporate elements of Korean settings (e.g., an ancient, moonlit palace; a dense, rainy alleyway in Seoul; a misty mountain temple) or folklore.
    - The final image should feel like a still from a Korean psychological horror film.

    **CRITICAL INSTRUCTION: The final image must NOT contain any text, letters, or writing.**

    Translate the core *feeling* of the dream (e.g., fear, helplessness, anxiety) into a visually rich, cinematic, and photorealistic scene. Focus on atmosphere, lighting, and composition.
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
    # [최종 수정] 원래 꿈의 맥락을 유지하도록 시스템 프롬프트 대폭 수정
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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dream_text}
            ],
            temperature=0.7, # 더 예측 가능하고 안정적인 결과를 위해 온도를 약간 낮춤
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"재구성 프롬프트 생성 중 오류 발생: {e}")
        return "프롬프트를 재구성하는 데 실패했습니다."