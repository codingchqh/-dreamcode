from openai import OpenAI
from core.config import API_KEY

if not API_KEY:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")

client = OpenAI(api_key=API_KEY)

def generate_image_from_prompt(prompt_text: str) -> str:
    """
    주어진 텍스트 프롬프트를 사용하여 DALL-E 3로 이미지를 생성합니다.

    Args:
        prompt_text (str): 이미지 생성을 위한 상세한 영어 프롬프트

    Returns:
        str: 생성된 이미지의 URL 또는 오류 메시지
    """
    # [디버깅] 어떤 프롬프트가 요청되는지 터미널에 출력
    print("---")
    print(f"[DEBUG] 이미지 생성 요청 프롬프트: {prompt_text}")
    print("---")

    # [안정성] 프롬프트가 비어있는 경우, API 요청 없이 바로 오류 반환
    if not prompt_text or not prompt_text.strip():
        return "이미지 생성을 위한 프롬프트가 비어있습니다."

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt_text,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        # 오류 메시지를 더 명확하게 출력
        print(f"이미지 생성 중 오류 발생: {e}")
        return f"이미지 생성 중 오류가 발생했습니다. (오류: {e})"