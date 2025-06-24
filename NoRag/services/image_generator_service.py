from openai import OpenAI
from core.config import API_KEY
from services import moderation_service

if not API_KEY:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")

client = OpenAI(api_key=API_KEY)

def generate_image_from_prompt(prompt_text: str) -> str:
    """
    주어진 텍스트 프롬프트를 사용하여 이미지를 생성합니다.
    (DALL-E 2 모델 사용)
    """
    print("---")
    print(f"[DEBUG] 이미지 생성 요청 프롬프트: {prompt_text}")
    print("---")

    if not prompt_text or not prompt_text.strip():
        return "이미지 생성을 위한 프롬프트가 비어있습니다."

    print("[DEBUG] 2차 안전성 검사: 생성된 프롬프트가 안전한지 확인 중...")
    safety_check = moderation_service.check_text_safety(prompt_text)
    if safety_check["flagged"]:
        error_message = f"생성된 프롬프트가 안전 정책에 위반되어 이미지를 만들 수 없습니다. (사유: {safety_check['text']})"
        print(f"[MODERATION] {error_message}")
        return error_message

    try:
        print("[DEBUG] 2차 안전성 검사 통과. DALL-E 이미지 생성 시작...")
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt_text,
            size="1024x1024",
            # [수정됨] DALL-E 2와 호환되지 않는 quality 파라미터 삭제
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        return f"이미지 생성 중 오류가 발생했습니다. (오류: {e})"