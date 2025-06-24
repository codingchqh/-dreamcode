from openai import OpenAI
from core.config import API_KEY
# 모더레이션 서비스를 이 파일에서도 사용하기 위해 임포트합니다.
from services import moderation_service

if not API_KEY:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")

client = OpenAI(api_key=API_KEY)

def generate_image_from_prompt(prompt_text: str) -> str:
    """
    주어진 텍스트 프롬프트를 사용하여 DALL-E 3로 이미지를 생성합니다.
    (이미지 생성 전, 프롬프트 자체에 대한 2차 안전성 검사 포함)

    Args:
        prompt_text (str): 이미지 생성을 위한 상세한 영어 프롬프트

    Returns:
        str: 생성된 이미지의 URL 또는 오류 메시지
    """
    print("---")
    print(f"[DEBUG] 이미지 생성 요청 프롬프트: {prompt_text}")
    print("---")

    if not prompt_text or not prompt_text.strip():
        return "이미지 생성을 위한 프롬프트가 비어있습니다."

    # [이중 안전 검사] 생성된 프롬프트가 안전한지 DALL-E 전송 전에 최종 확인
    print("[DEBUG] 2차 안전성 검사: 생성된 프롬프트가 안전한지 확인 중...")
    safety_check = moderation_service.check_text_safety(prompt_text)
    if safety_check["flagged"]:
        # 생성된 프롬프트가 안전하지 않으면, DALL-E에 보내지 않고 즉시 중단
        error_message = f"생성된 프롬프트가 안전 정책에 위반되어 이미지를 만들 수 없습니다. (사유: {safety_check['text']})"
        print(f"[MODERATION] {error_message}")
        return error_message

    # 2차 안전성 검사를 통과한 경우에만 이미지 생성을 시도
    try:
        print("[DEBUG] 2차 안전성 검사 통과. DALL-E 이미지 생성 시작...")
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt_text,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        return f"이미지 생성 중 오류가 발생했습니다. (오류: {e})"