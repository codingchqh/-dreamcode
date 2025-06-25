# services/moderation_service.py (수정 전)
# from core.config import API_KEY # <-- 이 부분이 오류의 원인입니다.

# services/moderation_service.py (수정 후)

from openai import OpenAI
# config.py에서 settings 객체를 임포트합니다.
from core.config import settings # 이 줄을 추가/수정합니다.

class ModerationService:
    def __init__(self, api_key: str):
        # 이제 api_key는 생성자를 통해 외부(main.py)에서 주입받습니다.
        self.client = OpenAI(api_key=api_key)

    def check_text_safety(self, text: str) -> dict:
        try:
            response = self.client.moderations.create(input=text)
            moderation_result = response.results[0]
            
            if moderation_result.flagged:
                # 어떤 카테고리가 플래그되었는지 더 상세한 정보를 반환
                flagged_categories = [
                    cat for cat, flag in moderation_result.categories.model_dump().items() if flag
                ]
                return {
                    "flagged": True,
                    "text": f"입력된 내용이 안전 정책을 위반할 수 있습니다: {', '.join(flagged_categories)}",
                    "details": moderation_result.model_dump()
                }
            else:
                return {
                    "flagged": False,
                    "text": "안전합니다.",
                    "details": moderation_result.model_dump()
                }
        except Exception as e:
            print(f"Error during moderation check: {e}")
            return {
                "flagged": True,
                "text": f"안전성 검사 중 오류가 발생했습니다: {e}",
                "details": {"error": str(e)}
            }