# services/moderation_service.py

from openai import OpenAI
# API 키는 생성자를 통해 주입받으므로, 여기서는 core.config를 임포트하지 않습니다.

class ModerationService:
    """
    텍스트 내용의 안전성을 검사하는 서비스를 제공하는 클래스입니다.
    OpenAI의 Moderation API를 사용합니다.
    """
    def __init__(self, api_key: str):
        """
        ModerationService를 초기화합니다.
        :param api_key: OpenAI API 키
        """
        self.client = OpenAI(api_key=api_key)

    def check_text_safety(self, text: str) -> dict:
        """
        주어진 텍스트의 안전성을 검사하고 결과를 딕셔너리로 반환합니다.
        :param text: 검사할 텍스트
        :return: flagged (bool), text (str), details (dict)를 포함하는 딕셔너리
        """
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