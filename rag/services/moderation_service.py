from openai import OpenAI # OpenAI API와 통신하기 위한 OpenAI 클라이언트 임포트

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
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=api_key)

    def check_text_safety(self, text: str) -> dict:
        """
        주어진 텍스트의 안전성을 검사하고 결과를 딕셔너리로 반환합니다.
        :param text: 검사할 텍스트
        :return: flagged (bool), text (str), details (dict)를 포함하는 딕셔너리
        """
        try:
            # Moderation API를 호출하여 텍스트 안전성 검사
            response = self.client.moderations.create(input=text)
            # 검사 결과의 첫 번째 요소 가져오기
            moderation_result = response.results[0]
            
            # 텍스트가 안전 정책을 위반했는지 확인
            if moderation_result.flagged:
                # 플래그된 카테고리 목록 생성
                flagged_categories = [
                    cat for cat, flag in moderation_result.categories.model_dump().items() if flag
                ]
                # 안전 정책 위반 결과 반환
                return {
                    "flagged": True,
                    "text": f"입력된 내용이 안전 정책을 위반할 수 있습니다: {', '.join(flagged_categories)}",
                    "details": moderation_result.model_dump()
                }
            else:
                # 안전한 경우 결과 반환
                return {
                    "flagged": False,
                    "text": "안전합니다.",
                    "details": moderation_result.model_dump()
                }
        except Exception as e:
            # 오류 발생 시 에러 메시지 출력 및 오류 결과 반환
            print(f"Error during moderation check: {e}")
            return {
                "flagged": True,
                "text": f"안전성 검사 중 오류가 발생했습니다: {e}",
                "details": {"error": str(e)}
            }