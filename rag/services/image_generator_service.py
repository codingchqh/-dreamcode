from openai import OpenAI, APIError
# API 키는 생성자를 통해 주입받으므로, 여기서는 core.config를 임포트하지 않습니다.

class ImageGeneratorService:
    """
    텍스트 프롬프트를 기반으로 이미지를 생성하는 서비스를 제공하는 클래스입니다.
    DALL-E 3 모델을 사용하여 이미지를 생성합니다.
    """
    def __init__(self, api_key: str):
        """
        ImageGeneratorService를 초기화합니다.
        :param api_key: OpenAI API 키
        """
        self.client = OpenAI(api_key=api_key)

    def generate_image_from_prompt(self, prompt: str) -> str:
        """
        주어진 프롬프트를 사용하여 이미지를 생성하고 이미지 URL을 반환합니다.
        :param prompt: 이미지 생성을 위한 텍스트 프롬프트 (영어)
        :return: 생성된 이미지의 URL, 또는 오류 메시지
        """
        try:
            response = self.client.images.generate(
                model="dall-e-3", # DALL-E 3 모델 사용
                prompt=prompt,
                size="1024x1024", # 이미지 크기 (DALL-E 3에서 지원하는 크기)
                quality="standard", # 이미지 품질 (standard 또는 hd)
                n=1, # 생성할 이미지 개수
            )
            
            # 응답 데이터가 유효한지 확인하고 이미지 URL을 반환합니다.
            if response.data and len(response.data) > 0 and response.data[0].url:
                image_url = response.data[0].url
                print(f"이미지 생성 성공, URL: {image_url}") # 디버깅을 위해 콘솔에 출력
                return image_url
            else:
                print("이미지 생성 실패: 응답 데이터 없음 또는 URL 누락.")
                return "이미지 생성 실패: 유효한 이미지 URL을 받을 수 없습니다."

        except APIError as e:
            # OpenAI API 호출 중 발생한 특정 오류 (예: 잘못된 API 키, rate limit 등)
            error_message = f"OpenAI API 오류 발생: 상태 코드 {e.status_code}, 메시지: {e.response.text}"
            print(error_message)
            return f"OpenAI API 오류 발생: {e.status_code} - {e.response.text}"
        except Exception as e:
            # 그 외 예상치 못한 일반적인 오류
            error_message = f"이미지 생성 중 예상치 못한 오류 발생: {e}"
            print(error_message)
            return f"이미지 생성 중 오류 발생: {e}"