# services/image_generator_service.py

from openai import OpenAI
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
            image_url = response.data[0].url
            return image_url
        except Exception as e:
            print(f"Error generating image: {e}")
            return f"Failed to generate image: {e}"