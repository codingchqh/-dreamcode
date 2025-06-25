# services/image_generator_service.py (수정 전)
# from core.config import API_KEY # <-- 이 부분이 오류의 원인입니다.

# services/image_generator_service.py (수정 후)

from openai import OpenAI
# config.py에서 settings 객체를 임포트합니다.
from core.config import settings # 이 줄을 추가/수정합니다.

class ImageGeneratorService:
    def __init__(self, api_key: str):
        # 이제 api_key는 생성자를 통해 외부(main.py)에서 주입받습니다.
        self.client = OpenAI(api_key=api_key)

    def generate_image_from_prompt(self, prompt: str) -> str:
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