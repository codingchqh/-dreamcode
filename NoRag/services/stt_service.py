# services/stt_service.py (수정 전)
# from core.config import API_KEY # <-- 이 부분이 오류의 원인입니다.

# services/stt_service.py (수정 후)

import os
from openai import OpenAI
# config.py에서 settings 객체를 임포트합니다.
from core.config import settings # 이 줄을 추가/수정합니다.

class STTService:
    def __init__(self, api_key: str):
        # 이제 api_key는 생성자를 통해 외부(main.py)에서 주입받습니다.
        self.client = OpenAI(api_key=api_key) 

    def transcribe_audio(self, audio_path: str) -> str:
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text