# services/stt_service.py

import os
from openai import OpenAI
# API 키는 생성자를 통해 주입받으므로, 여기서는 core.config를 임포트하지 않습니다.
# from core.config import settings # <- 이 줄은 삭제되어야 합니다.

class STTService:
    """
    Speech-to-Text (STT) 서비스를 제공하는 클래스입니다.
    오디오 파일을 텍스트로 변환하는 기능을 담당합니다.
    """
    def __init__(self, api_key: str):
        """
        STTService를 초기화합니다.
        :param api_key: OpenAI API 키
        """
        # OpenAI 클라이언트를 API 키를 사용하여 초기화합니다.
        self.client = OpenAI(api_key=api_key) 

    def transcribe_audio(self, audio_path: str) -> str:
        """
        주어진 오디오 파일 경로에서 음성을 텍스트로 변환합니다.
        :param audio_path: 변환할 오디오 파일의 경로
        :return: 변환된 텍스트
        """
        try:
            with open(audio_path, "rb") as audio_file:
                # Whisper 모델을 사용하여 음성을 텍스트로 변환합니다.
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcript.text
        except FileNotFoundError:
            print(f"Error: Audio file not found at {audio_path}")
            return "오디오 파일을 찾을 수 없습니다."
        except Exception as e:
            print(f"Error during audio transcription: {e}")
            return f"음성 변환 중 오류가 발생했습니다: {e}"