# services/stt_service.py

import os
from openai import OpenAI
import io # 메모리 상의 바이트 데이터를 다루기 위해 추가

class STTService:
    """
    [개선된 버전] 파일 경로 또는 메모리 상의 오디오 바이트를
    텍스트로 변환하는 STT(Speech-to-Text) 서비스를 제공합니다.
    """
    def __init__(self, api_key: str):
        """
        STTService를 초기화합니다.
        :param api_key: OpenAI API 키
        """
        self.client = OpenAI(api_key=api_key)

    def _transcribe(self, audio_file_buffer) -> str:
        """
        (내부용) 파일 버퍼를 받아 Whisper API를 호출하는 공통 함수
        """
        # Whisper 모델을 사용하여 음성을 텍스트로 변환합니다.
        # language="ko" 옵션을 추가하여 한국어 인식률을 높입니다.
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_buffer,
            language="ko" # 한국어 명시
        )
        return transcript.text

    def transcribe_from_file(self, audio_path: str) -> str:
        """
        주어진 오디오 파일 경로에서 음성을 텍스트로 변환합니다.
        :param audio_path: 변환할 오디오 파일의 경로
        :return: 변환된 텍스트 또는 오류 메시지
        """
        try:
            with open(audio_path, "rb") as audio_file:
                return self._transcribe(audio_file)
        except FileNotFoundError:
            print(f"Error: Audio file not found at {audio_path}")
            return "오디오 파일을 찾을 수 없습니다."
        except Exception as e:
            print(f"Error during audio transcription from file: {e}")
            return f"파일 음성 변환 중 오류: {e}"

    def transcribe_from_bytes(self, audio_bytes: bytes) -> str:
        """
        메모리에 있는 오디오 바이트 데이터에서 음성을 텍스트로 변환합니다.
        :param audio_bytes: 변환할 오디오 데이터 (바이트)
        :return: 변환된 텍스트 또는 오류 메시지
        """
        try:
            # 바이트 데이터를 파일처럼 다루기 위해 io.BytesIO 사용
            audio_buffer = io.BytesIO(audio_bytes)
            # Whisper API가 파일 이름을 요구하는 경우가 있으므로, 임의의 이름을 지정해줍니다.
            audio_buffer.name = "temp_audio.mp3"
            
            return self._transcribe(audio_buffer)
        except Exception as e:
            print(f"Error during audio transcription from bytes: {e}")
            return f"바이트 음성 변환 중 오류: {e}"