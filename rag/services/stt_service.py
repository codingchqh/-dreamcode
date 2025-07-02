import os
from openai import OpenAI
import openai # openai의 특정 오류를 처리하기 위해 임포트
from io import BytesIO

class STTService:
    """
    [최종 버전] 파일 경로 또는 메모리 상의 오디오 바이트를
    텍스트로 변환하는 STT(Speech-to-Text) 서비스를 제공합니다.
    """
    def __init__(self, api_key: str):
        """
        STTService를 초기화합니다.
        :param api_key: OpenAI API 키
        """
        self.client = OpenAI(api_key=api_key)

    def _transcribe(self, audio_file_buffer, language: str = "ko") -> str:
        """
        (내부용) 파일 버퍼를 받아 Whisper API를 호출하는 공통 함수
        """
        # Whisper 모델을 사용하여 음성을 텍스트로 변환 요청
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_buffer,
            language=language
        )
        return transcript.text

    def transcribe_audio(self, audio_path: str) -> str:
        """
        주어진 오디오 파일 경로에서 음성을 텍스트로 변환합니다.
        :param audio_path: 변환할 오디오 파일의 경로
        :return: 변환된 텍스트 또는 오류 메시지
        """
        try:
            with open(audio_path, "rb") as audio_file:
                print(f"DEBUG: STTService - '{audio_path}' 파일로 음성 변환을 시작합니다.")
                result = self._transcribe(audio_file)
                print("DEBUG: STTService - 파일 음성 변환 성공.")
                return result
        except FileNotFoundError:
            print(f"ERROR: STTService - 오디오 파일을 찾을 수 없습니다. 경로: {audio_path}")
            return "오디오 파일을 찾을 수 없습니다."
        except openai.AuthenticationError as e:
            print(f"ERROR: STTService - OpenAI API 인증 오류: {e}")
            return "오류: OpenAI API 키가 잘못되었거나 유효하지 않습니다."
        except openai.RateLimitError as e:
            print(f"ERROR: STTService - OpenAI API 사용량 한도 초과: {e}")
            return "오류: API 사용량 한도를 초과했습니다."
        except openai.APIConnectionError as e:
            print(f"ERROR: STTService - OpenAI API 연결 실패: {e}")
            return "오류: OpenAI 서버에 연결할 수 없습니다."
        except Exception as e:
            print(f"ERROR: STTService - 파일 음성 변환 중 알 수 없는 오류 발생: {e}")
            return f"음성 변환 중 알 수 없는 오류가 발생했습니다: {e}"

    def transcribe_from_bytes(self, audio_bytes: bytes, file_name: str = "audio.wav") -> str:
        """
        메모리에 있는 오디오 바이트 데이터에서 음성을 텍스트로 변환합니다.
        :param audio_bytes: 변환할 오디오 파일의 바이트 데이터
        :param file_name: Whisper API에 전달할 임시 파일 이름 (형식 추론용)
        :return: 변환된 텍스트
        """
        try:
            audio_buffer = BytesIO(audio_bytes)
            audio_buffer.name = file_name # API가 파일 형식을 알 수 있도록 이름 지정
            
            print(f"DEBUG: STTService - 바이트 데이터로 음성 변환을 시작합니다. 파일 이름: {file_name}")
            result = self._transcribe(audio_buffer)
            print("DEBUG: STTService - 바이트 데이터 음성 변환 성공.")
            return result
        except Exception as e:
            print(f"ERROR: STTService - 바이트 데이터 음성 변환 중 알 수 없는 오류 발생: {e}")
            # 이 오류는 더 상세하게 나눌 수 있지만, transcribe_audio에서 대부분 처리됩니다.
            return f"오디오 데이터 처리 중 오류가 발생했습니다: {e}"