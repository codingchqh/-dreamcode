# services/stt_service.py (안전성 강화 버전)

import os
from openai import OpenAI
import openai # openai의 특정 오류를 잡기 위해 임포트

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
        self.client = OpenAI(api_key=api_key) 

    def transcribe_audio(self, audio_path: str) -> str:
        """
        주어진 오디오 파일 경로에서 음성을 텍스트로 변환합니다.
        :param audio_path: 변환할 오디오 파일의 경로
        :return: 변환된 텍스트
        """
        try:
            with open(audio_path, "rb") as audio_file:
                print(f"DEBUG: STTService - '{audio_path}' 파일로 음성 변환을 시작합니다.")
                # Whisper 모델을 사용하여 음성을 텍스트로 변환합니다.
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko" # 한국어 인식률 향상을 위해 언어 지정
                )
                print("DEBUG: STTService - 음성 변환 성공.")
                return transcript.text
        
        except FileNotFoundError:
            print(f"ERROR: STTService - 오디오 파일을 찾을 수 없습니다. 경로: {audio_path}")
            return "오디오 파일을 찾을 수 없습니다."
        
        # --- 🔽 더 자세한 오류 처리를 위해 추가된 부분 🔽 ---
        except openai.AuthenticationError as e:
            print(f"ERROR: STTService - OpenAI API 인증 오류: {e}")
            return "오류: OpenAI API 키가 잘못되었거나 유효하지 않습니다. 환경변수를 확인해주세요."
        except openai.RateLimitError as e:
            print(f"ERROR: STTService - OpenAI API 사용량 한도 초과: {e}")
            return "오류: API 사용량 한도를 초과했습니다. 잠시 후 다시 시도하거나 플랜을 확인해주세요."
        except openai.APIConnectionError as e:
            print(f"ERROR: STTService - OpenAI API 연결 실패: {e}")
            return "오류: OpenAI 서버에 연결할 수 없습니다. 인터넷 연결을 확인해주세요."
        # --- 🔼 여기까지 추가된 부분 🔼 ---

        except Exception as e:
            print(f"ERROR: STTService - 음성 변환 중 알 수 없는 오류 발생: {e}")
            return f"음성 변환 중 알 수 없는 오류가 발생했습니다: {e}"