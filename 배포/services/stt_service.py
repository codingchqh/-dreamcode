import os # 운영체제와 상호작용하는 기능을 제공하는 os 모듈 임포트
from openai import OpenAI # OpenAI API와 통신하기 위한 OpenAI 클라이언트 임포트
import openai # openai의 특정 오류(AuthenticationError, RateLimitError, APIConnectionError 등)를 처리하기 위해 임포트
from io import BytesIO # 메모리 내에서 바이너리 데이터를 파일처럼 다룰 수 있게 해주는 BytesIO 임포트

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
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=api_key)

    def transcribe_audio(self, audio_path: str) -> str:
        """
        주어진 오디오 파일 경로에서 음성을 텍스트로 변환합니다.
        :param audio_path: 변환할 오디오 파일의 경로
        :return: 변환된 텍스트
        """
        try:
            # 오디오 파일을 바이너리 읽기 모드로 열기
            with open(audio_path, "rb") as audio_file:
                print(f"DEBUG: STTService - '{audio_path}' 파일로 음성 변환을 시작합니다.")
                # Whisper 모델을 사용하여 음성을 텍스트로 변환 요청
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", # 사용할 STT 모델 지정
                    file=audio_file, # 변환할 오디오 파일
                    language="ko" # 한국어 인식을 위해 언어 지정
                )
                print("DEBUG: STTService - 음성 변환 성공.")
                return transcript.text # 변환된 텍스트 반환
        
        except FileNotFoundError:
            # 파일이 없을 경우 오류 처리
            print(f"ERROR: STTService - 오디오 파일을 찾을 수 없습니다. 경로: {audio_path}")
            return "오디오 파일을 찾을 수 없습니다."
        
        except openai.AuthenticationError as e:
            # OpenAI API 인증 오류 처리
            print(f"ERROR: STTService - OpenAI API 인증 오류: {e}")
            return "오류: OpenAI API 키가 잘못되었거나 유효하지 않습니다. 환경변수를 확인해주세요."
        except openai.RateLimitError as e:
            # OpenAI API 사용량 한도 초과 오류 처리
            print(f"ERROR: STTService - OpenAI API 사용량 한도 초과: {e}")
            return "오류: API 사용량 한도를 초과했습니다. 잠시 후 다시 시도하거나 플랜을 확인해주세요."
        except openai.APIConnectionError as e:
            # OpenAI API 연결 실패 오류 처리
            print(f"ERROR: STTService - OpenAI API 연결 실패: {e}")
            return "오류: OpenAI 서버에 연결할 수 없습니다. 인터넷 연결을 확인해주세요."
        except Exception as e:
            # 그 외 모든 예외 처리
            print(f"ERROR: STTService - 음성 변환 중 알 수 없는 오류 발생: {e}")
            return f"음성 변환 중 알 수 없는 오류가 발생했습니다: {e}"

    def transcribe_from_bytes(self, audio_bytes: bytes, file_name: str = "audio.wav") -> str:
        """
        오디오 바이트 데이터에서 음성을 텍스트로 변환합니다.
        메모리 내 바이트 데이터를 처리할 수 있도록 추가된 메서드입니다.
        :param audio_bytes: 변환할 오디오 파일의 바이트 데이터
        :param file_name: Whisper API에 전달할 임시 파일 이름 (형식 추론용, .wav, .mp3 등)
        :return: 변환된 텍스트
        """
        try:
            # BytesIO를 사용하여 바이트 데이터를 파일 객체처럼 생성
            audio_file_like = BytesIO(audio_bytes)
            # OpenAI API가 파일 이름을 요구하므로 임시 파일 이름 설정
            audio_file_like.name = file_name

            print(f"DEBUG: STTService - 바이트 데이터로 음성 변환을 시작합니다. 파일 이름: {file_name}")
            # Whisper 모델을 사용하여 바이트 데이터로부터 음성을 텍스트로 변환 요청
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", # 사용할 STT 모델 지정
                file=audio_file_like, # 변환할 바이트 데이터 (파일 객체처럼)
                language="ko" # 한국어 인식을 위해 언어 지정
            )
            print("DEBUG: STTService - 바이트 데이터 음성 변환 성공.")
            return transcript.text # 변환된 텍스트 반환

        except openai.AuthenticationError as e:
            # OpenAI API 인증 오류 처리
            print(f"ERROR: STTService - OpenAI API 인증 오류: {e}")
            return "오류: OpenAI API 키가 잘못되었거나 유효하지 않습니다. 환경변수를 확인해주세요."
        except openai.RateLimitError as e:
            # OpenAI API 사용량 한도 초과 오류 처리
            print(f"ERROR: STTService - OpenAI API 사용량 한도 초과: {e}")
            return "오류: API 사용량 한도를 초과했습니다. 잠시 후 다시 시도하거나 플랜을 확인해주세요."
        except openai.APIConnectionError as e:
            # OpenAI API 연결 실패 오류 처리
            print(f"ERROR: STTService - OpenAI API 연결 실패: {e}")
            return "오류: OpenAI 서버에 연결할 수 없습니다. 인터넷 연결을 확인해주세요."
        except Exception as e:
            # 그 외 모든 예외 처리
            print(f"ERROR: STTService - 바이트 데이터 음성 변환 중 알 수 없는 오류 발생: {e}")
            return f"오류: 바이트 데이터 음성 변환 중 알 수 없는 오류가 발생했습니다: {e}"