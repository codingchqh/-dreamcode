# services/stt_service.py

from openai import OpenAI
from core.config import API_KEY

# API 키가 설정되지 않았다면 에러를 발생시킬 수 있도록 처리
if not API_KEY:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수를 확인하세요.")

client = OpenAI(api_key=API_KEY)

def transcribe_audio(audio_file_path: str) -> str:
    """
    주어진 오디오 파일 경로를 사용하여 음성을 텍스트로 변환합니다.

    Args:
        audio_file_path (str): 변환할 오디오 파일의 경로

    Returns:
        str: 변환된 텍스트
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"음성 변환 중 오류 발생: {e}")
        return "음성을 변환하는 데 실패했습니다. 파일을 다시 확인해주세요."