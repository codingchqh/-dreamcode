import os # 운영체제와 상호작용하는 os 모듈 임포트
from dotenv import load_dotenv # .env 파일 로드를 위한 load_dotenv 함수 임포트

load_dotenv() # .env 파일의 환경 변수 로드

API_KEY = os.environ.get("OPENAI_API_KEY") # 환경 변수에서 "OPENAI_API_KEY" 값을 가져와 API_KEY에 할당