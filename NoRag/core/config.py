# core/config.py

import os

class Settings:
    """
    애플리케이션 설정을 관리하는 클래스입니다.
    API 키를 시스템 환경 변수에서 직접 가져옵니다.
    """

    # OpenAI API 키
    # 시스템 환경 변수에 OPENAI_API_KEY가 설정되어 있어야 합니다.
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # 예시: 다른 설정값들
    # DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./mydatabase.db")
    # DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"

    def __init__(self):
        # API 키가 로드되지 않았을 경우 경고를 줄 수 있습니다.
        if not self.OPENAI_API_KEY:
            print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. API 관련 기능이 작동하지 않을 수 있습니다.")


# Settings 클래스의 인스턴스를 생성하여 프로젝트 전역에서 접근할 수 있도록 합니다.
settings = Settings()