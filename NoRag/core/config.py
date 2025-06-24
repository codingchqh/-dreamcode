import os
from dotenv import load_dotenv

# 이 함수는 로컬에서 개발할 때 프로젝트 루트에 .env 파일을 만들어두면
# 그 파일의 변수를 시스템 환경변수처럼 로드해주는 편리한 기능입니다.
# 서버에 배포할 때는 보통 직접 시스템 환경변수를 설정하므로 필요없지만,
# 로컬 개발 편의성을 위해 추가하는 것을 권장합니다.
load_dotenv()

# 시스템 환경변수에서 'OPENAI_API_KEY'를 가져옵니다.
API_KEY = os.environ.get("OPENAI_API_KEY")
