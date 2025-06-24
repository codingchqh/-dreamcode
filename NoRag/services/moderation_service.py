# services/moderation_service.py

from langchain_openai import OpenAIModerationChain
from core.config import API_KEY
from openai import OpenAI

# 모더레이션 체인은 일반 LLM이 아닌, 특화된 모더레이션 엔드포인트를 사용합니다.
# LangChain v0.2.0 이상에서는 client 인스턴스를 직접 전달해야 할 수 있습니다.
# 이 코드는 최신 버전에 맞춰 작성되었습니다.
moderation_chain = OpenAIModerationChain(client=OpenAI(api_key=API_KEY))

def check_text_safety(text: str) -> dict:
    """
    주어진 텍스트를 OpenAI Moderation API로 검사합니다.

    Args:
        text (str): 검사할 텍스트

    Returns:
        dict: 검사 결과. 'flagged': True/False, 'text': 원본 또는 에러 메시지
    """
    try:
        # .run() 메소드는 텍스트에 문제가 있으면 ValueError를 발생시킵니다.
        # 문제가 없으면 원본 텍스트를 그대로 반환합니다.
        moderated_text = moderation_chain.run(text)
        return {"flagged": False, "text": moderated_text}
    except ValueError as e:
        # 오류가 발생하면, 문제가 있다는 의미입니다.
        print(f"[MODERATION] 콘텐츠 정책 위반 감지: {e}")
        return {"flagged": True, "text": f"입력하신 내용에 부적절한 표현이 포함되어 있어 처리할 수 없습니다. (감지된 정책: {e})"}