# services/moderation_service.py (수정된 최종본)

from openai import OpenAI
from core.config import API_KEY
# [수정됨] langchain_openai 대신 langchain.chains.moderation 에서 임포트
from langchain.chains.moderation import OpenAIModerationChain

# 이 체인은 최신 버전에서 client 인스턴스 전달이 필요할 수 있습니다.
# 이전 코드에서 이미 OpenAI 클라이언트를 전달하고 있었으므로 이 부분은 그대로 둡니다.
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
        moderated_text = moderation_chain.run(text)
        return {"flagged": False, "text": moderated_text}
    except ValueError as e:
        print(f"[MODERATION] 콘텐츠 정책 위반 감지: {e}")
        return {"flagged": True, "text": f"입력하신 내용에 부적절한 표현이 포함되어 있어 처리할 수 없습니다. (감지된 정책: {e})"}