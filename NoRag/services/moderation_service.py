from openai import OpenAI
from core.config import API_KEY
from langchain.chains.moderation import OpenAIModerationChain

moderation_chain = OpenAIModerationChain(client=OpenAI(api_key=API_KEY))

def check_text_safety(text: str) -> dict:
    try:
        moderated_text = moderation_chain.run(text)
        return {"flagged": False, "text": moderated_text}
    except ValueError as e:
        print(f"[MODERATION] 콘텐츠 정책 위반 감지: {e}")
        return {"flagged": True, "text": f"입력하신 내용에 부적절한 표현이 포함되어 있어 처리할 수 없습니다. (감지된 정책: {e})"}