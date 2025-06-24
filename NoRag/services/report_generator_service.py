# services/report_generator_service.py (종합 분석 기능 강화)

import json
from openai import OpenAI
from core.config import API_KEY

client = OpenAI(api_key=API_KEY)

# [최종 수정] 종합 분석(analysis_summary)에 대한 지시를 훨씬 더 구체적으로 변경
SYSTEM_PROMPT = """
You are an expert dream analyst with a deep understanding of psychology. 
Your task is to analyze the user's dream description and create a structured emotion report.
Your output MUST be a valid JSON object. Do not add any text before or after the JSON object.

The JSON object must conform to the following structure:
{
  "emotions": [
    {"emotion": "감정이름(예: 불안)", "score": 0-100 사이의 정수},
    {"emotion": "감정이름(예: 무력감)", "score": 0-100 사이의 정수}
  ],
  "keywords": ["꿈의 핵심 키워드1", "핵심 키워드2"],
  "analysis_summary": "A detailed one or two-sentence analysis connecting the identified emotions and keywords to a potential psychological tendency. **You MUST cite specific events or keywords from the dream as evidence for your analysis.**"
}

Here are examples of a good vs. a bad 'analysis_summary':
- GOOD (specific and grounded): "'선임의 질책'과 '차가운 바닥'과 같은 키워드는 현실의 압박감과 무력감을 반영하며, 이는 스트레스 상황에 대한 회피적 경향을 나타낼 수 있습니다."
- BAD (generic and unhelpful): "불안과 공포가 높게 나타납니다."

All text values in the JSON should be in Korean.
"""

def generate_report(dream_text: str) -> dict:
    """
    꿈 텍스트를 분석하여 구조화된 감정 리포트를 생성합니다. (JSON 모드 사용)
    """
    try:
        print("[DEBUG] 감정 분석 리포트 생성 시작 (JSON 모드)...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": dream_text}
            ]
        )
        
        report_data = json.loads(response.choices[0].message.content)
        print("[DEBUG] 감정 분석 리포트 생성 완료.")
        return report_data

    except Exception as e:
        print(f"리포트 생성 중 최종 오류 발생: {e}")
        return {"emotions": [], "keywords": [], "analysis_summary": "리포트를 생성하는 데 실패했습니다. 잠시 후 다시 시도해주세요."}