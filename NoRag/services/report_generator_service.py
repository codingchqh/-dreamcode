# services/report_generator_service.py (JSON 모드 사용 최종본)

import json
from openai import OpenAI
from core.config import API_KEY

# OpenAI 클라이언트를 직접 사용합니다.
client = OpenAI(api_key=API_KEY)

# LLM에게 역할을 부여하고, JSON 형식으로만 답변하도록 지시하는 시스템 프롬프트
# pydantic의 format_instructions 대신, 직접 JSON 구조를 설명해줍니다.
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
  "analysis_summary": "꿈의 전반적인 심리적 경향에 대한 한 문장 요약"
}
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
            # [핵심] response_format을 json_object로 설정하여 JSON 출력 강제
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": dream_text}
            ]
        )
        
        # API가 보장하는 유효한 JSON 문자열을 딕셔너리로 변환
        report_data = json.loads(response.choices[0].message.content)
        print("[DEBUG] 감정 분석 리포트 생성 완료.")
        return report_data

    except Exception as e:
        print(f"리포트 생성 중 최종 오류 발생: {e}")
        # 오류 발생 시, 정해진 형식의 빈 딕셔너리 반환
        return {"emotions": [], "keywords": [], "analysis_summary": "리포트를 생성하는 데 실패했습니다. 잠시 후 다시 시도해주세요."}