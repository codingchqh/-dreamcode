import streamlit as st
import os
# services 임포트 부분에 moderation_service 추가
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import List
from openai import OpenAI
from core.config import API_KEY
# --- 페이지 설정 ---
st.set_page_config(
    page_title="보여dream | 당신의 악몽을 재구성합니다",
    page_icon="�",
    layout="wide"
)

# --- 세션 상태 초기화 ---
if 'dream_text' not in st.session_state:
    st.session_state.dream_text = ""
if 'nightmare_prompt' not in st.session_state:
    st.session_state.nightmare_prompt = ""
if 'reconstructed_prompt' not in st.session_state:
    st.session_state.reconstructed_prompt = ""
if 'nightmare_image_url' not in st.session_state:
    st.session_state.nightmare_image_url = ""
if 'reconstructed_image_url' not in st.session_state:
    st.session_state.reconstructed_image_url = ""

# --- UI 구성 ---
st.title("보여dream 🌙")
st.write("당신의 악몽을 음성으로 들려주세요. 긍정적인 꿈으로 재구성하여 보여드립니다.")

# 1. 음성 파일 업로드
uploaded_file = st.file_uploader(
    "여기에 악몽 음성 파일을 업로드하세요 (MP3, WAV, M4A 등)",
    type=['mp3', 'wav', 'm4a', 'ogg']
)

if uploaded_file is not None:
    # 2. 음성 -> 텍스트 변환 및 안전성 검사
    if st.session_state.dream_text == "":
        audio_dir = "user_data/audio"
        audio_path = os.path.join(audio_dir, uploaded_file.name)
        os.makedirs(audio_dir, exist_ok=True)

        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("음성을 텍스트로 변환하고 안전성을 검사하는 중입니다... 🕵️‍♂️"):
            # 2-1. 음성을 텍스트로 변환
            transcribed_text = stt_service.transcribe_audio(audio_path)
            
            # [NEW] 2-2. 변환된 텍스트의 안전성 검사
            safety_result = moderation_service.check_text_safety(transcribed_text)

            # 2-3. 검사 결과에 따라 처리
            if safety_result["flagged"]:
                # 문제가 있으면 에러 메시지를 표시하고 중단
                st.error(safety_result["text"])
                st.session_state.dream_text = "" # 다음 단계로 진행되지 않도록 초기화
            else:
                # 문제가 없으면 텍스트를 저장하고 계속 진행
                st.session_state.dream_text = safety_result["text"]
        
        os.remove(audio_path)

# 3. 변환된 텍스트와 선택 버튼 표시 (안전성 검사를 통과한 경우에만)
if st.session_state.dream_text:
    st.subheader("나의 악몽 이야기")
    st.write(st.session_state.dream_text)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("😱 악몽 이미지 그대로 보기"):
            with st.spinner("악몽을 시각화하는 중... 잠시만 기다려주세요."):
                nightmare_prompt = dream_analyzer_service.create_nightmare_prompt(st.session_state.dream_text)
                st.session_state.nightmare_prompt = nightmare_prompt
                nightmare_image_url = image_generator_service.generate_image_from_prompt(nightmare_prompt)
                st.session_state.nightmare_image_url = nightmare_image_url

    with col2:
        if st.button("✨ 재구성된 꿈 이미지 보기"):
            with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중... 🌈"):
                reconstructed_prompt = dream_analyzer_service.create_reconstructed_prompt(st.session_state.dream_text)
                st.session_state.reconstructed_prompt = reconstructed_prompt
                reconstructed_image_url = image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                st.session_state.reconstructed_image_url = reconstructed_image_url

# 4. 생성된 이미지 표시 (오류 처리 로직 포함)
if st.session_state.nightmare_image_url or st.session_state.reconstructed_image_url:
    st.markdown("---")
    st.subheader("생성된 꿈 이미지")

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        if st.session_state.nightmare_image_url:
            if st.session_state.nightmare_image_url.startswith("http"):
                st.image(st.session_state.nightmare_image_url, caption="악몽 시각화")
                with st.expander("생성 프롬프트 보기"):
                    st.write(st.session_state.nightmare_prompt)
            else:
                st.error(f"악몽 이미지 생성 실패: {st.session_state.nightmare_image_url}")

    with img_col2:
        if st.session_state.reconstructed_image_url:
            if st.session_state.reconstructed_image_url.startswith("http"):
                st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈")
                with st.expander("생성 프롬프트 보기"):
                    st.write(st.session_state.reconstructed_prompt)
            else:
                st.error(f"재구성된 꿈 이미지 생성 실패: {st.session_state.reconstructed_image_url}")
