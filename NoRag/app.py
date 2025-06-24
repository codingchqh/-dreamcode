import streamlit as st
import os
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service
from st_audiorec import st_audiorec

# --- 페이지 설정 ---
st.set_page_config(
    page_title="보여dream | 당신의 악몽을 재구성합니다",
    page_icon="🌙",
    layout="wide"
)

# --- 세션 상태 초기화 ---
def initialize_session_state():
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
    if 'audio_processed' not in st.session_state:
        st.session_state.audio_processed = False

initialize_session_state()

# --- UI 구성 ---
st.title("보여dream 🌙")
st.write("당신의 악몽을 실시간으로 녹음하거나, 오디오 파일을 업로드하여 들려주세요.")

tab1, tab2 = st.tabs(["🎤 실시간 녹음하기", "📁 오디오 파일 업로드"])

audio_bytes = None
file_name = None

with tab1:
    st.write("마이크 아이콘을 눌러 녹음을 시작/중지 하세요.")
    # [수정됨] 지원하지 않는 인자를 모두 제거하여 기본 위젯으로 호출
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        audio_bytes = wav_audio_data
        file_name = "recorded_dream.wav"

with tab2:
    st.write("가지고 있는 MP3, WAV 등의 오디오 파일을 업로드하세요.")
    uploaded_file = st.file_uploader(
        "악몽 오디오 파일을 선택하세요.",
        type=['mp3', 'wav', 'm4a', 'ogg'],
        key="dream_file_uploader"
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

# --- 통합 오디오 처리 로직 ---
if audio_bytes is not None and not st.session_state.audio_processed:
    initialize_session_state()

    audio_dir = "user_data/audio"
    audio_path = os.path.join(audio_dir, file_name)
    os.makedirs(audio_dir, exist_ok=True)

    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    with st.spinner("음성을 텍스트로 변환하고 안전성을 검사하는 중입니다... 🕵️‍♂️"):
        transcribed_text = stt_service.transcribe_audio(audio_path)
        safety_result = moderation_service.check_text_safety(transcribed_text)

        if safety_result["flagged"]:
            st.error(safety_result["text"])
            st.session_state.dream_text = ""
        else:
            st.session_state.dream_text = safety_result["text"]
    
    os.remove(audio_path)
    st.session_state.audio_processed = True

# --- 변환된 텍스트 및 이미지 생성 버튼 표시 ---
if st.session_state.dream_text:
    st.subheader("나의 악몽 이야기")
    st.write(st.session_state.dream_text)

    def reset_process_flag():
        st.session_state.audio_processed = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("😱 악몽 이미지 그대로 보기", on_click=reset_process_flag):
            with st.spinner("악몽을 시각화하는 중... 잠시만 기다려주세요."):
                nightmare_prompt = dream_analyzer_service.create_nightmare_prompt(st.session_state.dream_text)
                st.session_state.nightmare_prompt = nightmare_prompt
                nightmare_image_url = image_generator_service.generate_image_from_prompt(nightmare_prompt)
                st.session_state.nightmare_image_url = nightmare_image_url

    with col2:
        if st.button("✨ 재구성된 꿈 이미지 보기", on_click=reset_process_flag):
            with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중... 🌈"):
                reconstructed_prompt = dream_analyzer_service.create_reconstructed_prompt(st.session_state.dream_text)
                st.session_state.reconstructed_prompt = reconstructed_prompt
                reconstructed_image_url = image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                st.session_state.reconstructed_image_url = reconstructed_image_url

# --- 생성된 이미지 표시 ---
if st.session_state.get('nightmare_image_url') or st.session_state.get('reconstructed_image_url'):
    st.markdown("---")
    st.subheader("생성된 꿈 이미지")

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        if st.session_state.get('nightmare_image_url'):
            if st.session_state.nightmare_image_url.startswith("http"):
                st.image(st.session_state.nightmare_image_url, caption="악몽 시각화")
                with st.expander("생성 프롬프트 보기"):
                    st.write(st.session_state.nightmare_prompt)
            else:
                st.error(f"악몽 이미지 생성 실패: {st.session_state.nightmare_image_url}")

    with img_col2:
        if st.session_state.get('reconstructed_image_url'):
            if st.session_state.reconstructed_image_url.startswith("http"):
                st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈")
                with st.expander("생성 프롬프트 보기"):
                    st.write(st.session_state.reconstructed_prompt)
            else:
                st.error(f"재구성된 꿈 이미지 생성 실패: {st.session_state.reconstructed_image_url}")