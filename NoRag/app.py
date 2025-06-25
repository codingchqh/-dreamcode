import streamlit as st
import os
from PIL import Image
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service, report_generator_service
from st_audiorec import st_audiorec

# --- 페이지 설정 ---
st.set_page_config(
    page_title="보여dream | 당신의 악몽을 재구성합니다",
    page_icon="🌙",
    layout="wide"
)
# 로고 + 타이틀 수평 정렬
st.markdown(
    """
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <img src="Logo.png" width="80" style="margin-right: 20px;">
        <h1 style="margin: 0; font-size: 2.5rem;">보여dream 🌙</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# 로고 이미지 삽입 (파일 경로는 상대경로 또는 절대경로로)
logo_path = "C:/Users/user/Desktop/qqq/NoRag/NoRag/user_data/image/Logo.png"  # 로고 이미지가 현재 디렉토리에 있어야 함

# 로고 표시
st.image(Image.open(logo_path), width=200)  # 너비는 원하는 값으로 조정

# 제목 등 UI 구성 계속 진행

st.write("악몽을 녹음하거나 파일을 업로드해 주세요.")
# 텍스트 저장 경로
dream_text_path = "user_data/dream_text.txt"
os.makedirs("user_data", exist_ok=True)

# 텍스트 저장 함수
def save_dream_text(text):
    with open(dream_text_path, "w", encoding="utf-8") as f:
        f.write(text)

# 텍스트 불러오기 함수
def load_dream_text():
    if os.path.exists(dream_text_path):
        with open(dream_text_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

# --- 세션 상태 기본값 초기화 (앱 시작 시) ---
if "dream_text" not in st.session_state:
    st.session_state.dream_text = ""


if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False

if "derisked_text" not in st.session_state:
    st.session_state.derisked_text = ""

if "dream_report" not in st.session_state:
    st.session_state.dream_report = None

if "nightmare_prompt" not in st.session_state:
    st.session_state.nightmare_prompt = ""

if "reconstructed_prompt" not in st.session_state:
    st.session_state.reconstructed_prompt = ""

if "nightmare_image_url" not in st.session_state:
    st.session_state.nightmare_image_url = ""

if "reconstructed_image_url" not in st.session_state:
    st.session_state.reconstructed_image_url = ""

# --- 세션 상태 초기화 함수 ---
def initialize_session_state():
    # dream_text는 유지 (초기화하지 않음)
    st.session_state.derisked_text = ""
    st.session_state.dream_report = None
    st.session_state.nightmare_prompt = ""
    st.session_state.reconstructed_prompt = ""
    st.session_state.nightmare_image_url = ""
    st.session_state.reconstructed_image_url = ""
    st.session_state.audio_processed = False
    st.session_state.analysis_started = False

# --- UI 구성 ---


tab1, tab2 = st.tabs(["🎤 실시간 녹음하기", "📁 오디오 파일 업로드"])

audio_bytes = None
file_name = None

with tab1:
    st.write("녹음 버튼을 눌러 녹음하세요.")
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        audio_bytes = wav_audio_data
        file_name = "recorded_dream.wav"

with tab2:
    st.write("오디오 파일을 업로드하세요.")
    uploaded_file = st.file_uploader(
        "악몽 오디오 선택",
        type=["mp3", "wav", "m4a", "ogg"],
        key="audio_uploader"
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

# --- 1단계: 오디오 → 텍스트 전사 + 안전성 검사 ---
if audio_bytes is not None and not st.session_state.audio_processed:
    initialize_session_state()  # 상태 초기화 (dream_text 유지)
    
    audio_dir = "user_data/audio"
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, file_name)

    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    with st.spinner("음성을 텍스트로 변환하고 안전성 검사 중... 🕵️‍♂️"):
        transcribed_text = stt_service.transcribe_audio(audio_path)
        safety_result = moderation_service.check_text_safety(transcribed_text)

        if safety_result["flagged"]:
            st.error(safety_result["text"])
            # 음성 처리 실패 시, 상태 리셋 (옵션)
            st.session_state.audio_processed = False
        else:
            st.session_state.dream_text = safety_result["text"]
            st.session_state.audio_processed = True

    os.remove(audio_path)
    st.rerun()

# --- 2단계: 텍스트 출력 및 분석 시작 버튼 ---
if st.session_state.dream_text:
    st.markdown("---")
    st.subheader("📝 나의 악몽 이야기 (텍스트 변환 결과)")
    st.info(st.session_state.dream_text)

    if not st.session_state.analysis_started:
        if st.button("✅ 이 내용으로 꿈 분석하기"):
            st.session_state.analysis_started = True
            st.rerun()

# --- 3단계: 분석 시작 시 리포트 생성 ---
if st.session_state.analysis_started and st.session_state.dream_report is None:
    with st.spinner("꿈 내용을 분석하여 리포트를 생성하는 중... 🧠"):
        report = report_generator_service.generate_report(st.session_state.dream_text)
        st.session_state.dream_report = report
        st.rerun()

# --- 4단계: 리포트 출력 ---
if st.session_state.dream_report:
    report = st.session_state.dream_report
    st.markdown("---")
    st.subheader("📊 감정 분석 리포트")

    emotions = report.get("emotions", [])
    if emotions:
        st.markdown("##### 꿈 속 감정 구성:")
        for emotion in emotions:
            st.write(f"- {emotion.get('emotion', '알 수 없는 감정')}")
            score = emotion.get('score', 0)
            st.progress(score, text=f"{score}%")

    keywords = report.get("keywords", [])
    if keywords:
        st.markdown("##### 감정 키워드:")
        keywords_str = ", ".join(f'"{keyword}"' for keyword in keywords)
        st.code(f"[{keywords_str}]", language="json")

    summary = report.get("analysis_summary", "")
    if summary:
        st.markdown("##### 📝 종합 분석:")
        st.info(summary)
    
    # 이미지 생성 버튼 표시
    st.markdown("---")
    st.subheader("🎨 꿈 이미지 생성하기")
    st.write("분석 리포트를 바탕으로, 이제 꿈을 시각화해 보세요.")
    
    col1, col2 = st.columns(2)

    with col1:
        if st.button("😱 악몽 이미지 그대로 보기"):
            with st.spinner("악몽을 시각화하는 중... 잠시만 기다려주세요."):
                nightmare_prompt = dream_analyzer_service.create_nightmare_prompt(st.session_state.dream_text)
                st.session_state.nightmare_prompt = nightmare_prompt
                nightmare_image_url = image_generator_service.generate_image_from_prompt(nightmare_prompt)
                st.session_state.nightmare_image_url = nightmare_image_url
                st.rerun()

    with col2:
        if st.button("✨ 재구성된 꿈 이미지 보기"):
            with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중... 🌈"):
                reconstructed_prompt = dream_analyzer_service.create_reconstructed_prompt(st.session_state.dream_text)
                st.session_state.reconstructed_prompt = reconstructed_prompt
                reconstructed_image_url = image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                st.session_state.reconstructed_image_url = reconstructed_image_url
                st.rerun()

# --- 생성된 이미지 표시 ---
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
