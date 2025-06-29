import os
import streamlit as st
import base64
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import concurrent.futures
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import io
import tempfile

# 우리가 만든 모든 서비스들을 가져옵니다.
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service, report_generator_service

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="보여DREAM | 당신의 악몽을 재구성합니다",
    page_icon="🌙",
    layout="wide"
)

# --- 2. 서비스 초기화 (캐싱으로 성능 최적화) ---
@st.cache_resource
def initialize_services():
    """ API 키 확인, 모든 서비스 및 모델 객체들을 생성하고 캐싱합니다. """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        st.stop()
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
        
        # 각 서비스 객체 생성
        stt = stt_service.STTService(api_key=api_key)
        analyzer = dream_analyzer_service.DreamAnalyzerService(api_key=api_key)
        img_gen = image_generator_service.ImageGeneratorService(api_key=api_key)
        moderator = moderation_service.ModerationService(api_key=api_key)
        report_gen = report_generator_service.ReportGeneratorService(api_key=api_key, retriever=retriever)
        
        return stt, analyzer, img_gen, moderator, report_gen
    except Exception as e:
        st.error(f"서비스 초기화 중 오류: {e}")
        st.info("'faiss_index' 폴더가 있는지, 라이브러리가 모두 설치되었는지 확인해주세요.")
        st.stop()

# --- 3. 오디오 처리 및 분석 파이프라인 함수 ---
class AudioFrameHandler(AudioProcessorBase):
    def __init__(self): self.audio_frames = []
    def recv(self, frame): self.audio_frames.append(frame.to_ndarray()); return frame
    def get_audio_bytes(self):
        if not self.audio_frames: return None
        sound_chunk = np.concatenate(self.audio_frames)
        return io.BytesIO((sound_chunk * 32767).astype(np.int16).tobytes())

def run_analysis_pipeline(text_to_analyze, services):
    """ 입력받은 텍스트로 전체 분석/생성 파이프라인을 실행하고, 결과를 세션 상태에 저장합니다. """
    _stt, _analyzer, _img_gen, _moderator, _report_gen = services
    st.session_state.analysis_results = None

    with st.spinner("RAG가 지식 베이스를 참조하여 꿈을 심층 분석 중입니다..."):
        dream_report = _report_gen.generate_report_with_rag(text_to_analyze)
        st.session_state.dream_report = dream_report
        
        nightmare_prompt = _analyzer.create_nightmare_prompt(text_to_analyze, dream_report)
        st.session_state.nightmare_prompt = nightmare_prompt
        
        reconstructed_prompt, summary, mappings = _analyzer.create_reconstructed_prompt_and_analysis(text_to_analyze, dream_report)
        st.session_state.reconstructed_prompt = reconstructed_prompt
        st.session_state.transformation_summary = summary
        st.session_state.keyword_mappings = mappings

    with st.spinner("DALL-E 3가 꿈을 이미지로 그리고 있습니다... (1분 정도 소요될 수 있습니다)"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_nightmare = executor.submit(_img_gen.generate_image_from_prompt, nightmare_prompt)
            future_reconstructed = executor.submit(_img_gen.generate_image_from_prompt, reconstructed_prompt)
            st.session_state.nightmare_image_url = future_nightmare.result()
            st.session_state.reconstructed_image_url = future_reconstructed.result()
            
    st.session_state.analysis_started = True # 분석이 완료되었음을 표시

# --- 4. 메인 앱 실행 ---
def main():
    # 서비스 초기화
    _stt, _analyzer, _img_gen, _moderator, _report_gen = initialize_services()

    # --- UI ---
    # 로고와 타이틀 표시 (수정된 버전)
    logo_path = os.path.join("user_data/image", "보여dream로고.png")
    try:
        with open(logo_path, "rb") as image_file: logo_base64 = base64.b64encode(image_file.read()).decode()
        st.markdown(f'<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;"><img src="data:image/png;base64,{logo_base64}" width="80" style="margin-right: 20px;"/><h1 style="margin: 0; white-space: nowrap;">보여dream 🌙</h1></div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.title("보여dream 🌙")
    
    st.info("당신의 악몽 이야기를 들려주세요. AI가 악몽을 분석하고 긍정적인 이미지로 재구성해 드립니다.")
    st.divider()

    # --- 세션 상태 초기화 ---
    if "dream_text" not in st.session_state: st.session_state.dream_text = ""
    if "analysis_started" not in st.session_state: st.session_state.analysis_started = False

    # --- 입력 UI ---
    # 텍스트 직접 입력
    st.session_state.dream_text = st.text_area("✍️ 여기에 꿈 내용을 직접 입력하거나, 아래 음성 입력을 통해 자동으로 채울 수 있습니다.", value=st.session_state.dream_text, height=150)

    # 음성 입력
    with st.expander("🎤 음성으로 입력하기 (파일 업로드 또는 실시간 녹음)"):
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("파일 업로드", type=['mp3', 'm4a', 'wav', 'ogg'], label_visibility="collapsed")
            if uploaded_file:
                with st.spinner("파일 변환 중..."):
                    st.session_state.dream_text = _stt.transcribe_from_bytes(uploaded_file.getvalue())
                    st.session_state.analysis_started = False # 새 입력이므로 분석 상태 초기화
                    st.rerun()
        with col2:
            wav_audio_data = st_audiorec()
            if wav_audio_data:
                with st.spinner("녹음 변환 중..."):
                    st.session_state.dream_text = _stt.transcribe_from_bytes(wav_audio_data)
                    st.session_state.analysis_started = False # 새 입력이므로 분석 상태 초기화
                    st.rerun()

    # --- 분석 실행 버튼 ---
    if st.button("✅ 꿈 분석 및 재구성 시작하기", type="primary", use_container_width=True, disabled=(not st.session_state.dream_text)):
        # 안전성 검사
        with st.spinner("입력 내용 안전성 검사 중..."):
            safety_result = _moderator.check_text_safety(st.session_state.dream_text)
        if safety_result["flagged"]:
            st.error(safety_result["text"])
            st.session_state.analysis_started = False
        else:
            st.success("안전성 검사 통과!")
            # 파이프라인 실행
            run_analysis_pipeline(st.session_state.dream_text, (_stt, _analyzer, _img_gen, _moderator, _report_gen))

    # --- 결과 표시 ---
    if st.session_state.analysis_started:
        report = st.session_state.dream_report
        st.markdown("---"); st.subheader("📊 감정 분석 리포트")
        with st.container(border=True):
            st.markdown("##### 📝 종합 분석:"); st.info(report.get("analysis_summary", ""))
            emotions = report.get("emotions", [])
            if emotions:
                st.markdown("##### 꿈 속 감정 구성:");
                for emotion in emotions:
                    score = emotion.get('score', 0); st.progress(score, text=f"{emotion.get('emotion', '알 수 없음')} - {score*100:.1f}%")
            keywords = report.get("keywords", [])
            if keywords:
                st.markdown("##### 감정 키워드:"); st.code(f"[{', '.join(keywords)}]", language="json")
        
        st.markdown("---"); st.subheader("🎨 생성된 꿈 이미지")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("###### 악몽의 시각화 (Before)")
            if st.session_state.nightmare_image_url.startswith("http"): st.image(st.session_state.nightmare_image_url, caption="악몽 시각화")
            elif st.session_state.nightmare_image_url: st.error(f"이미지 생성 실패: {st.session_state.nightmare_image_url}")
        with img_col2:
            st.markdown("###### 재구성된 꿈 (After)")
            if st.session_state.reconstructed_image_url.startswith("http"): st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈")
            elif st.session_state.reconstructed_image_url: st.error(f"이미지 생성 실패: {st.session_state.reconstructed_image_url}")

if __name__ == "__main__":
    main()