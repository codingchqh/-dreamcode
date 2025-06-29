import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import concurrent.futures
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import io

# 우리가 만든 모든 서비스들을 가져옵니다.
from services.dream_analyzer_service import DreamAnalyzerService
from services.report_generator_service import ReportGeneratorService
from services.image_generator_service import ImageGeneratorService
from services.stt_service import STTService

# --- 1. 페이지 설정 및 서비스 초기화 ---
st.set_page_config(page_title="보여DREAM", page_icon="🌙", layout="wide")

@st.cache_resource
def initialize_services():
    """ API 키 확인, 모든 서비스 및 모델 객체들을 생성하고 캐싱합니다. """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다."); st.stop()
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
        report_generator = ReportGeneratorService(api_key=api_key, retriever=retriever)
        dream_analyzer = DreamAnalyzerService(api_key=api_key)
        image_generator = ImageGeneratorService(api_key=api_key)
        stt_service = STTService(api_key=api_key)
        return report_generator, dream_analyzer, image_generator, stt_service
    except Exception as e:
        st.error(f"서비스 초기화 중 오류: {e}"); st.info("faiss_index 폴더를 확인해주세요."); st.stop()

# --- 2. 실시간 오디오 녹음 처리 클래스 ---
class AudioFrameHandler(AudioProcessorBase):
    def __init__(self): self.audio_frames = []
    def recv(self, frame): self.audio_frames.append(frame.to_ndarray()); return frame
    def get_audio_bytes(self):
        if not self.audio_frames: return None
        sound_chunk = np.concatenate(self.audio_frames)
        return io.BytesIO((sound_chunk * 32767).astype(np.int16).tobytes())

# --- 3. 분석 및 결과 표시를 위한 공통 함수 ---
def run_analysis_pipeline(dream_text):
    """ 입력받은 텍스트로 전체 분석/생성 파이프라인을 실행하고, 결과를 st.session_state에 저장합니다. """
    if not dream_text or "오류" in dream_text or "찾을 수 없습니다" in dream_text:
        st.error(dream_text or "분석할 텍스트가 없습니다."); return
    st.session_state.analysis_results = None
    with st.spinner("RAG가 지식 베이스를 참조하여 꿈을 심층 분석 중입니다..."):
        dream_report = st.session_state.report_generator.generate_report_with_rag(dream_text)
        nightmare_prompt, reconstructed_prompt, summary, mappings = st.session_state.dream_analyzer.create_reconstructed_prompt_and_analysis(dream_text, dream_report)
    with st.spinner("DALL-E 3가 꿈을 이미지로 그리고 있습니다... (1분 정도 소요될 수 있습니다)"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_nightmare = executor.submit(st.session_state.image_generator.generate_image_from_prompt, nightmare_prompt)
            future_reconstructed = executor.submit(st.session_state.image_generator.generate_image_from_prompt, reconstructed_prompt)
            nightmare_image_url = future_nightmare.result()
            reconstructed_image_url = future_reconstructed.result()
    st.session_state.analysis_results = {
        "dream_report": dream_report,
        "nightmare_image_url": nightmare_image_url,
        "reconstructed_image_url": reconstructed_image_url,
        "summary": summary,
        "mappings": mappings
    }

def display_results():
    """ st.session_state에 저장된 분석 결과를 화면에 표시합니다. """
    results = st.session_state.analysis_results
    dream_report = results["dream_report"]
    st.subheader("📝 AI 심층 분석 리포트")
    with st.container(border=True):
        st.markdown("##### 심층 분석 요약"); st.write(dream_report.get("analysis_summary", "요약 정보 없음"))
        st.markdown("##### 주요 감정")
        for emo in dream_report.get("emotions", []): st.progress(emo['score'], text=f"{emo['emotion']} ({int(emo['score']*100)}%)")
        st.markdown("##### 핵심 키워드"); st.write(" &nbsp; ".join(f"`{kw}`" for kw in dream_report.get("keywords", [])))
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("악몽의 시각화 (Before)");
        if results["nightmare_image_url"].startswith("http"): st.image(results["nightmare_image_url"], caption="AI가 그린 당신의 악몽")
        else: st.error(f"이미지 생성 실패: {results['nightmare_image_url']}")
    with col2:
        st.subheader("재구성된 꿈 (After)");
        if results["reconstructed_image_url"].startswith("http"): st.image(results["reconstructed_image_url"], caption="AI가 긍정적으로 재구성한 꿈")
        else: st.error(f"이미지 생성 실패: {results['reconstructed_image_url']}")
    st.divider()
    st.subheader("✨ 이렇게 바뀌었어요!"); st.write(results["summary"])
    for mapping in results["mappings"]: st.markdown(f"- `{mapping['original']}` &nbsp; ➡️ &nbsp; **`{mapping['transformed']}`**")

# --- 4. 메인 앱 실행 ---
def main():
    st.title("보여DREAM 🌙")
    st.write("당신의 꿈 이야기를 들려주세요. AI가 악몽을 분석하고 긍정적인 이미지로 재구성해 드립니다.")
    
    st.session_state.report_generator, st.session_state.dream_analyzer, st.session_state.image_generator, st.session_state.stt_service = initialize_services()

    if "transcribed_text" not in st.session_state: st.session_state.transcribed_text = ""
    if "analysis_results" not in st.session_state: st.session_state.analysis_results = None

    tab1, tab2, tab3 = st.tabs(["✍️ 텍스트로 입력", "⬆️ 파일 업로드", "🎤 실시간 녹음"])
    with tab1:
        dream_text_input = st.text_area("어젯밤 어떤 꿈을 꾸셨나요?", height=200, key="text_input")
        if st.button("분석 시작 (텍스트)", type="primary", use_container_width=True, key="analyze_text"):
            run_analysis_pipeline(dream_text_input)
    with tab2:
        uploaded_file = st.file_uploader("음성 파일(mp3, wav, m4a 등)을 업로드하세요.", type=['mp3', 'm4a', 'wav', 'ogg'], key="file_uploader")
        if uploaded_file and st.button("이 파일로 텍스트 변환하기", use_container_width=True, key="transcribe_file"):
            with st.spinner("음성 파일을 텍스트로 변환 중입니다..."):
                st.session_state.transcribed_text = st.session_state.stt_service.transcribe_from_bytes(uploaded_file.getvalue())
        st.text_area("**변환된 텍스트**", value=st.session_state.transcribed_text, height=150, key="transcribed_text_file")
        if st.session_state.transcribed_text and st.button("분석 시작 (업로드 파일)", type="primary", use_container_width=True, key="analyze_file"):
            run_analysis_pipeline(st.session_state.transcribed_text)
    with tab3:
        st.write("아래 'START' 버튼을 누르고 마이크에 꿈 이야기를 녹음하세요.")
        webrtc_ctx = webrtc_streamer(key="audio-recorder", mode=WebRtcMode.SENDONLY, audio_processor_factory=AudioFrameHandler)
        if webrtc_ctx.audio_processor and st.button("녹음 내용 텍스트로 변환", use_container_width=True, key="transcribe_mic"):
            audio_bytes_io = webrtc_ctx.audio_processor.get_audio_bytes()
            if audio_bytes_io:
                with st.spinner("녹음된 오디오를 텍스트로 변환합니다..."):
                    st.session_state.transcribed_text = st.session_state.stt_service.transcribe_from_bytes(audio_bytes_io.getvalue())
            else: st.warning("녹음된 내용이 없습니다.")
        st.text_area("**변환된 텍스트**", value=st.session_state.transcribed_text, height=150, key="transcribed_text_mic")
        if st.session_state.transcribed_text and st.button("분석 시작 (녹음 내용)", type="primary", use_container_width=True, key="analyze_mic"):
            run_analysis_pipeline(st.session_state.transcribed_text)

    if st.session_state.analysis_results:
        display_results()

if __name__ == "__main__":
    main()