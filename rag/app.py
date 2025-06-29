# app.py (궁극의 디버깅 버전)

import os
import streamlit as st
import time # 디버깅을 위해 추가

# ... (다른 import 구문들은 이전과 동일)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import concurrent.futures
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import io
from services.dream_analyzer_service import DreamAnalyzerService
from services.report_generator_service import ReportGeneratorService
from services.image_generator_service import ImageGeneratorService
from services.stt_service import STTService

# --- 서비스 초기화 및 오디오 핸들러 (이전과 동일) ---
st.set_page_config(page_title="보여DREAM", page_icon="🌙", layout="wide")
@st.cache_resource
def initialize_services():
    # ... (이전과 동일)
    api_key = os.getenv("OPENAI_API_KEY");
    if not api_key: st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다."); st.stop()
    try:
        embeddings = OpenAIEmbeddings(); vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True); retriever = vector_store.as_retriever()
        report_generator = ReportGeneratorService(api_key=api_key, retriever=retriever); dream_analyzer = DreamAnalyzerService(api_key=api_key); image_generator = ImageGeneratorService(api_key=api_key); stt_service = STTService(api_key=api_key)
        return report_generator, dream_analyzer, image_generator, stt_service
    except Exception as e:
        st.error(f"서비스 초기화 중 오류: {e}"); st.info("faiss_index 폴더를 확인해주세요."); st.stop()
class AudioFrameHandler(AudioProcessorBase):
    def __init__(self): self.audio_frames = []
    def recv(self, frame): self.audio_frames.append(frame.to_ndarray()); return frame
    def get_audio_bytes(self):
        if not self.audio_frames: return None
        sound_chunk = np.concatenate(self.audio_frames); return io.BytesIO((sound_chunk * 32767).astype(np.int16).tobytes())

# --- 분석 및 결과 표시 함수 (디버깅 print 추가) ---
def run_analysis_pipeline(dream_text):
    print("DEBUG: 1. run_analysis_pipeline 함수 시작됨.")
    if not dream_text or "오류" in dream_text or "찾을 수 없습니다" in dream_text:
        st.error(dream_text or "분석할 텍스트가 없습니다."); print("DEBUG: ERROR! 분석할 텍스트 없음."); return
    st.session_state.analysis_results = None
    try:
        with st.spinner("RAG가 지식 베이스를 참조하여 꿈을 심층 분석 중입니다..."):
            print("DEBUG: 2. 리포트 생성 서비스 호출 시작...")
            dream_report = st.session_state.report_generator.generate_report_with_rag(dream_text)
            print("DEBUG: 3. 리포트 생성 완료.")
            nightmare_prompt, reconstructed_prompt, summary, mappings = st.session_state.dream_analyzer.create_reconstructed_prompt_and_analysis(dream_text, dream_report)
            print("DEBUG: 4. 프롬프트 생성 완료.")
        with st.spinner("DALL-E 3가 꿈을 이미지로 그리고 있습니다..."):
            print("DEBUG: 5. 이미지 생성 서비스 호출 시작...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_nightmare = executor.submit(st.session_state.image_generator.generate_image_from_prompt, nightmare_prompt)
                future_reconstructed = executor.submit(st.session_state.image_generator.generate_image_from_prompt, reconstructed_prompt)
                nightmare_image_url = future_nightmare.result(); reconstructed_image_url = future_reconstructed.result()
            print("DEBUG: 6. 이미지 생성 완료.")
        print("DEBUG: 7. 세션 상태에 모든 결과 저장 시작...")
        st.session_state.analysis_results = { "dream_report": dream_report, "nightmare_image_url": nightmare_image_url, "reconstructed_image_url": reconstructed_image_url, "summary": summary, "mappings": mappings }
        print("DEBUG: 8. 세션 상태에 결과 저장 완료.")
    except Exception as e:
        print(f"DEBUG: ERROR! 파이프라인 실행 중 심각한 오류 발생: {e}"); st.error(f"분석 파이프라인 실행 중 오류가 발생했습니다: {e}")

def display_results():
    print("DEBUG: 10. display_results 함수 시작됨.")
    # ... (이전과 동일)
    results = st.session_state.analysis_results; dream_report = results["dream_report"]
    st.subheader("📝 AI 심층 분석 리포트")
    with st.container(border=True):
        st.markdown("##### 심층 분석 요약"); st.write(dream_report.get("analysis_summary", "요약 정보 없음"))
        st.markdown("##### 주요 감정");
        for emo in dream_report.get("emotions", []): st.progress(emo['score'], text=f"{emo['emotion']} ({int(emo['score']*100)}%)")
        st.markdown("##### 핵심 키워드"); st.write(" &nbsp; ".join(f"`{kw}`" for kw in dream_report.get("keywords", [])))
    st.divider()
    col1, col2 = st.columns(2);
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
    print("DEBUG: 11. display_results 함수 완료.")

# --- 메인 앱 실행 함수 (디버깅 print 및 st.rerun() 추가) ---
def main():
    print(f"\n--- SCRIPT RERUN AT {time.time()} ---")
    st.title("보여DREAM 🌙")
    st.session_state.report_generator, st.session_state.dream_analyzer, st.session_state.image_generator, st.session_state.stt_service = initialize_services()
    if "transcribed_text" not in st.session_state: st.session_state.transcribed_text = ""
    if "analysis_results" not in st.session_state: st.session_state.analysis_results = None
    def handle_file_upload():
        if st.session_state.file_uploader:
            with st.spinner("..."):
                st.session_state.transcribed_text = st.session_state.stt_service.transcribe_from_bytes(st.session_state.file_uploader.getvalue())
                st.session_state.analysis_results = None
    tab1, tab2, tab3 = st.tabs(["✍️ 텍스트로 입력", "⬆️ 파일 업로드", "🎤 실시간 녹음"])
    with tab1:
        text_input = st.text_area("...", key="text_input_area")
        if st.button("분석 시작 (텍스트)", key="analyze_text"):
            print(">>> '분석 시작 (텍스트)' 버튼 클릭됨"); run_analysis_pipeline(text_input); st.rerun()
    with tab2:
        st.file_uploader("...", key="file_uploader", on_change=handle_file_upload)
        if st.session_state.transcribed_text:
            st.text_area("...", value=st.session_state.transcribed_text, key="transcribed_text_area")
            if st.button("분석 시작 (업로드 파일)", key="analyze_file"):
                print(">>> '분석 시작 (업로드 파일)' 버튼 클릭됨"); run_analysis_pipeline(st.session_state.transcribed_text); st.rerun()
    with tab3:
        webrtc_ctx = webrtc_streamer(...)
        if webrtc_ctx.audio_processor and st.button("녹음 완료 및 텍스트 변환", key="transcribe_mic"):
            # ... (이전과 동일)
            audio_bytes_io = webrtc_ctx.audio_processor.get_audio_bytes()
            if audio_bytes_io:
                with st.spinner("..."):
                    st.session_state.transcribed_text = st.session_state.stt_service.transcribe_from_bytes(audio_bytes_io.getvalue())
                    st.session_state.analysis_results = None
            else: st.warning("녹음된 내용이 없습니다.")
        if st.session_state.transcribed_text:
            st.text_area("...", value=st.session_state.transcribed_text, key="mic_text_area")
            if st.button("분석 시작 (녹음 내용)", key="analyze_mic"):
                print(">>> '분석 시작 (녹음 내용)' 버튼 클릭됨"); run_analysis_pipeline(st.session_state.transcribed_text); st.rerun()

    print(f"--- Final Check (analysis_results is None: {st.session_state.analysis_results is None}) ---")
    if st.session_state.analysis_results:
        display_results()
    print("--- SCRIPT END ---")

if __name__ == "__main__":
    main()