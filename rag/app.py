import streamlit as st
import os
from PIL import Image
# 우리가 만든 모든 서비스들을 가져옵니다.
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service, report_generator_service
from st_audiorec import st_audiorec
import base64
import tempfile

# --- RAG 기능을 위해 추가해야 할 임포트 ---
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# ===============================================

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="보여dream | 당신의 악몽을 재구성합니다", page_icon="🌙", layout="wide")

# --- 2. API 키 로드 및 서비스 초기화 ---
openai_api_key = os.getenv("OPENAI_API_KEY", "")
if not openai_api_key:
    st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    st.stop()

try:
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
except Exception as e:
    st.error(f"RAG 시스템(faiss_index) 초기화 중 오류: {e}")
    st.info("프로젝트 루트 폴더에서 'python core/indexing_service.py'를 먼저 실행하여 'faiss_index' 폴더를 생성했는지 확인해주세요.")
    st.stop()

_stt_service = stt_service.STTService(api_key=openai_api_key)
_dream_analyzer_service = dream_analyzer_service.DreamAnalyzerService(api_key=openai_api_key)
_image_generator_service = image_generator_service.ImageGeneratorService(api_key=openai_api_key)
_moderation_service = moderation_service.ModerationService(api_key=openai_api_key)
_report_generator_service = report_generator_service.ReportGeneratorService(api_key=openai_api_key, retriever=retriever)

# --- 3. 로고 이미지 로딩 및 표시 ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError: return None
    except Exception as e: st.error(f"로고 로드 오류: {e}"); return None

# '보여dream로고.png'가 배경이 투명한 로고라면 더 좋습니다.
logo_path = os.path.join("user_data/image", "보여dream로고.png") 
logo_base64 = get_base64_image(logo_path)

col_left, col_center, col_right = st.columns([1, 4, 1]) 
with col_center:
    # --- 수정된 로고 및 타이틀 표시 부분 ---
    if logo_base64:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{logo_base64}" width="80" style="margin-right: 15px;"/>
                <h1 style="margin: 0; white-space: nowrap; font-size: 3em;">보여dream 🌙</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.title("보여dream 🌙") # 로고 로드 실패 시 대체 타이틀
    st.write("악몽을 녹음하거나 파일을 업로드해 주세요.")

    # --- 5. 세션 상태 기본값 초기화 ---
    session_defaults = {
        "dream_text": "", "original_dream_text": "", "analysis_started": False,
        "audio_processed": False, "derisked_text": "", "dream_report": None,
        "nightmare_prompt": "", "reconstructed_prompt": "", "transformation_summary": "",
        "keyword_mappings": [], "nightmare_image_url": "", "reconstructed_image_url": ""
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[:key] = value

    # --- 6. 세션 상태 초기화 함수 ---
    def initialize_session_state():
        for key, value in session_defaults.items():
            st.session_state[:key] = value

    # --- 7. UI 구성: 오디오 입력 부분 ---
    tab1, tab2 = st.tabs(["🎤 실시간 녹음하기", "📁 오디오 파일 업로드"])
    audio_bytes, file_name = None, None
    with tab1:
        wav_audio_data = st_audiorec()
        if wav_audio_data: audio_bytes, file_name = wav_audio_data, "recorded_dream.wav"
    with tab2:
        uploaded_file = st.file_uploader("악몽 오디오 파일 선택", type=["mp3", "wav", "m4a", "ogg"])
        if uploaded_file: audio_bytes, file_name = uploaded_file.getvalue(), uploaded_file.name

    # --- 8. 1단계: 오디오 → 텍스트 전사 + 안전성 검사 ---
    if audio_bytes and not st.session_state.audio_processed:
        initialize_session_state()
        temp_audio_dir = "user_data/audio"; os.makedirs(temp_audio_dir, exist_ok=True)
        audio_path = None
        try:
            suffix = os.path.splitext(file_name)[1] if file_name else ".wav"
            
            with st.spinner("음성을 텍스트로 변환하고 안전성 검사 중..."):
                transcribed_text = _stt_service.transcribe_from_bytes(audio_bytes, file_name=file_name) 
                
                st.session_state.original_dream_text = transcribed_text 
                safety_result = _moderation_service.check_text_safety(transcribed_text)
                if safety_result["flagged"]:
                    st.error(safety_result["text"]); st.session_state.dream_text = ""
                else:
                    st.session_state.dream_text = transcribed_text; st.success("안전성 검사: " + safety_result["text"])
                st.session_state.audio_processed = True
        except Exception as e: 
            st.error(f"음성 변환 및 안전성 검사 중 오류 발생: {e}")
            st.session_state.audio_processed = False 
            st.session_state.dream_text = ""
        st.rerun()

    # --- 9. 2단계: 전사된 텍스트 출력 및 분석 시작 버튼 ---
    if st.session_state.original_dream_text: 
        st.markdown("---"); st.subheader("📝 나의 악몽 이야기 (텍스트 변환 결과)")
        st.info(st.session_state.original_dream_text)
        if st.session_state.dream_text and not st.session_state.analysis_started: 
            if st.button("✅ 이 내용으로 꿈 분석하기"):
                st.session_state.analysis_started = True; st.rerun()
        elif not st.session_state.dream_text and st.session_state.audio_processed:
            st.warning("입력된 꿈 내용이 안전성 검사를 통과하지 못했습니다.")

    # --- 10. 3단계: 리포트 생성 ---
    if st.session_state.analysis_started and st.session_state.dream_report is None:
        if st.session_state.original_dream_text:
            with st.spinner("RAG가 지식 베이스를 참조하여 리포트를 생성하는 중... 🧠"):
                report = _report_generator_service.generate_report_with_rag(st.session_state.original_dream_text)
                st.session_state.dream_report = report
                st.rerun()
        else:
            st.error("분석할 꿈 텍스트가 없습니다."); st.session_state.analysis_started = False
    
    # --- 11. 4단계: 감정 분석 리포트 출력 및 이미지 생성 버튼 ---
    if st.session_state.dream_report:
        report = st.session_state.dream_report
        st.markdown("---"); st.subheader("📊 감정 분석 리포트")
        emotions = report.get("emotions", [])
        if emotions:
            st.markdown("##### 꿈 속 감정 구성:");
            for emotion in emotions:
                score = emotion.get('score', 0); st.progress(score, text=f"{emotion.get('emotion', '알 수 없음')} - {score*100:.1f}%")
        keywords = report.get("keywords", [])
        if keywords:
            st.markdown("##### 감정 키워드:"); st.code(f"[{', '.join(keywords)}]", language="json")
        summary = report.get("analysis_summary", "")
        if summary:
            st.markdown("##### 📝 종합 분석:"); st.info(summary)
        st.markdown("---"); st.subheader("🎨 꿈 이미지 생성하기")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("😱 악몽 이미지 그대로 보기"):
                with st.spinner("악몽을 시각화하는 중..."):
                    # dream_report 인자를 추가하여 전달합니다.
                    prompt = _dream_analyzer_service.create_nightmare_prompt(
                        st.session_state.original_dream_text,
                        st.session_state.dream_report
                    )
                    st.session_state.nightmare_prompt = prompt
                    st.session_state.nightmare_image_url = _image_generator_service.generate_image_from_prompt(prompt)
                    st.rerun() 
        with col2:
            if st.button("✨ 재구성된 꿈 이미지 보기"):
                with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중..."):
                    reconstructed_prompt, transformation_summary, keyword_mappings = \
                        _dream_analyzer_service.create_reconstructed_prompt_and_analysis(
                            st.session_state.original_dream_text, 
                            st.session_state.dream_report
                        )
                    st.session_state.reconstructed_prompt = reconstructed_prompt
                    st.session_state.transformation_summary = transformation_summary
                    st.session_state.keyword_mappings = keyword_mappings
                    st.session_state.reconstructed_image_url = _image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                    st.rerun()

    # --- 12. 5단계: 생성된 이미지 표시 ---
    # `st.session_state.nightmare_image_url` 또는 `st.session_state.reconstructed_image_url`이 비어있지 않거나 HTTP로 시작하는 경우에만 표시
    if (st.session_state.nightmare_image_url and st.session_state.nightmare_image_url.startswith("http")) or \
       (st.session_state.reconstructed_image_url and st.session_state.reconstructed_image_url.startswith("http")):
        st.markdown("---"); st.subheader("생성된 꿈 이미지")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            if st.session_state.nightmare_image_url.startswith("http"):
                st.image(st.session_state.nightmare_image_url, caption="악몽 시각화")
                with st.expander("생성 프롬프트 보기"): st.write(st.session_state.nightmare_prompt)
            elif st.session_state.nightmare_image_url:
                st.error(f"악몽 이미지 생성 실패: {st.session_state.nightmare_image_url}")
        with img_col2:
            if st.session_state.reconstructed_image_url.startswith("http"):
                st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈")
                with st.expander("생성 프롬프트 및 변환 과정 보기"): 
                    st.write(f"**프롬프트:** {st.session_state.reconstructed_prompt}")
                    st.markdown("**변환 요약:**")
                    st.write(st.session_state.transformation_summary)
            elif st.session_state.reconstructed_image_url:
                st.error(f"재구성 이미지 생성 실패: {st.session_state.reconstructed_image_url}")