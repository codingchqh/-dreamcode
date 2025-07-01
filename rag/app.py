import streamlit as st
import os
from PIL import Image
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service, report_generator_service
from st_audiorec import st_audiorec
import base64
import tempfile
import re

# RAG(Retrieval-Augmented Generation) 기능을 위한 임포트
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. 페이지 설정 (가장 먼저 실행되어야 함) ---
st.set_page_config(
    page_title="보여dream | 당신의 악몽을 재구성합니다",
    page_icon="🌙",
    layout="wide"
)

# --- 2. API 키 로드 및 서비스 초기화 ---
@st.cache_resource
def initialize_services():
    """ API 키 확인, 모든 서비스 및 모델 객체들을 생성하고 캐싱합니다. """
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        st.stop()
    try:
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
        
        _stt = stt_service.STTService(api_key=openai_api_key)
        _analyzer = dream_analyzer_service.DreamAnalyzerService(api_key=openai_api_key)
        _img_gen = image_generator_service.ImageGeneratorService(api_key=openai_api_key)
        _moderator = moderation_service.ModerationService(api_key=openai_api_key)
        _report_gen = report_generator_service.ReportGeneratorService(api_key=openai_api_key, retriever=retriever)
        return _stt, _analyzer, _img_gen, _moderator, _report_gen
    except Exception as e:
        st.error(f"RAG 시스템(faiss_index) 초기화 중 오류: {e}")
        st.info("프로젝트 루트 폴더에서 'python core/indexing_service.py'를 실행하여 'faiss_index' 폴더를 생성했는지 확인해주세요.")
        st.stop()

# --- 3. 헬퍼 함수 정의 ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

def translate_image_error_message(raw_error_message: str) -> str:
    if "image_generation_user_error" in raw_error_message or "safety system" in raw_error_message:
        return "감정적으로 불편함을 줄 수 있는 요소는 자동 필터링되어, 안전하고 편안한 이미지로 구성되었습니다."
    if "APIConnectionError" in raw_error_message or "500" in raw_error_message:
        return "OpenAI 서버에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
    if "AuthenticationError" in raw_error_message:
        return "OpenAI API 키가 유효하지 않습니다."
    return "알 수 없는 오류로 이미지 생성에 실패했습니다."

def highlight_keywords(text: str, keywords: list, color: str = "red") -> str:
    if not keywords or not text: return text
    sorted_keywords = sorted(list(set(keywords)), key=len, reverse=True)
    highlighted_text = text
    for keyword in sorted_keywords:
        if not keyword.strip(): continue
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_text = pattern.sub(f"<span style='color:{color}; font-weight:bold;'>{keyword}</span>", highlighted_text)
    return highlighted_text

# --- 4. 메인 애플리케이션 실행 ---

# 서비스 초기화
_stt, _analyzer, _img_gen, _moderator, _report_gen = initialize_services()

# UI 중앙 정렬 컬럼
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    # 로고 및 타이틀
    logo_path = os.path.join("user_data/image", "보여dream로고 투명.png")
    logo_base64 = get_base64_image(logo_path)
    if logo_base64:
        st.markdown(f"""<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;"><img src="data:image/png;base64,{logo_base64}" width="120" style="margin-right: 20px;"/><h1 style="margin: 0; white-space: nowrap; font-size: 3em;">보여dream 🌙</h1></div>""", unsafe_allow_html=True)
    else:
        st.title("보여dream 🌙")

    # 챗봇 이미지와 안내문구
    navimong_path = os.path.join("user_data/image", "나비몽 챗봇.png")
    if os.path.exists(navimong_path):
        img_col, txt_col = st.columns([0.15, 0.85])
        with img_col: st.image(navimong_path, width=150)
        with txt_col: st.markdown("<h3 style='margin-top: 15px;'>악몽을 녹음하거나 파일을 업로드해 주세요.</h3>", unsafe_allow_html=True)
    else:
        st.write("악몽을 녹음하거나 파일을 업로드해 주세요.")
    st.markdown("---")

    # 세션 상태 초기화 로직
    session_defaults = {"dream_text": "", "original_dream_text": "", "analysis_started": False, "audio_processed": False, "dream_report": None, "nightmare_prompt": "", "reconstructed_prompt": "", "transformation_summary": "", "keyword_mappings": [], "nightmare_image_url": "", "reconstructed_image_url": "", "nightmare_keywords": []}
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    def initialize_session_state():
        for key, value in session_defaults.items():
            st.session_state[key] = value

    # 오디오 입력 UI
    tab1, tab2 = st.tabs(["🎤 실시간 녹음하기", "📁 오디오 파일 업로드"])
    audio_bytes, file_name = None, None
    with tab1:
        wav_audio_data = st_audiorec()
        if wav_audio_data:
            audio_bytes, file_name = wav_audio_data, "recorded_dream.wav"
    with tab2:
        uploaded_file = st.file_uploader("악몽 오디오 파일 선택", type=["mp3", "wav", "m4a", "ogg"])
        if uploaded_file:
            audio_bytes, file_name = uploaded_file.getvalue(), uploaded_file.name

    # 오디오 처리 로직
    if audio_bytes and not st.session_state.audio_processed:
        initialize_session_state()
        audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_bytes)
                audio_path = temp_file.name
            with st.spinner("음성을 텍스트로 변환하고 안전성 검사 중..."):
                transcribed_text = _stt.transcribe_audio(audio_path)
                st.session_state.original_dream_text = transcribed_text
                safety_result = _moderator.check_text_safety(transcribed_text)
                if not safety_result["flagged"]:
                    st.session_state.dream_text = transcribed_text
                    st.success("안전성 검사: " + safety_result["text"])
                else:
                    st.error(safety_result["text"])
                    st.session_state.dream_text = ""
                st.session_state.audio_processed = True
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        st.rerun()

    # 분석 시작 로직
    if st.session_state.original_dream_text:
        st.markdown("---")
        st.subheader("📝 나의 악몽 이야기 (텍스트 변환 결과)")
        st.info(st.session_state.original_dream_text)
        if st.session_state.dream_text and not st.session_state.analysis_started:
            if st.button("✅ 이 내용으로 꿈 분석하기"):
                st.session_state.analysis_started = True
                st.rerun()
        elif not st.session_state.dream_text and st.session_state.audio_processed:
            st.warning("입력된 꿈 내용이 안전성 검사를 통과하지 못했습니다.")

    # 리포트 생성 로직
    if st.session_state.analysis_started and st.session_state.dream_report is None:
        if st.session_state.original_dream_text:
            with st.spinner("RAG가 지식 베이스를 참조하여 리포트를 생성하는 중... 🧠"):
                report = _report_gen.generate_report_with_rag(st.session_state.original_dream_text)
                st.session_state.dream_report = report
                st.session_state.nightmare_keywords = report.get("keywords", [])
                st.rerun()
        else:
            st.error("분석할 꿈 텍스트가 없습니다.")
            st.session_state.analysis_started = False

    # 결과 표시 로직
    if st.session_state.dream_report:
        report = st.session_state.dream_report
        st.markdown("---")
        st.subheader("📊 감정 분석 리포트")
        emotions = report.get("emotions", [])
        if emotions:
            st.markdown("##### 꿈 속 감정 구성:")
            for emotion in emotions:
                score = emotion.get('score', 0)
                st.progress(score, text=f"{emotion.get('emotion', '알 수 없음')} - {score*100:.1f}%")
        
        keywords = report.get("keywords", [])
        if keywords:
            st.markdown("##### 감정 키워드:")
            # SyntaxError를 해결한 부분
            keywords_html = ", ".join([f"<span style='color: red; font-weight: bold;'>{keyword}</span>" for keyword in keywords])
            st.markdown(f"[{keywords_html}]", unsafe_allow_html=True)

        summary = report.get("analysis_summary", "")
        if summary:
            st.markdown("##### 📝 종합 분석:")
            st.info(summary)
        
        st.markdown("---")
        st.subheader("🎨 꿈 이미지 생성하기")
        st.write("분석 리포트를 바탕으로, 이제 꿈을 시각화해 보세요. 어떤 이미지를 먼저 보시겠어요?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("😱 악몽 이미지 그대로 보기"):
                with st.spinner("악몽을 시각화하는 중..."):
                    prompt = _analyzer.create_nightmare_prompt(st.session_state.original_dream_text, report)
                    st.session_state.nightmare_prompt = prompt
                    st.session_state.nightmare_image_url = _img_gen.generate_image_from_prompt(prompt)
                    st.rerun()
        with col2:
            if st.button("✨ 재구성된 꿈 이미지 보기"):
                with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중..."):
                    prompt, summary, mappings = _analyzer.create_reconstructed_prompt_and_analysis(st.session_state.original_dream_text, report)
                    st.session_state.reconstructed_prompt, st.session_state.transformation_summary, st.session_state.keyword_mappings = prompt, summary, mappings
                    st.session_state.reconstructed_image_url = _img_gen.generate_image_from_prompt(prompt)
                    st.rerun()

        # 생성된 이미지 표시
        if st.session_state.nightmare_image_url or st.session_state.reconstructed_image_url:
            st.markdown("---")
            st.subheader("생성된 꿈 이미지")
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                if st.session_state.nightmare_image_url:
                    if st.session_state.nightmare_image_url.startswith("http"):
                        st.image(st.session_state.nightmare_image_url, caption="악몽 시각화")
                        with st.expander("생성 프롬프트 및 주요 키워드 보기"):
                            all_nightmare_keywords_for_highlight = st.session_state.nightmare_keywords
                            highlighted_nightmare_prompt = highlight_keywords(st.session_state.nightmare_prompt, all_nightmare_keywords_for_highlight, "red")
                            st.markdown(f"**프롬프트:** {highlighted_nightmare_prompt}", unsafe_allow_html=True)
                            if all_nightmare_keywords_for_highlight:
                                st.markdown("---")
                                highlighted_list = [f"<span style='color:red; font-weight:bold;'>{k}</span>" for k in all_nightmare_keywords_for_highlight]
                                st.markdown(f"**주요 키워드:** {', '.join(highlighted_list)}", unsafe_allow_html=True)
                    else:
                        friendly_error = translate_image_error_message(st.session_state.nightmare_image_url)
                        st.error(friendly_error)
                        with st.expander("자세한 기술 오류 정보 보기"): st.code(st.session_state.nightmare_image_url)
            with img_col2:
                if st.session_state.reconstructed_image_url:
                    if st.session_state.reconstructed_image_url.startswith("http"):
                        st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈")
                        with st.expander("생성 프롬프트 및 변환 과정 보기"):
                            transformed_keywords = [m.get('transformed', '') for m in st.session_state.keyword_mappings]
                            highlighted_prompt = highlight_keywords(st.session_state.reconstructed_prompt, transformed_keywords, "green")
                            st.markdown(f"**프롬프트:** {highlighted_prompt}", unsafe_allow_html=True)
                            st.markdown("---"); st.markdown("**변환 요약:**"); st.write(st.session_state.transformation_summary)
                            if st.session_state.keyword_mappings:
                                st.markdown("---"); st.markdown(f"**변환된 키워드:**")
                                transformed_display = [f"<span style='color:red;'>{m.get('original')}</span> → <span style='color:green;'>{m.get('transformed')}</span>" for m in st.session_state.keyword_mappings]
                                st.markdown(', '.join(transformed_display), unsafe_allow_html=True)
                    else:
                        friendly_error = translate_image_error_message(st.session_state.reconstructed_image_url)
                        st.error(friendly_error)
                        with st.expander("자세한 기술 오류 정보 보기"): st.code(st.session_state.reconstructed_image_url)