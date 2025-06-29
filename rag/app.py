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
# @st.cache_resource 데코레이터를 사용하여 서비스 객체들을 캐싱합니다.
# 이렇게 하면 Streamlit이 리런될 때마다 서비스 객체들을 다시 생성하지 않아도 됩니다.
@st.cache_resource
def initialize_services(api_key: str):
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
        
        _stt_service = stt_service.STTService(api_key=api_key)
        _dream_analyzer_service = dream_analyzer_service.DreamAnalyzerService(api_key=api_key)
        _image_generator_service = image_generator_service.ImageGeneratorService(api_key=api_key)
        _moderation_service = moderation_service.ModerationService(api_key=api_key)
        _report_generator_service = report_generator_service.ReportGeneratorService(api_key=api_key, retriever=retriever)
        
        return _stt_service, _dream_analyzer_service, _image_generator_service, _moderation_service, _report_generator_service
    except Exception as e:
        st.error(f"서비스 초기화 중 오류: {e}")
        st.info("'faiss_index' 폴더가 있는지, 라이브러리가 모두 설치되었는지 확인해주세요.")
        st.stop()


openai_api_key = os.getenv("OPENAI_API_KEY", "")
if not openai_api_key:
    st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    st.stop()

# 서비스 초기화 (캐시된 객체 사용)
_stt_service, _dream_analyzer_service, _image_generator_service, _moderation_service, _report_generator_service = initialize_services(openai_api_key)


# --- 3. 로고 이미지 로딩 및 표시 ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError: return None
    except Exception as e: st.error(f"로고 로드 오류: {e}"); return None

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
        "dream_text_input": "", # 텍스트 직접 입력을 위한 새로운 세션 상태 변수
        "dream_text": "", 
        "original_dream_text": "", 
        "analysis_started": False,
        "audio_processed": False, 
        "derisked_text": "", 
        "dream_report": None,
        "nightmare_prompt": "", 
        "reconstructed_prompt": "", 
        "transformation_summary": "",
        "keyword_mappings": [], 
        "nightmare_image_url": "", 
        "reconstructed_image_url": ""
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --- 6. 세션 상태 초기화 함수 (모든 분석 관련 상태 초기화) ---
    # 사용자가 새로운 입력 방식을 선택하거나, 새로운 입력을 시작할 때 호출됩니다.
    def initialize_analysis_state():
        st.session_state.original_dream_text = ""
        st.session_state.dream_text = ""
        st.session_state.analysis_started = False
        st.session_state.audio_processed = False
        st.session_state.derisked_text = ""
        st.session_state.dream_report = None
        st.session_state.nightmare_prompt = ""
        st.session_state.reconstructed_prompt = ""
        st.session_state.transformation_summary = ""
        st.session_state.keyword_mappings = []
        st.session_state.nightmare_image_url = ""
        st.session_state.reconstructed_image_url = ""


    # --- 7. UI 구성: 텍스트 입력 및 오디오 입력 부분 ---
    # 텍스트 직접 입력 탭을 가장 먼저 배치
    tab_text, tab_record, tab_upload = st.tabs(["✍️ 텍스트 직접 입력", "🎤 실시간 녹음하기", "📁 오디오 파일 업로드"])
    
    # 텍스트 입력 처리
    with tab_text:
        # text_input_key는 Streamlit이 위젯을 식별하는 데 사용됩니다.
        # on_change 콜백을 사용하여 텍스트 변경 시 세션 상태를 초기화합니다.
        new_text_input = st.text_area(
            "여기에 꿈 내용을 직접 입력해주세요.", 
            value=st.session_state.dream_text_input, 
            height=200,
            key="dream_text_area" # Streamlit 위젯의 고유 키
        )
        # 사용자가 텍스트를 변경했을 때만 세션 상태 초기화 및 반영
        if new_text_input != st.session_state.dream_text_input:
            st.session_state.dream_text_input = new_text_input
            initialize_analysis_state() # 새로운 텍스트 입력 시 모든 분석 상태 초기화
            st.session_state.original_dream_text = st.session_state.dream_text_input
            
            if st.session_state.original_dream_text: # 입력된 텍스트가 있으면 안전성 검사 바로 실행
                with st.spinner("입력 내용 안전성 검사 중..."):
                    safety_result = _moderation_service.check_text_safety(st.session_state.original_dream_text)
                if safety_result["flagged"]:
                    st.error(safety_result["text"])
                    st.session_state.dream_text = "" # 안전하지 않으면 dream_text를 비워 분석 방지
                else:
                    st.success("안전성 검사 통과!")
                    st.session_state.dream_text = st.session_state.original_dream_text # 안전하면 dream_text에 할당
            else: # 텍스트 필드가 비어 있으면 안전성 검사 메시지 초기화
                st.session_state.dream_text = ""
            
            st.rerun() # 변경 사항 반영을 위해 다시 실행

    # 오디오 입력 처리 (기존 로직 유지)
    audio_bytes, file_name = None, None
    with tab_record:
        wav_audio_data = st_audiorec()
        if wav_audio_data: 
            audio_bytes, file_name = wav_audio_data, "recorded_dream.wav"
            initialize_analysis_state() # 오디오 입력 시 모든 분석 상태 초기화
            st.session_state.dream_text_input = "" # 텍스트 입력 필드 비움 (다른 입력 방식 선택 시 초기화)
    with tab_upload:
        uploaded_file = st.file_uploader("악몽 오디오 파일 선택", type=["mp3", "wav", "m4a", "ogg"])
        if uploaded_file: 
            audio_bytes, file_name = uploaded_file.getvalue(), uploaded_file.name
            initialize_analysis_state() # 오디오 입력 시 모든 분석 상태 초기화
            st.session_state.dream_text_input = "" # 텍스트 입력 필드 비움 (다른 입력 방식 선택 시 초기화)

    # --- 8. 1단계: 오디오 → 텍스트 전사 + 안전성 검사 (오디오 입력이 있을 경우에만 실행) ---
    if audio_bytes and not st.session_state.audio_processed:
        temp_audio_dir = "user_data/audio"; os.makedirs(temp_audio_dir, exist_ok=True)
        try:
            # transcribe_from_bytes 메서드를 직접 호출합니다.
            with st.spinner("음성을 텍스트로 변환하고 안전성 검사 중..."):
                transcribed_text = _stt_service.transcribe_from_bytes(audio_bytes, file_name=file_name) 
                
                st.session_state.original_dream_text = transcribed_text 
                safety_result = _moderation_service.check_text_safety(transcribed_text)
                if safety_result["flagged"]:
                    st.error(safety_result["text"]); st.session_state.dream_text = "" # 안전하지 않으면 dream_text 비움
                else:
                    st.success("안전성 검사: " + safety_result["text"])
                    st.session_state.dream_text = transcribed_text # 안전하면 dream_text에 할당
                st.session_state.audio_processed = True
        except Exception as e: 
            st.error(f"음성 변환 및 안전성 검사 중 오류 발생: {e}")
            st.session_state.audio_processed = False 
            st.session_state.dream_text = ""
        st.rerun()

    # --- 9. 2단계: 전사된 텍스트 또는 직접 입력된 텍스트 출력 및 분석 시작 버튼 ---
    # original_dream_text가 채워져 있으면 (음성 변환이든 직접 입력이든)
    if st.session_state.original_dream_text: 
        st.markdown("---"); st.subheader("📝 나의 악몽 이야기") # 텍스트 변환 결과 대신 더 일반적인 제목으로 변경
        st.info(st.session_state.original_dream_text) # 원본 텍스트 표시
        
        # 실제 분석에 사용될 텍스트가 안전성 검사를 통과했을 때만 버튼 활성화
        # dream_text 세션 상태가 비어있지 않아야 (안전성 검사 통과) 버튼이 활성화됩니다.
        if st.session_state.dream_text and not st.session_state.analysis_started: 
            if st.button("✅ 이 내용으로 꿈 분석하기", type="primary", use_container_width=True):
                st.session_state.analysis_started = True; st.rerun()
        elif not st.session_state.dream_text: # dream_text가 비어있으면 (안전성 검사 실패 시) 경고
             st.warning("입력된 꿈 내용이 안전성 검사를 통과하지 못했습니다. 내용을 수정하거나 다시 시도해주세요.")
    
    # --- 10. 3단계: 리포트 생성 ---
    if st.session_state.analysis_started and st.session_state.dream_report is None:
        if st.session_state.original_dream_text: # original_dream_text를 사용하여 리포트 생성
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
                    # create_nightmare_prompt에 dream_report 인자 추가
                    prompt = _dream_analyzer_service.create_nightmare_prompt(
                        st.session_state.original_dream_text,
                        st.session_state.dream_report # <-- 여기가 수정된 부분입니다.
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