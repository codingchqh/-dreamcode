import streamlit as st
import os
from PIL import Image
# 우리가 만든 모든 서비스들을 가져옵니다.
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service, report_generator_service
from st_audiorec import st_audiorec # st_audiorec 임포트
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
        # allow_dangerous_deserialization=True는 보안에 주의해야 합니다.
        # 신뢰할 수 있는 소스에서 생성된 FAISS 인덱스만 로드해야 합니다.
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
        st.info("'faiss_index' 폴더가 있는지, 라이브러리가 모두 설치되었는지 확인해주세요. 'python core/indexing_service.py'를 먼저 실행해야 할 수도 있습니다.")
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
        # "dream_text_input": "", # 텍스트 직접 입력을 위한 변수 제거
        "dream_text": "", 
        "original_dream_text": "", 
        "analysis_started": False,
        "audio_processed": False, # 오디오 처리 완료 플래그 (STT 실행 여부)
        "audio_data_to_process": None, # 처리할 오디오 바이트 데이터
        "audio_file_name": None, # 처리할 오디오 파일 이름
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
        print("DEBUG: Initializing analysis state...")
        # 오디오 관련 상태도 같이 초기화
        st.session_state.original_dream_text = ""
        st.session_state.dream_text = ""
        st.session_state.analysis_started = False
        st.session_state.audio_processed = False
        st.session_state.audio_data_to_process = None # 핵심 수정: 처리할 오디오 데이터 초기화
        st.session_state.audio_file_name = None
        st.session_state.derisked_text = ""
        st.session_state.dream_report = None
        st.session_state.nightmare_prompt = ""
        st.session_state.reconstructed_prompt = ""
        st.session_state.transformation_summary = ""
        st.session_state.keyword_mappings = []
        st.session_state.nightmare_image_url = ""
        st.session_state.reconstructed_image_url = ""

    # --- 7. UI 구성: 오디오 입력 부분 ---
    # 텍스트 직접 입력 탭 제거, 오디오 입력 탭만 남김
    tab_record, tab_upload = st.tabs(["🎤 실시간 녹음하기", "📁 오디오 파일 업로드"])
    
    # 오디오 입력 처리 탭들
    audio_bytes_from_input = None # 이 변수는 현재 프레임에서 받은 오디오 데이터를 임시로 저장
    
    with tab_record:
        # key 인자 제거 (st_audiorec()는 key를 지원하지 않을 수 있음)
        wav_audio_data = st_audiorec() 
        if wav_audio_data: 
            initialize_analysis_state() # 새로운 오디오 입력이므로 분석 상태 초기화
            st.session_state.audio_data_to_process = wav_audio_data # 핵심: 오디오 데이터를 세션 상태에 저장
            st.session_state.audio_file_name = "recorded_dream.wav"
            st.rerun() # 오디오 데이터가 세션에 저장되었으니 리런하여 다음 로직으로 이동

    with tab_upload:
        uploaded_file = st.file_uploader("악몽 오디오 파일 선택", type=["mp3", "wav", "m4a", "ogg"], key="file_uploader_widget") # key는 Streamlit 내장 위젯에 유효
        if uploaded_file: 
            initialize_analysis_state() # 새로운 오디오 입력이므로 분석 상태 초기화
            st.session_state.audio_data_to_process = uploaded_file.getvalue() # 핵심: 오디오 데이터를 세션 상태에 저장
            st.session_state.audio_file_name = uploaded_file.name
            st.rerun() # 오디오 데이터가 세션에 저장되었으니 리런하여 다음 로직으로 이동

    # --- 8. 1단계: 오디오 → 텍스트 전사 + 안전성 검사 (audio_data_to_process가 있을 경우에만 실행) ---
    # `audio_data_to_process`가 있고 아직 처리되지 않았다면 STT 실행
    if st.session_state.audio_data_to_process is not None and not st.session_state.audio_processed:
        try:
            with st.spinner("음성을 텍스트로 변환하고 안전성 검사 중..."):
                print("DEBUG: Starting audio transcription and safety check from session state...")
                transcribed_text = _stt_service.transcribe_from_bytes(
                    st.session_state.audio_data_to_process, 
                    file_name=st.session_state.audio_file_name
                ) 
                
                st.session_state.original_dream_text = transcribed_text 
                safety_result = _moderation_service.check_text_safety(transcribed_text)
                if safety_result["flagged"]:
                    st.error(safety_result["text"]); st.session_state.dream_text = "" 
                else:
                    st.success("안전성 검사: " + safety_result["text"])
                    st.session_state.dream_text = transcribed_text 
                st.session_state.audio_processed = True
                st.session_state.audio_data_to_process = None # 핵심 수정: 처리 완료 후 데이터 비움
                st.session_state.audio_file_name = None # 핵심 수정: 파일 이름 비움
                print("DEBUG: Audio processing complete. Rerunning...")
        except Exception as e: 
            st.error(f"음성 변환 및 안전성 검사 중 오류 발생: {e}")
            st.session_state.audio_processed = False 
            st.session_state.dream_text = ""
            st.session_state.audio_data_to_process = None # 오류 발생 시에도 데이터 비움
            st.session_state.audio_file_name = None
            print(f"ERROR: Audio processing failed: {e}")
        st.rerun() # STT 처리 완료 (또는 실패) 후 UI 업데이트를 위해 리런

    # --- 9. 2단계: 전사된 텍스트 출력 및 분석 시작 버튼 ---
    if st.session_state.original_dream_text: 
        st.markdown("---"); st.subheader("📝 나의 악몽 이야기") # 텍스트 변환 결과 대신 더 일반적인 제목으로 변경
        st.info(st.session_state.original_dream_text) # 원본 텍스트 표시
        
        # 실제 분석에 사용될 텍스트가 안전성 검사를 통과했을 때만 버튼 활성화
        # dream_text 세션 상태가 비어있지 않아야 (안전성 검사 통과) 버튼이 활성화됩니다.
        if st.session_state.dream_text and not st.session_state.analysis_started: 
            if st.button("✅ 이 내용으로 꿈 분석하기", type="primary", use_container_width=True):
                st.session_state.analysis_started = True; 
                st.rerun() # 분석 시작 버튼 클릭 시 리런하여 다음 단계로 진행
        elif not st.session_state.dream_text: # dream_text가 비어있으면 (안전성 검사 실패 시) 경고
             st.warning("입력된 꿈 내용이 안전성 검사를 통과하지 못했습니다. 내용을 수정하거나 다시 시도해주세요.")
    
    # --- 10. 3단계: 리포트 생성 ---
    # analysis_started가 True이고 dream_report가 아직 생성되지 않았을 때만 실행
    if st.session_state.analysis_started and st.session_state.dream_report is None:
        if st.session_state.original_dream_text: # original_dream_text를 사용하여 리포트 생성
            with st.spinner("RAG가 지식 베이스를 참조하여 리포트를 생성하는 중... 🧠"):
                print("DEBUG: Starting report generation...")
                report = _report_generator_service.generate_report_with_rag(st.session_state.original_dream_text)
                st.session_state.dream_report = report
                print("DEBUG: Report generated. Rerunning...")
                st.rerun()
        else:
            st.error("분석할 꿈 텍스트가 없습니다."); st.session_state.analysis_started = False
            print("ERROR: No dream text to analyze for report.")
    
    # --- 11. 4단계: 감정 분석 리포트 출력 및 이미지 생성 버튼 ---
    if st.session_state.dream_report: # dream_report가 있어야 이 섹션이 표시됩니다.
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
                    print("DEBUG: Generating nightmare prompt...")
                    # create_nightmare_prompt에 dream_report 인자 추가
                    prompt = _dream_analyzer_service.create_nightmare_prompt(
                        st.session_state.original_dream_text,
                        st.session_state.dream_report 
                    )
                    st.session_state.nightmare_prompt = prompt
                    print("DEBUG: Generating nightmare image...")
                    st.session_state.nightmare_image_url = _image_generator_service.generate_image_from_prompt(prompt)
                    print("DEBUG: Nightmare image generated. Rerunning...")
                    st.rerun() 
        with col2:
            if st.button("✨ 재구성된 꿈 이미지 보기"):
                with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중..."):
                    print("DEBUG: Generating reconstructed prompt and analysis...")
                    reconstructed_prompt, transformation_summary, keyword_mappings = \
                        _dream_analyzer_service.create_reconstructed_prompt_and_analysis(
                            st.session_state.original_dream_text, 
                            st.session_state.dream_report
                        )
                    st.session_state.reconstructed_prompt = reconstructed_prompt
                    st.session_state.transformation_summary = transformation_summary
                    st.session_state.keyword_mappings = keyword_mappings
                    print("DEBUG: Generating reconstructed image...")
                    st.session_state.reconstructed_image_url = _image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                    print("DEBUG: Reconstructed image generated. Rerunning...")
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