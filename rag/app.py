import streamlit as st
import os
from PIL import Image
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service, report_generator_service
from st_audiorec import st_audiorec
import base64
import tempfile

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="보여dream | 당신의 악몽을 재구성합니다", page_icon="🌙", layout="wide")

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

_stt_service, _dream_analyzer_service, _image_generator_service, _moderation_service, _report_generator_service = initialize_services(openai_api_key)

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
        st.title("보여dream 🌙")
    st.write("악몽을 녹음하거나 파일을 업로드해 주세요.")

    session_defaults = {
        "dream_text_input": "",
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

    tab_text, tab_record, tab_upload = st.tabs(["✍️ 텍스트 직접 입력", "🎤 실시간 녹음하기", "📁 오디오 파일 업로드"])
    
    with tab_text:
        new_text_input = st.text_area(
            "여기에 꿈 내용을 직접 입력해주세요.", 
            value=st.session_state.dream_text_input, 
            height=200,
            key="dream_text_area"
        )
        
        if new_text_input != st.session_state.dream_text_input:
            st.session_state.dream_text_input = new_text_input
            initialize_analysis_state() # 새로운 입력이므로 분석 상태 초기화
            st.session_state.original_dream_text = st.session_state.dream_text_input
            
            if st.session_state.original_dream_text:
                with st.spinner("입력 내용 안전성 검사 중..."):
                    safety_result = _moderation_service.check_text_safety(st.session_state.original_dream_text)
                if safety_result["flagged"]:
                    st.error(safety_result["text"])
                    st.session_state.dream_text = ""
                else:
                    st.success("안전성 검사 통과!")
                    st.session_state.dream_text = st.session_state.original_dream_text
            else:
                st.session_state.dream_text = ""
            
            st.rerun()

    with tab_record:
        wav_audio_data = st_audiorec(key="st_audiorec_widget")
        if wav_audio_data: 
            initialize_analysis_state() # 새로운 오디오 입력이므로 분석 상태 초기화
            st.session_state.dream_text_input = "" # 텍스트 입력 필드 비움
            st.session_state.audio_data_to_process = wav_audio_data # 핵심: 오디오 데이터를 세션 상태에 저장
            st.session_state.audio_file_name = "recorded_dream.wav"
            st.rerun() # 오디오 데이터가 세션에 저장되었으니 리런하여 다음 로직으로 이동

    with tab_upload:
        uploaded_file = st.file_uploader("악몽 오디오 파일 선택", type=["mp3", "wav", "m4a", "ogg"], key="file_uploader_widget")
        if uploaded_file: 
            initialize_analysis_state() # 새로운 오디오 입력이므로 분석 상태 초기화
            st.session_state.dream_text_input = "" # 텍스트 입력 필드 비움
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
        st.rerun()

    # --- 9. 2단계: 전사된 텍스트 또는 직접 입력된 텍스트 출력 및 분석 시작 버튼 ---
    if st.session_state.original_dream_text: 
        st.markdown("---"); st.subheader("📝 나의 악몽 이야기")
        st.info(st.session_state.original_dream_text)
        
        if st.session_state.dream_text and not st.session_state.analysis_started: 
            if st.button("✅ 이 내용으로 꿈 분석하기", type="primary", use_container_width=True):
                st.session_state.analysis_started = True; 
                st.rerun()
        elif not st.session_state.dream_text: 
             st.warning("입력된 꿈 내용이 안전성 검사를 통과하지 못했습니다. 내용을 수정하거나 다시 시도해주세요.")
    
    # --- 10. 3단계: 리포트 생성 ---
    if st.session_state.analysis_started and st.session_state.dream_report is None:
        if st.session_state.original_dream_text:
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
                    print("DEBUG: Generating nightmare prompt...")
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