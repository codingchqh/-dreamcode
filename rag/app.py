import streamlit as st
import os
from PIL import Image
# 우리가 만든 모든 서비스들을 가져옵니다.
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service, report_generator_service
from st_audiorec import st_audiorec
import base64
import tempfile 
import re # 정규표현식 모듈 추가

# --- RAG 기능을 위해 추가해야 할 임포트 ---
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# ===============================================

# --- 1. 페이지 설정 (반드시 모든 st. 명령보다 먼저 와야 합니다!) ---
st.set_page_config(
    page_title="보여dream | 당신의 악몽을 재구성합니다",
    page_icon="🌙",
    layout="wide"
)

# --- 2. API 키 로드 및 서비스 초기화 ---
openai_api_key = os.getenv("OPENAI_API_KEY", "")

if not openai_api_key:
    st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 시스템 환경 변수를 확인하거나 '.env' 파일을 설정해주세요.")
    st.stop()

# --- RAG 시스템 초기화 ---
try:
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
except Exception as e:
    st.error(f"RAG 시스템(faiss_index) 초기화 중 오류: {e}")
    st.info("프로젝트 루트 폴더에서 'python core/indexing_service.py'를 먼저 실행하여 'faiss_index' 폴더를 생성했는지 확인해주세요.")
    st.stop()

# 서비스 초기화 시 retriever 전달
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
    except FileNotFoundError:
        st.warning(f"로고 파일이 없습니다: '{image_path}'. 기본 타이틀을 사용합니다.")
        return None
    except Exception as e:
        st.error(f"로고를 로드하는 중 오류가 발생했습니다: {e}")
        return None

logo_dir = "user_data/image"
os.makedirs(logo_dir, exist_ok=True)
logo_path = os.path.join(logo_dir, "보여dream로고 투명.png")

logo_base64 = get_base64_image(logo_path)

# --- UI 중앙 정렬을 위한 컬럼 설정 ---
col_left, col_center, col_right = st.columns([1, 4, 1]) 

with col_center: # 모든 UI 요소를 이 중앙 컬럼 안에 배치합니다.
    # --- 로고 및 타이틀 표시 ---
    if logo_base64:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                <img src="data:image/png;base64,{logo_base64}" width="120" style="margin-right: 20px;"/>
                <h1 style="margin: 0; white-space: nowrap; font-size: 3em;">보여dream 🌙</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.title("보여dream 🌙")

    # --- '악몽을 녹음하거나 파일을 업로드해 주세요.' 텍스트 왼쪽에 나비몽 챗봇 이미지 배치 ---
    # 나비몽 챗봇 이미지 경로 정의
    navimong_chatbot_image_path = os.path.join("user_data/image", "나비몽 챗봇.png")
    navimong_chatbot_image_exists = os.path.exists(navimong_chatbot_image_path)

    col_chatbot_img, col_text = st.columns([0.15, 0.85]) 
    
    with col_chatbot_img:
        if navimong_chatbot_image_exists:
            # ===> 나비몽 챗봇 이미지 크기 150으로 변경 <===
            st.image(navimong_chatbot_image_path, width=150) 
    
    with col_text:
        st.markdown("<h3 style='margin-top: 15px; margin-left: 0px;'>악몽을 녹음하거나 파일을 업로드해 주세요.</h3>", unsafe_allow_html=True)


    st.markdown("---") # 구분선

    # --- 5. 세션 상태 기본값 초기화 ---
    session_defaults = {
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
        "reconstructed_image_url": "",
        "nightmare_keywords": [], 
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --- 6. 세션 상태 초기화 함수 ---
    def initialize_session_state():
        for key, value in session_defaults.items():
            st.session_state[key] = value

    # --- 7. UI 구성: 오디오 입력 부분 ---
    tab1, tab2 = st.tabs(["🎤 실시간 녹음하기", "📁 오디오 파일 업로드"])

    audio_bytes = None
    file_name = None

    with tab1:
        st.write("녹음 버튼을 눌러 악몽을 이야기해 주세요.")
        wav_audio_data = st_audiorec()
        if wav_audio_data is not None:
            audio_bytes = wav_audio_data
            file_name = "recorded_dream.wav"

    with tab2:
        st.write("또는 오디오 파일을 직접 업로드할 수도 있습니다.")
        uploaded_file = st.file_uploader(
            "악몽 오디오 파일 선택",
            type=["mp3", "wav", "m4a", "ogg"],
            key="audio_uploader"
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

    # --- 8. 1단계: 오디오 → 텍스트 전사 (STT) + 안전성 검사 ---
    if audio_bytes is not None and not st.session_state.audio_processed:
        initialize_session_state() 
        
        temp_audio_dir = "user_data/audio"
        os.makedirs(temp_audio_dir, exist_ok=True)

        audio_path = None

        try:
            suffix = os.path.splitext(file_name)[1] if file_name else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, dir=temp_audio_dir, suffix=suffix) as temp_file:
                temp_file.write(audio_bytes)
                audio_path = temp_file.name
            
            if not audio_path or not os.path.exists(audio_path):
                st.error("임시 오디오 파일 생성에 실패했습니다.")
                st.session_state.audio_processed = False
                st.rerun()

            with st.spinner("음성을 텍스트로 변환하고 안전성 검사 중... 🕵️‍♂️"):
                transcribed_text = _stt_service.transcribe_audio(audio_path)
                
                st.session_state.original_dream_text = transcribed_text 

                safety_result = _moderation_service.check_text_safety(transcribed_text)

                if safety_result["flagged"]:
                    st.error(safety_result["text"])
                    st.session_state.audio_processed = False
                    st.session_state.dream_text = ""
                else:
                    st.session_state.dream_text = transcribed_text
                    st.success("안전성 검사: " + safety_result["text"])
                    st.session_state.audio_processed = True

        except Exception as e:
            st.error(f"오디오 처리 중 예상치 못한 오류가 발생했습니다: {e}")
            st.session_state.audio_processed = False
            st.session_state.dream_text = ""
            print(f"ERROR during audio processing: {e}")
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"DEBUG: 임시 오디오 파일 성공적으로 삭제됨: {audio_path}")
                except Exception as e:
                    print(f"WARNING: 임시 오디오 파일 '{audio_path}' 삭제 실패: {e}")
            elif audio_path:
                print(f"DEBUG: 임시 오디오 파일 '{audio_path}'은 이미 존재하지 않아 삭제를 건너뛰었습니다.")
        
        st.rerun() # UI 갱신을 위해 재실행

    # --- 9. 2단계: 전사된 텍스트 출력 및 분석 시작 버튼 ---
    if st.session_state.original_dream_text:
        st.markdown("---")
        st.subheader("📝 나의 악몽 이야기 (텍스트 변환 결과)")
        st.info(st.session_state.original_dream_text)

        if st.session_state.dream_text and not st.session_state.analysis_started:
            if st.button("✅ 이 내용으로 꿈 분석하기"):
                st.session_state.analysis_started = True
                st.rerun()
        elif not st.session_state.dream_text and st.session_state.audio_processed:
            st.warning("입력된 꿈 내용이 안전성 검사를 통과하지 못했습니다. 분석을 진행할 수 없습니다.")


    # --- 10. 3단계: 리포트 생성 ---
    if st.session_state.analysis_started and st.session_state.dream_report is None:
        if st.session_state.original_dream_text:
            with st.spinner("RAG가 지식 베이스를 참조하여 리포트를 생성하는 중... 🧠"):
                # ReportGeneratorService는 RAG를 사용하므로, original_dream_text를 기반으로 분석합니다.
                report = _report_generator_service.generate_report_with_rag(st.session_state.original_dream_text)
                st.session_state.dream_report = report
                st.session_state.nightmare_keywords = report.get("keywords", []) # 리포트에서 키워드 저장
                st.rerun()
        else:
            st.error("분석할 꿈 텍스트가 없습니다. 다시 시도해주세요.")
            st.session_state.analysis_started = False


    # --- 11. 4단계: 감정 분석 리포트 출력 및 이미지 생성 버튼 ---
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
                st.progress(score, text=f"{score*100:.1f}%")

        keywords = report.get("keywords", [])
        if keywords:
            st.markdown("##### 감정 키워드:")
            # ===> 변경된 부분: st.code 대신 st.markdown으로 키워드에 색상 적용 <===
            keywords_str_list = [f'<span style="color: red; font-weight: bold;">{keyword}</span>' for keyword in keywords]
            keywords_html = f"[{', '.join(keywords_str_list)}]"
            st.markdown(keywords_html, unsafe_allow_html=True)
            # =========================================================

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
                with st.spinner("악몽을 시각화하는 중... 잠시만 기다려주세요."):
                    # _dream_analyzer_service.create_nightmare_prompt 함수 호출 시 인자 추가 (수정됨)
                    prompt = _dream_analyzer_service.create_nightmare_prompt(
                        st.session_state.original_dream_text,
                        st.session_state.dream_report
                    )
                    st.session_state.nightmare_prompt = prompt
                    nightmare_image_url = _image_generator_service.generate_image_from_prompt(prompt)
                    st.session_state.nightmare_image_url = nightmare_image_url
                    st.rerun()

        with col2:
            if st.button("✨ 재구성된 꿈 이미지 보기"):
                with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중... 🌈"):
                    reconstructed_prompt, transformation_summary, keyword_mappings = \
                        _dream_analyzer_service.create_reconstructed_prompt_and_analysis(
                            st.session_state.original_dream_text,
                            st.session_state.dream_report
                        )
                    st.session_state.reconstructed_prompt = reconstructed_prompt
                    st.session_state.transformation_summary = transformation_summary
                    st.session_state.keyword_mappings = keyword_mappings
                    
                    reconstructed_image_url = _image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                    st.session_state.reconstructed_image_url = reconstructed_image_url
                    st.rerun()

    # --- 12. 5단계: 생성된 이미지 표시 및 키워드 강조 ---
    # 키워드에 색상을 입히는 헬퍼 함수
    # 이 함수는 정규 표현식 look-behind 오류를 피하기 위해 HTML 태그와 일반 텍스트를 분리하여 처리합니다.
    def highlight_keywords(text, keywords, color="red"):
        # 키워드를 길이 역순으로 정렬하여 긴 키워드가 먼저 매치되도록 합니다.
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        # HTML 태그를 분리하기 위한 정규식 (시작/끝 태그가 있는 경우와 단일 태그)
        html_tag_splitter = re.compile(r'(?s)(<[^>]+>.*?<\/[^>]+>|<[^>]+\/>)')
        
        # 텍스트를 HTML 태그 부분과 일반 텍스트 부분으로 분리합니다.
        segments = html_tag_splitter.split(text)
        
        processed_parts = []
        for i, segment in enumerate(segments):
            if i % 2 == 0: # 짝수 인덱스(0, 2, 4...)는 일반 텍스트 부분
                current_text_segment = segment
                for keyword in sorted_keywords:
                    if not keyword.strip(): # 비어있는 키워드는 건너뜁니다.
                        continue
                    
                    # 단어 경계 및 대소문자 무시
                    # re.escape는 키워드 내의 특수문자를 처리합니다.
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    
                    # re.sub를 사용하여 해당 키워드에 강조 태그 삽입
                    # 이미 강조된 키워드를 다시 강조하지 않는 로직은 여기서 생략됩니다.
                    # (성능과 복잡성 때문에 단순화)
                    current_text_segment = re.sub(pattern, f"<span style='color:{color}; font-weight:bold;'>{keyword}</span>", current_text_segment, flags=re.IGNORECASE)
                processed_parts.append(current_text_segment)
            else: # 홀수 인덱스(1, 3, 5...)는 HTML 태그 부분 (그대로 유지)
                processed_parts.append(segment)
                
        # 분리된 부분을 다시 합쳐서 최종 결과 문자열을 만듭니다.
        return "".join(processed_parts)


    if (st.session_state.nightmare_image_url and st.session_state.nightmare_image_url.startswith("http")) or \
       (st.session_state.reconstructed_image_url and st.session_state.reconstructed_image_url.startswith("http")):
        st.markdown("---"); st.subheader("생성된 꿈 이미지")
        img_col1, img_col2 = st.columns(2)

        with img_col1:
            if st.session_state.nightmare_image_url:
                if st.session_state.nightmare_image_url.startswith("http"):
                    st.image(st.session_state.nightmare_image_url, caption="악몽 시각화")
                    with st.expander("생성 프롬프트 및 주요 키워드 보기"):
                        # --- 악몽 프롬프트 키워드 강조 적용 ---
                        all_nightmare_keywords_for_highlight = st.session_state.nightmare_keywords
                        
                        highlighted_nightmare_prompt = highlight_keywords(st.session_state.nightmare_prompt, all_nightmare_keywords_for_highlight, "red")
                        st.markdown(f"**프롬프트:** {highlighted_nightmare_prompt}", unsafe_allow_html=True)
                        
                        if all_nightmare_keywords_for_highlight:
                            st.markdown("---")
                            # 강조된 키워드 리스트를 직접 출력
                            highlighted_list = [f"<span style='color:red; font-weight:bold;'>{k}</span>" for k in all_nightmare_keywords_for_highlight]
                            st.markdown(f"**주요 키워드:** {', '.join(highlighted_list)}", unsafe_allow_html=True)
                else:
                    st.error(f"악몽 이미지 생성 실패: {st.session_state.nightmare_image_url}")
        
        with img_col2:
            if st.session_state.reconstructed_image_url:
                if st.session_state.reconstructed_image_url.startswith("http"):
                    st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈")
                    with st.expander("생성 프롬프트 및 변환 과정 보기"):
                        # --- 재구성 프롬프트 키워드 강조 적용 ---
                        transformed_only_keywords_from_mapping = [mapping.get('transformed', '') for mapping in st.session_state.keyword_mappings if mapping.get('transformed')]
                        all_reconstructed_keywords_for_highlight = transformed_only_keywords_from_mapping

                        highlighted_reconstructed_prompt = highlight_keywords(st.session_state.reconstructed_prompt, all_reconstructed_keywords_for_highlight, "green")
                        
                        st.markdown(f"**프롬프트:** {highlighted_reconstructed_prompt}", unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("**변환 요약:**")
                        st.write(st.session_state.transformation_summary)
                        
                        # 변환된 키워드 목록을 '원본(빨간색) → 변환(초록색)' 형식으로 표시
                        if st.session_state.keyword_mappings:
                            transformed_keywords_display_list = []
                            for mapping in st.session_state.keyword_mappings:
                                original = mapping.get('original', 'N/A')
                                transformed = mapping.get('transformed', 'N/A')
                                transformed_keywords_display_list.append(f"<span style='color:red;'>{original}</span> → <span style='color:green;'>{transformed}</span>")
                            
                            st.markdown("---")
                            st.markdown(f"**변환된 키워드:** {', '.join(transformed_keywords_display_list)}", unsafe_allow_html=True)
                elif st.session_state.reconstructed_image_url:
                    st.error(f"재구성 이미지 생성 실패: {st.session_state.reconstructed_image_url}")