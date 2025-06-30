import streamlit as st # Streamlit 라이브러리 임포트 (웹 앱 구축용)
import os # 운영체제와 상호작용하는 기능 (파일 경로 등) 제공
from PIL import Image # Pillow 라이브러리 임포트 (이미지 처리용)
# 개발한 서비스 모듈들 임포트
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service, report_generator_service
from st_audiorec import st_audiorec # Streamlit 오디오 녹음 위젯
import base64 # Base64 인코딩/디코딩 모듈
import tempfile # 임시 파일 생성을 위한 모듈
import re # 정규표현식 모듈

# RAG(Retrieval-Augmented Generation) 기능을 위한 임포트
from langchain_openai import OpenAIEmbeddings # OpenAI 임베딩 모델
from langchain_community.vectorstores import FAISS # FAISS 벡터 스토어
# ===============================================

# --- 1. 페이지 설정 (가장 먼저 실행되어야 함) ---
st.set_page_config(
    page_title="보여dream | 당신의 악몽을 재구성합니다", # 웹 페이지 제목
    page_icon="🌙", # 웹 페이지 아이콘
    layout="wide" # 넓은 레이아웃 사용
)

# --- 2. API 키 로드 및 서비스 초기화 ---
openai_api_key = os.getenv("OPENAI_API_KEY", "") # 환경 변수에서 OpenAI API 키 가져오기

if not openai_api_key:
    st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 시스템 환경 변수를 확인하거나 '.env' 파일을 설정해주세요.")
    st.stop() # API 키가 없으면 앱 실행 중지

# RAG 시스템 초기화
try:
    embeddings = OpenAIEmbeddings(api_key=openai_api_key) # OpenAI 임베딩 객체 생성
    # 로컬에 저장된 FAISS 벡터 스토어 로드
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever() # 벡터 스토어를 검색기(retriever)로 사용
except Exception as e:
    st.error(f"RAG 시스템(faiss_index) 초기화 중 오류: {e}")
    st.info("프로젝트 루트 폴더에서 'python core/indexing_service.py'를 먼저 실행하여 'faiss_index' 폴더를 생성했는지 확인해주세요.")
    st.stop() # RAG 초기화 실패 시 앱 실행 중지

# 서비스 초기화 (초기화 시 retriever 객체 전달)
_stt_service = stt_service.STTService(api_key=openai_api_key) # 음성-텍스트 변환 서비스
_dream_analyzer_service = dream_analyzer_service.DreamAnalyzerService(api_key=openai_api_key) # 꿈 분석 서비스
_image_generator_service = image_generator_service.ImageGeneratorService(api_key=openai_api_key) # 이미지 생성 서비스
_moderation_service = moderation_service.ModerationService(api_key=openai_api_key) # 콘텐츠 검열 서비스
_report_generator_service = report_generator_service.ReportGeneratorService(api_key=openai_api_key, retriever=retriever) # 리포트 생성 서비스 (RAG 포함)

# --- 3. 로고 이미지 로딩 및 표시 ---
# 이미지를 Base64로 인코딩하여 웹에 표시할 수 있도록 하는 함수
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

logo_dir = "user_data/image" # 로고 이미지 디렉토리 설정
os.makedirs(logo_dir, exist_ok=True) # 디렉토리가 없으면 생성
logo_path = os.path.join(logo_dir, "보여dream로고 투명.png") # 로고 파일 경로

logo_base64 = get_base64_image(logo_path) # 로고 이미지를 Base64로 인코딩

# --- UI 중앙 정렬을 위한 컬럼 설정 ---
col_left, col_center, col_right = st.columns([1, 4, 1]) # 좌, 중앙, 우 3개 컬럼 생성 (비율 1:4:1)

with col_center: # 모든 UI 요소를 이 중앙 컬럼 안에 배치
    # --- 로고 및 타이틀 표시 ---
    if logo_base64:
        # Base64 인코딩된 이미지를 HTML 마크다운으로 표시 (중앙 정렬)
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
        st.title("보여dream 🌙") # 로고가 없으면 기본 타이틀 표시

    # --- '악몽을 녹음하거나 파일을 업로드해 주세요.' 텍스트 왼쪽에 나비몽 챗봇 이미지 배치 ---
    navimong_chatbot_image_path = os.path.join("user_data/image", "나비몽 챗봇.png") # 나비몽 챗봇 이미지 경로
    navimong_chatbot_image_exists = os.path.exists(navimong_chatbot_image_path) # 이미지 파일 존재 여부 확인

    col_chatbot_img, col_text = st.columns([0.15, 0.85]) # 이미지와 텍스트를 위한 2개 컬럼 생성

    with col_chatbot_img:
        if navimong_chatbot_image_exists:
            st.image(navimong_chatbot_image_path, width=150) # 나비몽 챗봇 이미지 표시 (크기 150)
    
    with col_text:
        st.markdown("<h3 style='margin-top: 15px; margin-left: 0px;'>악몽을 녹음하거나 파일을 업로드해 주세요.</h3>", unsafe_allow_html=True)

    st.markdown("---") # 구분선 표시

    # --- 5. 세션 상태 기본값 초기화 ---
    # Streamlit 세션 상태 변수들의 기본값 정의
    session_defaults = {
        "dream_text": "", # STT 변환 후 안전성 검사를 통과한 꿈 텍스트
        "original_dream_text": "", # STT 변환된 원본 꿈 텍스트 (안전성 검사 전)
        "analysis_started": False, # 분석 시작 여부 플래그
        "audio_processed": False, # 오디오 처리 완료 여부 플래그
        "derisked_text": "", # (현재 사용되지 않음, 이전 버전 흔적)
        "dream_report": None, # 꿈 분석 리포트 결과
        "nightmare_prompt": "", # 악몽 이미지 생성 프롬프트
        "reconstructed_prompt": "", # 재구성된 꿈 이미지 생성 프롬프트
        "transformation_summary": "", # 꿈 재구성 요약
        "keyword_mappings": [], # 키워드 변환 매핑
        "nightmare_image_url": "", # 악몽 이미지 URL
        "reconstructed_image_url": "", # 재구성된 꿈 이미지 URL
        "nightmare_keywords": [], # 악몽의 핵심 키워드
    }
    # 세션 상태 변수가 존재하지 않으면 기본값으로 초기화
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --- 6. 세션 상태 초기화 함수 ---
    def initialize_session_state():
        # 모든 세션 상태 변수를 기본값으로 재설정
        for key, value in session_defaults.items():
            st.session_state[key] = value

    # --- 7. UI 구성: 오디오 입력 부분 ---
    tab1, tab2 = st.tabs(["🎤 실시간 녹음하기", "📁 오디오 파일 업로드"]) # 두 개의 탭 생성

    audio_bytes = None # 오디오 바이트 데이터를 저장할 변수
    file_name = None # 오디오 파일 이름을 저장할 변수

    with tab1: # 실시간 녹음 탭
        st.write("녹음 버튼을 눌러 악몽을 이야기해 주세요.")
        wav_audio_data = st_audiorec() # st_audiorec 위젯으로 오디오 녹음
        if wav_audio_data is not None:
            audio_bytes = wav_audio_data # 녹음된 오디오 데이터 저장
            file_name = "recorded_dream.wav" # 파일 이름 설정

    with tab2: # 오디오 파일 업로드 탭
        st.write("또는 오디오 파일을 직접 업로드할 수도 있습니다.")
        uploaded_file = st.file_uploader(
            "악몽 오디오 파일 선택",
            type=["mp3", "wav", "m4a", "ogg"], # 지원하는 파일 형식
            key="audio_uploader" # 위젯의 고유 키
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue() # 업로드된 파일의 바이트 데이터 저장
            file_name = uploaded_file.name # 업로드된 파일의 이름 저장

    # --- 8. 1단계: 오디오 → 텍스트 전사 (STT) + 안전성 검사 ---
    # 오디오 데이터가 있고 아직 처리되지 않았다면
    if audio_bytes is not None and not st.session_state.audio_processed:
        initialize_session_state() # 새로운 오디오가 들어오면 세션 상태 초기화
        
        temp_audio_dir = "user_data/audio" # 임시 오디오 파일 저장 디렉토리
        os.makedirs(temp_audio_dir, exist_ok=True) # 디렉토리가 없으면 생성

        audio_path = None # 임시 오디오 파일 경로

        try:
            # 파일 확장자 추출 또는 기본값 설정
            suffix = os.path.splitext(file_name)[1] if file_name else ".wav"
            # 임시 파일 생성 및 오디오 바이트 데이터 쓰기
            with tempfile.NamedTemporaryFile(delete=False, dir=temp_audio_dir, suffix=suffix) as temp_file:
                temp_file.write(audio_bytes)
                audio_path = temp_file.name # 임시 파일 경로 저장
            
            # 임시 파일이 제대로 생성되지 않은 경우 오류 처리
            if not audio_path or not os.path.exists(audio_path):
                st.error("임시 오디오 파일 생성에 실패했습니다.")
                st.session_state.audio_processed = False
                st.rerun() # UI 재실행하여 상태 갱신

            with st.spinner("음성을 텍스트로 변환하고 안전성 검사 중... 🕵️‍♂️"):
                transcribed_text = _stt_service.transcribe_audio(audio_path) # STT 서비스로 음성 텍스트 변환
                
                st.session_state.original_dream_text = transcribed_text # 원본 텍스트 저장

                safety_result = _moderation_service.check_text_safety(transcribed_text) # 변환된 텍스트 안전성 검사

                if safety_result["flagged"]: # 안전성 검사 실패 시
                    st.error(safety_result["text"]) # 에러 메시지 출력
                    st.session_state.audio_processed = False # 오디오 처리 상태 초기화
                    st.session_state.dream_text = "" # 꿈 텍스트 비움
                else: # 안전성 검사 통과 시
                    st.session_state.dream_text = transcribed_text # 꿈 텍스트 저장
                    st.success("안전성 검사: " + safety_result["text"]) # 성공 메시지 출력
                    st.session_state.audio_processed = True # 오디오 처리 완료 상태로 변경

        except Exception as e:
            st.error(f"오디오 처리 중 예상치 못한 오류가 발생했습니다: {e}")
            st.session_state.audio_processed = False
            st.session_state.dream_text = ""
            print(f"ERROR during audio processing: {e}")
        finally:
            # 임시 오디오 파일 삭제
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
    if st.session_state.original_dream_text: # 원본 꿈 텍스트가 있다면
        st.markdown("---")
        st.subheader("📝 나의 악몽 이야기 (텍스트 변환 결과)")
        st.info(st.session_state.original_dream_text) # 변환된 텍스트 표시

        # 꿈 텍스트가 안전성 검사를 통과했고, 아직 분석이 시작되지 않았다면
        if st.session_state.dream_text and not st.session_state.analysis_started:
            if st.button("✅ 이 내용으로 꿈 분석하기"): # 분석 시작 버튼
                st.session_state.analysis_started = True # 분석 시작 플래그 설정
                st.rerun() # UI 재실행하여 상태 갱신
        elif not st.session_state.dream_text and st.session_state.audio_processed:
            st.warning("입력된 꿈 내용이 안전성 검사를 통과하지 못했습니다. 분석을 진행할 수 없습니다.") # 안전성 검사 실패 시 경고

    # --- 10. 3단계: 리포트 생성 ---
    # 분석이 시작되었고 아직 리포트가 생성되지 않았다면
    if st.session_state.analysis_started and st.session_state.dream_report is None:
        if st.session_state.original_dream_text: # 원본 꿈 텍스트가 있다면
            with st.spinner("RAG가 지식 베이스를 참조하여 리포트를 생성하는 중... 🧠"):
                # RAG를 활용한 리포트 생성 서비스 호출
                report = _report_generator_service.generate_report_with_rag(st.session_state.original_dream_text)
                st.session_state.dream_report = report # 생성된 리포트 저장
                st.session_state.nightmare_keywords = report.get("keywords", []) # 리포트에서 키워드 추출하여 저장
                st.rerun() # UI 재실행하여 상태 갱신
        else:
            st.error("분석할 꿈 텍스트가 없습니다. 다시 시도해주세요.")
            st.session_state.analysis_started = False # 분석 시작 플래그 초기화

    # --- 11. 4단계: 감정 분석 리포트 출력 및 이미지 생성 버튼 ---
    if st.session_state.dream_report: # 꿈 리포트가 있다면
        report = st.session_state.dream_report # 세션 상태에서 리포트 가져오기
        st.markdown("---")
        st.subheader("📊 감정 분석 리포트") # 리포트 섹션 제목

        emotions = report.get("emotions", []) # 감정 목록 가져오기
        if emotions:
            st.markdown("##### 꿈 속 감정 구성:")
            for emotion in emotions:
                st.write(f"- {emotion.get('emotion', '알 수 없는 감정')}") # 감정 명칭 출력
                score = emotion.get('score', 0)
                st.progress(score, text=f"{score*100:.1f}%") # 감정 점수를 진행바로 표시

        keywords = report.get("keywords", []) # 키워드 목록 가져오기
        if keywords:
            st.markdown("##### 감정 키워드:")
            # 키워드에 빨간색 강조 스타일 적용하여 HTML로 표시
            keywords_str_list = [f'<span style="color: red; font-weight: bold;">{keyword}</span>' for keyword in keywords]
            keywords_html = f"[{', '.join(keywords_str_list)}]"
            st.markdown(keywords_html, unsafe_allow_html=True) # HTML 렌더링 허용

        summary = report.get("analysis_summary", "") # 분석 요약 가져오기
        if summary:
            st.markdown("##### 📝 종합 분석:")
            st.info(summary) # 분석 요약 정보 박스로 표시
        
        st.markdown("---")
        st.subheader("🎨 꿈 이미지 생성하기") # 이미지 생성 섹션 제목
        st.write("분석 리포트를 바탕으로, 이제 꿈을 시각화해 보세요. 어떤 이미지를 먼저 보시겠어요?")
        
        col1, col2 = st.columns(2) # 이미지 생성 버튼을 위한 2개 컬럼 생성

        with col1: # 악몽 이미지 생성 컬럼
            if st.button("😱 악몽 이미지 그대로 보기"): # 악몽 이미지 버튼
                with st.spinner("악몽을 시각화하는 중... 잠시만 기다려주세요."):
                    # 악몽 이미지 생성 프롬프트 생성
                    prompt = _dream_analyzer_service.create_nightmare_prompt(
                        st.session_state.original_dream_text, # 원본 꿈 텍스트
                        st.session_state.dream_report # 꿈 리포트
                    )
                    st.session_state.nightmare_prompt = prompt # 생성된 프롬프트 저장
                    # 이미지 생성 서비스로 악몽 이미지 생성
                    nightmare_image_url = _image_generator_service.generate_image_from_prompt(prompt)
                    st.session_state.nightmare_image_url = nightmare_image_url # 생성된 이미지 URL 저장
                    st.rerun() # UI 재실행하여 상태 갱신

        with col2: # 재구성된 꿈 이미지 생성 컬럼
            if st.button("✨ 재구성된 꿈 이미지 보기"): # 재구성된 꿈 이미지 버튼
                with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중... 🌈"):
                    # 꿈 재구성 프롬프트 및 분석 결과 생성
                    reconstructed_prompt, transformation_summary, keyword_mappings = \
                        _dream_analyzer_service.create_reconstructed_prompt_and_analysis(
                            st.session_state.original_dream_text, # 원본 꿈 텍스트
                            st.session_state.dream_report # 꿈 리포트
                        )
                    st.session_state.reconstructed_prompt = reconstructed_prompt # 재구성된 프롬프트 저장
                    st.session_state.transformation_summary = transformation_summary # 변환 요약 저장
                    st.session_state.keyword_mappings = keyword_mappings # 키워드 매핑 저장
                    
                    # 이미지 생성 서비스로 재구성된 이미지 생성
                    reconstructed_image_url = _image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                    st.session_state.reconstructed_image_url = reconstructed_image_url # 생성된 이미지 URL 저장
                    st.rerun() # UI 재실행하여 상태 갱신

    # --- 12. 5단계: 생성된 이미지 표시 및 키워드 강조 ---
    # 텍스트 내 키워드를 강조하는 헬퍼 함수
    def highlight_keywords(text, keywords, color="red"):
        # 키워드를 길이 역순으로 정렬하여 긴 키워드가 먼저 매치되도록 함
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        # HTML 태그와 일반 텍스트를 분리하기 위한 정규식
        html_tag_splitter = re.compile(r'(?s)(<[^>]+>.*?<\/[^>]+>|<[^>]+\/>)')
        
        # 텍스트를 HTML 태그 부분과 일반 텍스트 부분으로 분리
        segments = html_tag_splitter.split(text)
        
        processed_parts = []
        for i, segment in enumerate(segments):
            if i % 2 == 0: # 짝수 인덱스는 일반 텍스트 부분
                current_text_segment = segment
                for keyword in sorted_keywords:
                    if not keyword.strip(): # 비어있는 키워드는 건너뜀
                        continue
                    
                    # 단어 경계 및 대소문자 무시 (re.escape로 특수문자 처리)
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    
                    # 정규식 대체로 키워드에 강조 태그 삽입
                    current_text_segment = re.sub(pattern, f"<span style='color:{color}; font-weight:bold;'>{keyword}</span>", current_text_segment, flags=re.IGNORECASE)
                processed_parts.append(current_text_segment)
            else: # 홀수 인덱스는 HTML 태그 부분 (그대로 유지)
                processed_parts.append(segment)
                
        return "".join(processed_parts) # 분리된 부분을 다시 합쳐서 최종 결과 반환

    # 악몽 이미지 또는 재구성된 이미지가 생성되었다면
    if (st.session_state.nightmare_image_url and st.session_state.nightmare_image_url.startswith("http")) or \
       (st.session_state.reconstructed_image_url and st.session_state.reconstructed_image_url.startswith("http")):
        st.markdown("---")
        st.subheader("생성된 꿈 이미지")
        img_col1, img_col2 = st.columns(2) # 2개 컬럼으로 이미지 표시

        with img_col1: # 악몽 이미지 표시 컬럼
            if st.session_state.nightmare_image_url:
                if st.session_state.nightmare_image_url.startswith("http"): # 유효한 URL인 경우
                    st.image(st.session_state.nightmare_image_url, caption="악몽 시각화") # 이미지 표시
                    with st.expander("생성 프롬프트 및 주요 키워드 보기"): # 프롬프트와 키워드를 숨김/보임 토글
                        # 악몽 프롬프트에 키워드 강조 적용
                        all_nightmare_keywords_for_highlight = st.session_state.nightmare_keywords
                        highlighted_nightmare_prompt = highlight_keywords(st.session_state.nightmare_prompt, all_nightmare_keywords_for_highlight, "red")
                        st.markdown(f"**프롬프트:** {highlighted_nightmare_prompt}", unsafe_allow_html=True)
                        
                        if all_nightmare_keywords_for_highlight:
                            st.markdown("---")
                            # 강조된 키워드 리스트를 직접 출력
                            highlighted_list = [f"<span style='color:red; font-weight:bold;'>{k}</span>" for k in all_nightmare_keywords_for_highlight]
                            st.markdown(f"**주요 키워드:** {', '.join(highlighted_list)}", unsafe_allow_html=True)
                else:
                    st.error(f"악몽 이미지 생성 실패: {st.session_state.nightmare_image_url}") # 이미지 생성 실패 메시지

        with img_col2: # 재구성된 꿈 이미지 표시 컬럼
            if st.session_state.reconstructed_image_url:
                if st.session_state.reconstructed_image_url.startswith("http"): # 유효한 URL인 경우
                    st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈") # 이미지 표시
                    with st.expander("생성 프롬프트 및 변환 과정 보기"): # 프롬프트와 변환 과정을 숨김/보임 토글
                        # 재구성 프롬프트에 키워드 강조 적용
                        transformed_only_keywords_from_mapping = [mapping.get('transformed', '') for mapping in st.session_state.keyword_mappings if mapping.get('transformed')]
                        all_reconstructed_keywords_for_highlight = transformed_only_keywords_from_mapping

                        highlighted_reconstructed_prompt = highlight_keywords(st.session_state.reconstructed_prompt, all_reconstructed_keywords_for_highlight, "green")
                        
                        st.markdown(f"**프롬프트:** {highlighted_reconstructed_prompt}", unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("**변환 요약:**")
                        st.write(st.session_state.transformation_summary) # 변환 요약 출력
                        
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
                    st.error(f"재구성 이미지 생성 실패: {st.session_state.reconstructed_image_url}") # 이미지 생성 실패 메시지