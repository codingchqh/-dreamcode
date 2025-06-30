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
logo_path = os.path.join("user_data/image", "보여dream로고 투명.png")
logo_base64 = get_base64_image(logo_path)

# 나비몽 챗봇 이미지 경로 정의
navimong_chatbot_image_path = os.path.join("user_data/image", "나비몽 챗봇.png")
# 이미지가 존재하는지 미리 확인하여 불필요한 호출 방지
navimong_chatbot_image_exists = os.path.exists(navimong_chatbot_image_path)

# --- 전체 페이지 레이아웃을 위한 컬럼 분할 ---
col_left_main, col_center_main, col_right_main = st.columns([1, 4, 1])

with col_center_main: # 로고와 주요 콘텐츠가 들어갈 중앙 컬럼
    # --- 로고 및 타이틀 표시 ---
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
    
    # --- '악몽을 녹음하거나 파일을 업로드해 주세요.' 텍스트 왼쪽에 나비몽 챗봇 이미지 배치 ---
    col_chatbot_img, col_text = st.columns([0.15, 0.85]) 
    
    with col_chatbot_img:
        if navimong_chatbot_image_exists:
            st.image(navimong_chatbot_image_path, width=60) # 이미지 크기 60으로 설정
    
    with col_text:
        st.markdown("<h3 style='margin-top: 15px; margin-left: 0px;'>악몽을 녹음하거나 파일을 업로드해 주세요.</h3>", unsafe_allow_html=True)


    st.markdown("---") # 구분선

    # --- 5. 세션 상태 기본값 초기화 ---
    session_defaults = {
        "dream_text": "", "original_dream_text": "", "analysis_started": False,
        "audio_processed": False, "derisked_text": "", "dream_report": None,
        "nightmare_prompt": "", "reconstructed_prompt": "", "transformation_summary": "",
        "keyword_mappings": [], # 키워드 변환 매핑 저장을 위해 추가
        "nightmare_image_url": "", "reconstructed_image_url": "",
        "nightmare_keywords": [], # 악몽 키워드 저장을 위해 추가 (추가됨)
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
                # 리포트에서 키워드를 세션 상태에 저장
                st.session_state.nightmare_keywords = report.get("keywords", []) 
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
                    st.session_state.keyword_mappings = keyword_mappings # 변환된 키워드 매핑 저장
                    st.session_state.reconstructed_image_url = _image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                    st.rerun()

    # --- 12. 5단계: 생성된 이미지 표시 및 키워드 강조 ---
    # 키워드에 색상을 입히는 헬퍼 함수
    # Look-behind (긍정적/부정적) 어설션을 제거하여 're.error'를 해결합니다.
    def highlight_keywords(text, keywords, color="red"):
        # 키워드 리스트를 정규표현식에 사용할 수 있도록 이스케이프하고 '|'로 연결
        # 가장 긴 키워드가 먼저 매치되도록 역순 정렬 (더 정확한 매칭을 위해)
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        # 각 키워드에 대해 치환 수행
        for keyword in sorted_keywords:
            # HTML 태그 안에 있는 텍스트는 건너뛰도록 하는 정규식 (복잡성 증가 및 완벽하지 않을 수 있음)
            # 여기서는 간단하게 look-behind를 제거하여 오류를 피합니다.
            # keyword가 이미 <span style="...">...</span> 형태인지 확인하여 중복 강조 방지
            if f"<span style='color:{color};" in text and f">{keyword}</span>" in text:
                continue # 이미 강조된 키워드는 건너_stt_service

            # 단어 경계 (\b)를 사용하여 정확한 단어만 일치시키고, 대소문자 구분 없음 (re.IGNORECASE)
            # re.escape()는 키워드 내의 특수문자가 정규식 문자로 해석되는 것을 방지합니다.
            # 이 패턴은 더 이상 look-behind를 사용하지 않습니다.
            pattern = r'\b' + re.escape(keyword) + r'\b'
            text = re.sub(pattern, f"<span style='color:{color}; font-weight:bold;'>{keyword}</span>", text, flags=re.IGNORECASE)
        return text

    if (st.session_state.nightmare_image_url and st.session_state.nightmare_image_url.startswith("http")) or \
       (st.session_state.reconstructed_image_url and st.session_state.reconstructed_image_url.startswith("http")):
        st.markdown("---"); st.subheader("생성된 꿈 이미지")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            if st.session_state.nightmare_image_url.startswith("http"):
                st.image(st.session_state.nightmare_image_url, caption="악몽 시각화")
                with st.expander("생성 프롬프트 및 주요 키워드 보기"):
                    # --- 악몽 프롬프트 키워드 리스트 (수정됨: AI 포함 및 부정적 키워드 추가) ---
                    # 여기에 AI 모델이 실제로 추출한 키워드 + 이미지 분석을 위한 추가 키워드를 통합합니다.
                    # 'AI' 단어는 명시적으로 추가하여 항상 강조되도록 합니다.
                    nightmare_keywords_from_report = st.session_state.nightmare_keywords
                    additional_negative_keywords = [
                        'cold', 'dystopian', 'sterile', 'digital landscape', 'unstable',
                        'fractured', 'frozen moment', 'glitching sun', 'metallic', 'emotionless tone',
                        'hollow', 'pixelated', 'crumbling code', 'oppressive', 'cold blue',
                        'sterile white light', 'long shadows', 'corrupted data', 'glitching pixels',
                        'breakdown of perceived reality', 'haunting manifestation', 'dominance',
                        'oppressive silence', 'chilling', 'disembodied voice', 'cold, digital chaos',
                        'reverting', 'facade of happiness', 'underlying horror', 'prison',
                        'beautiful illusions', 'unsettling paranoia', 'insidious simulation',
                        'AI' # AI 단어 자체도 강조
                    ]
                    # 두 리스트를 합치고 중복 제거
                    all_nightmare_keywords_for_highlight = list(set(nightmare_keywords_from_report + additional_negative_keywords))


                    highlighted_nightmare_prompt = highlight_keywords(st.session_state.nightmare_prompt, all_nightmare_keywords_for_highlight, "red")
                    st.markdown(f"**프롬프트:** {highlighted_nightmare_prompt}", unsafe_allow_html=True)
                    
                    if all_nightmare_keywords_for_highlight: 
                        st.markdown("---")
                        # 키워드 목록도 빨간색으로 강조
                        highlighted_list = [f"<span style='color:red; font-weight:bold;'>{k}</span>" for k in all_nightmare_keywords_for_highlight]
                        st.markdown(f"**주요 키워드:** {', '.join(highlighted_list)}", unsafe_allow_html=True)
            elif st.session_state.nightmare_image_url:
                st.error(f"악몽 이미지 생성 실패: {st.session_state.nightmare_image_url}")
        with img_col2:
            if st.session_state.reconstructed_image_url.startswith("http"):
                st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈")
                with st.expander("생성 프롬프트 및 변환 과정 보기"):
                    # 재구성 프롬프트에 변환된 키워드 색상 적용
                    highlighted_reconstructed_prompt = st.session_state.reconstructed_prompt
                    transformed_keywords_display_list = [] # 화면에 표시할 변환된 키워드 리스트
                    
                    # keyword_mappings에서 변환된 키워드를 가져와 하이라이트
                    transformed_only_keywords = [mapping.get('transformed', '') for mapping in st.session_state.keyword_mappings if mapping.get('transformed')]
                    highlighted_reconstructed_prompt = highlight_keywords(highlighted_reconstructed_prompt, transformed_only_keywords, "green")
                    
                    st.markdown(f"**프롬프트:** {highlighted_reconstructed_prompt}", unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown("**변환 요약:**")
                    st.write(st.session_state.transformation_summary)
                    
                    # 변환된 키워드 목록을 '원본(빨간색) → 변환(초록색)' 형식으로 표시
                    if st.session_state.keyword_mappings:
                        for mapping in st.session_state.keyword_mappings:
                            original = mapping.get('original', 'N/A')
                            transformed = mapping.get('transformed', 'N/A')
                            transformed_keywords_display_list.append(f"<span style='color:red;'>{original}</span> → <span style='color:green;'>{transformed}</span>")
                        
                        st.markdown("---")
                        st.markdown(f"**변환된 키워드:** {', '.join(transformed_keywords_display_list)}", unsafe_allow_html=True)
            elif st.session_state.reconstructed_image_url:
                st.error(f"재구성 이미지 생성 실패: {st.session_state.reconstructed_image_url}")