import streamlit as st
import os
from PIL import Image # PIL은 직접 사용되지 않지만, 기존 코드에 있었으므로 유지
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service, report_generator_service
from st_audiorec import st_audiorec
import base64
# from core.config import settings # <-- 이제 이 줄은 필요 없습니다.
# core/config.py가 .env 파일을 로드하도록 되어 있으므로, 별도의 import는 불필요합니다.
# 하지만, Streamlit 앱 시작 전에 확실히 load_dotenv가 실행되도록
# core.config 모듈을 한 번 임포트해주는 것이 좋습니다.
import core.config # core/config.py 모듈 자체를 임포트하여 load_dotenv가 실행되도록 합니다.

# --- 1. 페이지 설정 (반드시 모든 st. 명령보다 먼저 와야 합니다!) ---
st.set_page_config(
    page_title="보여dream | 당신의 악몽을 재구성합니다",
    page_icon="🌙",
    layout="wide"
)

# --- 2. API 키 로드 및 서비스 초기화 ---
# core.config 모듈이 임포트되면서 load_dotenv()가 이미 호출되었으므로,
# 여기서는 os.getenv()로 환경 변수를 직접 가져옵니다.
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# API 키가 제대로 설정되었는지 확인하고, 없으면 앱 실행을 중단합니다.
if not openai_api_key:
    st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 시스템 환경 변수를 확인하거나 '.env' 파일을 설정해주세요.")
    st.stop() # API 키가 없으면 애플리케이션 실행을 중단합니다.

# 각 서비스 클래스의 인스턴스를 생성합니다.
# 이렇게 한 번 생성해두면 앱 전체에서 재사용할 수 있어 효율적입니다.
_stt_service = stt_service.STTService(api_key=openai_api_key)
_dream_analyzer_service = dream_analyzer_service.DreamAnalyzerService(api_key=openai_api_key)
_image_generator_service = image_generator_service.ImageGeneratorService(api_key=openai_api_key)
_moderation_service = moderation_service.ModerationService(api_key=openai_api_key)
_report_generator_service = report_generator_service.ReportGeneratorService(api_key=openai_api_key)

# --- 3. 로고 이미지 로딩 및 표시 ---

# base64로 이미지 인코딩 (업로드한 파일 기준)
def get_base64_image(image_path):
    """
    주어진 경로의 이미지를 base64 문자열로 인코딩합니다.
    파일을 찾을 수 없거나 로드 오류 시 None을 반환합니다.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"로고 파일이 없습니다: '{image_path}'. 기본 타이틀을 사용합니다.")
        return None
    except Exception as e:
        st.error(f"로고를 로드하는 중 오류가 발생했습니다: {e}")
        return None

# 로고 이미지 경로 설정 (user_data/image 디렉토리 생성 후)
logo_dir = "user_data/image"
os.makedirs(logo_dir, exist_ok=True) # 로고 이미지를 위한 디렉토리 생성
logo_path = os.path.join(logo_dir, "Logo.png") # 로고 파일명은 'Logo.png'로 가정

# base64 인코딩된 이미지 불러오기
logo_base64 = get_base64_image(logo_path)

# 로고 + 타이틀 정렬 (로고가 성공적으로 로드된 경우에만 표시)
if logo_base64:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{logo_base64}" width="120" style="margin-right: 20px;">
            <h1 style="margin: 0;">보여dream 🌙</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("보여dream 🌙")

st.write("악몽을 녹음하거나 파일을 업로드해 주세요.")

# --- 4. 텍스트 저장/로드 함수 및 경로 설정 ---
# 텍스트 저장 경로 (사용자 데이터를 임시 저장할 경로)
dream_text_path = "user_data/dream_text.txt"
os.makedirs("user_data", exist_ok=True) # user_data 디렉토리가 없으면 생성

def save_dream_text(text):
    """주어진 텍스트를 파일에 저장합니다."""
    with open(dream_text_path, "w", encoding="utf-8") as f:
        f.write(text)

def load_dream_text():
    """파일에서 텍스트를 불러옵니다."""
    if os.path.exists(dream_text_path):
        with open(dream_text_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

# --- 5. 세션 상태 기본값 초기화 (앱 시작 시) ---
if "dream_text" not in st.session_state:
    st.session_state.dream_text = ""

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False

if "derisked_text" not in st.session_state: # 현재 코드에서는 사용되지 않지만, 잠재적 확장을 위해 유지
    st.session_state.derisked_text = ""

if "dream_report" not in st.session_state:
    st.session_state.dream_report = None

if "nightmare_prompt" not in st.session_state:
    st.session_state.nightmare_prompt = ""

if "reconstructed_prompt" not in st.session_state:
    st.session_state.reconstructed_prompt = ""

if "transformation_summary" not in st.session_state: # 변환 요약 세션 상태 추가
    st.session_state.transformation_summary = ""

if "keyword_mappings" not in st.session_state: # 키워드 매핑 세션 상태 추가
    st.session_state.keyword_mappings = []

if "nightmare_image_url" not in st.session_state:
    st.session_state.nightmare_image_url = ""

if "reconstructed_image_url" not in st.session_state:
    st.session_state.reconstructed_image_url = ""

# --- 6. 세션 상태 초기화 함수 (새로운 녹음/파일 업로드 시 기존 상태 초기화) ---
def initialize_session_state():
    """
    새로운 오디오 처리 시작 시, 이전 분석 및 이미지 생성 관련 세션 상태를 초기화합니다.
    'dream_text'는 새 오디오 텍스트로 덮어쓰여질 것이므로 초기화하지 않습니다.
    """
    st.session_state.derisked_text = ""
    st.session_state.dream_report = None
    st.session_state.nightmare_prompt = ""
    st.session_state.reconstructed_prompt = ""
    st.session_state.transformation_summary = "" # 초기화 추가
    st.session_state.keyword_mappings = []       # 초기화 추가
    st.session_state.nightmare_image_url = ""
    st.session_state.reconstructed_image_url = ""
    st.session_state.audio_processed = False
    st.session_state.analysis_started = False

# --- 7. UI 구성: 오디오 입력 부분 ---
tab1, tab2 = st.tabs(["🎤 실시간 녹음하기", "📁 오디오 파일 업로드"])

audio_bytes = None # 오디오 데이터 (바이트)를 저장할 변수
file_name = None   # 오디오 파일 이름을 저장할 변수

with tab1:
    st.write("녹음 버튼을 눌러 악몽을 이야기해 주세요.")
    wav_audio_data = st_audiorec() # st_audiorec 라이브러리를 통해 오디오 녹음
    if wav_audio_data is not None:
        audio_bytes = wav_audio_data
        file_name = "recorded_dream.wav"

with tab2:
    st.write("또는 오디오 파일을 직접 업로드할 수도 있습니다.")
    uploaded_file = st.file_uploader(
        "악몽 오디오 파일 선택",
        type=["mp3", "wav", "m4a", "ogg"], # 지원하는 오디오 파일 형식
        key="audio_uploader" # Streamlit 위젯의 고유 키 (세션 간 상태 유지를 위해 필요)
    )
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

# --- 8. 1단계: 오디오 → 텍스트 전사 (STT) + 안전성 검사 ---
if audio_bytes is not None and not st.session_state.audio_processed:
    initialize_session_state()
    
    audio_dir = "user_data/audio"
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, file_name)

    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    with st.spinner("음성을 텍스트로 변환하고 안전성 검사 중... 🕵️‍♂️"):
        transcribed_text = _stt_service.transcribe_audio(audio_path)
        safety_result = _moderation_service.check_text_safety(transcribed_text)

        if safety_result["flagged"]:
            st.error(safety_result["text"])
            st.session_state.audio_processed = False
        else:
            st.session_state.dream_text = safety_result["text"]
            st.session_state.audio_processed = True

    os.remove(audio_path)
    st.rerun()

# --- 9. 2단계: 전사된 텍스트 출력 및 분석 시작 버튼 ---
if st.session_state.dream_text:
    st.markdown("---")
    st.subheader("📝 나의 악몽 이야기 (텍스트 변환 결과)")
    st.info(st.session_state.dream_text)

    if not st.session_state.analysis_started:
        if st.button("✅ 이 내용으로 꿈 분석하기"):
            st.session_state.analysis_started = True
            st.rerun()

# --- 10. 3단계: 분석 시작 시 감정 분석 리포트 생성 ---
if st.session_state.analysis_started and st.session_state.dream_report is None:
    with st.spinner("꿈 내용을 분석하여 리포트를 생성하는 중... 🧠"):
        report = _report_generator_service.generate_report(st.session_state.dream_text)
        st.session_state.dream_report = report
        st.rerun()

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
            st.progress(score, text=f"{score}%")

    keywords = report.get("keywords", [])
    if keywords:
        st.markdown("##### 감정 키워드:")
        keywords_str = ", ".join(f'"{keyword}"' for keyword in keywords)
        st.code(f"[{keywords_str}]", language="json")

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
                nightmare_prompt = _dream_analyzer_service.create_nightmare_prompt(st.session_state.dream_text)
                st.session_state.nightmare_prompt = nightmare_prompt
                nightmare_image_url = _image_generator_service.generate_image_from_prompt(nightmare_prompt)
                st.session_state.nightmare_image_url = nightmare_image_url
                
                # 재구성 이미지 관련 세션 상태는 악몽 이미지 선택 시 초기화
                st.session_state.reconstructed_prompt = ""
                st.session_state.transformation_summary = ""
                st.session_state.keyword_mappings = []
                st.session_state.reconstructed_image_url = ""

                st.rerun()

    with col2:
        if st.button("✨ 재구성된 꿈 이미지 보기"):
            with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중... 🌈"):
                reconstructed_prompt, transformation_summary, keyword_mappings = \
                    _dream_analyzer_service.create_reconstructed_prompt(
                        st.session_state.dream_text, 
                        st.session_state.dream_report
                    )
                st.session_state.reconstructed_prompt = reconstructed_prompt
                st.session_state.transformation_summary = transformation_summary
                st.session_state.keyword_mappings = keyword_mappings           

                reconstructed_image_url = _image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                st.session_state.reconstructed_image_url = reconstructed_image_url

                # 악몽 이미지 관련 세션 상태는 재구성 이미지 선택 시 초기화
                st.session_state.nightmare_prompt = ""
                st.session_state.nightmare_image_url = ""

                st.rerun()

# --- 12. 5단계: 생성된 이미지 표시 ---
if st.session_state.nightmare_image_url or st.session_state.reconstructed_image_url:
    st.markdown("---")
    st.subheader("생성된 꿈 이미지")

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        if st.session_state.nightmare_image_url:
            if st.session_state.nightmare_image_url.startswith("http"):
                st.image(st.session_state.nightmare_image_url, caption="악몽 시각화")
                with st.expander("생성 프롬프트 보기"):
                    st.write(st.session_state.nightmare_prompt)
            else:
                st.error(f"악몽 이미지 생성 실패: {st.session_state.nightmare_image_url}")

    with img_col2:
        if st.session_state.reconstructed_image_url:
            if st.session_state.reconstructed_image_url.startswith("http"):
                st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈")
                with st.expander("생성 프롬프트 보기"):
                    highlighted_prompt = st.session_state.reconstructed_prompt
                    for mapping in st.session_state.keyword_mappings:
                        original_concept = mapping.get("original")
                        transformed_concept = mapping.get("transformed")
                        if transformed_concept and transformed_concept in highlighted_prompt:
                            highlighted_prompt = highlighted_prompt.replace(
                                transformed_concept,
                                f'**<span style="color: blue; font-weight: bold;">{transformed_concept}</span>**'
                            )
                    st.markdown(highlighted_prompt, unsafe_allow_html=True)

                if st.session_state.transformation_summary:
                    st.markdown("---")
                    st.subheader("💡 꿈 변환 요약")
                    st.info(st.session_state.transformation_summary)
                
                if st.session_state.keyword_mappings:
                    st.markdown("---")
                    st.subheader("↔️ 주요 변환 요소:")
                    for mapping in st.session_state.keyword_mappings:
                        original = mapping.get('original', '알 수 없음')
                        transformed = mapping.get('transformed', '알 수 없음')
                        st.write(f"- **{original}** ➡️ **{transformed}**")
                
            else:
                st.error(f"재구성된 꿈 이미지 생성 실패: {st.session_state.reconstructed_image_url}")
