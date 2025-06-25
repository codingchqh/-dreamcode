import streamlit as st
import os
from PIL import Image # PIL is imported but not used directly in the provided snippet for image processing in Streamlit
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service, report_generator_service
from st_audiorec import st_audiorec
import base64
from core.config import settings # core/config.py에서 settings 객체를 임포트합니다.

# --- 1. 페이지 설정 (반드시 모든 st. 명령보다 먼저 와야 합니다!) ---
st.set_page_config(
    page_title="보여dream | 당신의 악몽을 재구성합니다",
    page_icon="🌙",
    layout="wide"
)

# --- 2. API 키 로드 및 서비스 초기화 ---
# config.py에서 가져온 settings 객체를 통해 OPENAI_API_KEY에 접근합니다.
openai_api_key = settings.OPENAI_API_KEY

# API 키가 제대로 설정되었는지 확인하고, 없으면 앱 실행을 중단합니다.
if not openai_api_key:
    st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 시스템 환경 변수를 확인하거나 'core/config.py' 파일을 설정해주세요.")
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
    st.title("보여dream 🌙") # 로고가 없을 경우 기본 타이틀 표시

st.write("악몽을 녹음하거나 파일을 업로드해 주세요.") # 로고/타이틀 아래 앱 설명

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
# Streamlit 앱 상태 관리를 위한 세션 상태 변수들
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
# 오디오 데이터가 존재하고 아직 처리되지 않았다면 STT 및 안전성 검사 실행
if audio_bytes is not None and not st.session_state.audio_processed:
    initialize_session_state()  # 새로운 오디오가 들어왔으므로 세션 상태 초기화
    
    audio_dir = "user_data/audio"
    os.makedirs(audio_dir, exist_ok=True) # 오디오 저장 디렉토리 생성
    audio_path = os.path.join(audio_dir, file_name)

    # 임시 오디오 파일 저장
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    with st.spinner("음성을 텍스트로 변환하고 안전성 검사 중... 🕵️‍♂️"):
        # STT 서비스 호출 (인스턴스 사용)
        transcribed_text = _stt_service.transcribe_audio(audio_path)
        # 안전성 검사 서비스 호출 (인스턴스 사용)
        safety_result = _moderation_service.check_text_safety(transcribed_text)

        if safety_result["flagged"]: # 안전성 검사에서 문제가 감지된 경우
            st.error(safety_result["text"]) # 사용자에게 경고 메시지 표시
            st.session_state.audio_processed = False # 다시 오디오 처리 시도 가능하도록 상태 리셋
        else:
            st.session_state.dream_text = safety_result["text"] # 안전한 텍스트를 세션 상태에 저장
            st.session_state.audio_processed = True # 오디오 처리 완료 상태로 설정

    os.remove(audio_path) # 임시 오디오 파일 삭제
    st.rerun() # Streamlit 앱 다시 실행하여 다음 단계로 진행 (상태 업데이트 반영)

# --- 9. 2단계: 전사된 텍스트 출력 및 분석 시작 버튼 ---
if st.session_state.dream_text: # 꿈 텍스트가 세션 상태에 존재하면
    st.markdown("---")
    st.subheader("📝 나의 악몽 이야기 (텍스트 변환 결과)")
    st.info(st.session_state.dream_text) # 변환된 텍스트 사용자에게 표시

    if not st.session_state.analysis_started: # 아직 분석이 시작되지 않았다면
        if st.button("✅ 이 내용으로 꿈 분석하기"): # 분석 시작 버튼 표시
            st.session_state.analysis_started = True # 분석 시작 상태로 변경
            st.rerun() # 앱 다시 실행하여 분석 단계로 진행

# --- 10. 3단계: 분석 시작 시 감정 분석 리포트 생성 ---
# 분석이 시작되었고 리포트가 아직 생성되지 않았다면 리포트 생성
if st.session_state.analysis_started and st.session_state.dream_report is None:
    with st.spinner("꿈 내용을 분석하여 리포트를 생성하는 중... 🧠"):
        # 감정 분석 리포트 생성 서비스 호출 (인스턴스 사용)
        report = _report_generator_service.generate_report(st.session_state.dream_text)
        st.session_state.dream_report = report # 생성된 리포트를 세션 상태에 저장
        st.rerun() # 앱 다시 실행하여 리포트 출력 단계로 진행 (상태 업데이트 반영)

# --- 11. 4단계: 감정 분석 리포트 출력 및 이미지 생성 버튼 ---
if st.session_state.dream_report: # 감정 분석 리포트가 세션 상태에 존재하면
    report = st.session_state.dream_report
    st.markdown("---")
    st.subheader("📊 감정 분석 리포트")

    # 리포트에서 감정 정보 가져와서 시각화
    emotions = report.get("emotions", [])
    if emotions:
        st.markdown("##### 꿈 속 감정 구성:")
        for emotion in emotions:
            st.write(f"- {emotion.get('emotion', '알 수 없는 감정')}")
            score = emotion.get('score', 0)
            st.progress(score, text=f"{score}%") # 진행 바로 감정 점수 표시

    # 리포트에서 키워드 정보 가져와서 표시
    keywords = report.get("keywords", [])
    if keywords:
        st.markdown("##### 감정 키워드:")
        keywords_str = ", ".join(f'"{keyword}"' for keyword in keywords)
        st.code(f"[{keywords_str}]", language="json")

    # 리포트에서 종합 분석 요약 가져와서 표시
    summary = report.get("analysis_summary", "")
    if summary:
        st.markdown("##### 📝 종합 분석:")
        st.info(summary)
    
    # 이미지 생성 버튼 표시
    st.markdown("---")
    st.subheader("🎨 꿈 이미지 생성하기")
    st.write("분석 리포트를 바탕으로, 이제 꿈을 시각화해 보세요. 어떤 이미지를 먼저 보시겠어요?")
    
    # 이미지 선택을 위한 두 개의 컬럼
    col1, col2 = st.columns(2)

    with col1:
        # '악몽 이미지 그대로 보기' 버튼
        if st.button("😱 악몽 이미지 그대로 보기"):
            with st.spinner("악몽을 시각화하는 중... 잠시만 기다려주세요."):
                # 악몽 프롬프트 생성 서비스 호출 (인스턴스 사용)
                nightmare_prompt = _dream_analyzer_service.create_nightmare_prompt(st.session_state.dream_text)
                st.session_state.nightmare_prompt = nightmare_prompt # 생성된 프롬프트 저장
                # 이미지 생성 서비스 호출 (인스턴스 사용)
                nightmare_image_url = _image_generator_service.generate_image_from_prompt(nightmare_prompt)
                st.session_state.nightmare_image_url = nightmare_image_url # 생성된 이미지 URL 저장
                st.rerun() # 앱 다시 실행하여 이미지 표시

    with col2:
        # '재구성된 꿈 이미지 보기' 버튼
        if st.button("✨ 재구성된 꿈 이미지 보기"):
            with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중... 🌈"):
                # 재구성된 꿈 프롬프트 생성 서비스 호출 (인스턴스 사용)
                # 변경된 부분: dream_report를 함께 전달합니다!
                reconstructed_prompt = _dream_analyzer_service.create_reconstructed_prompt(
                    st.session_state.dream_text, 
                    st.session_state.dream_report # 감정 분석 리포트 객체를 함께 전달!
                )
                st.session_state.reconstructed_prompt = reconstructed_prompt # 생성된 프롬프트 저장
                # 이미지 생성 서비스 호출 (인스턴스 사용)
                reconstructed_image_url = _image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                st.session_state.reconstructed_image_url = reconstructed_image_url # 생성된 이미지 URL 저장
                st.rerun() # 앱 다시 실행하여 이미지 표시

# --- 12. 5단계: 생성된 이미지 표시 ---
# 악몽 이미지 또는 재구성된 꿈 이미지가 생성되었다면 표시
if st.session_state.nightmare_image_url or st.session_state.reconstructed_image_url:
    st.markdown("---")
    st.subheader("생성된 꿈 이미지")

    # 이미지를 나란히 표시하기 위한 두 개의 컬럼
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        if st.session_state.nightmare_image_url:
            if st.session_state.nightmare_image_url.startswith("http"):
                st.image(st.session_state.nightmare_image_url, caption="악몽 시각화")
                with st.expander("생성 프롬프트 보기"): # 생성된 프롬프트를 볼 수 있는 확장 가능한 섹션
                    st.write(st.session_state.nightmare_prompt)
            else:
                st.error(f"악몽 이미지 생성 실패: {st.session_state.nightmare_image_url}") # 이미지 URL이 유효하지 않을 경우 오류 메시지

    with img_col2:
        if st.session_state.reconstructed_image_url:
            if st.session_state.reconstructed_image_url.startswith("http"):
                st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈")
                with st.expander("생성 프롬프트 보기"): # 생성된 프롬프트를 볼 수 있는 확장 가능한 섹션
                    st.write(st.session_state.reconstructed_prompt)
            else:
                st.error(f"재구성된 꿈 이미지 생성 실패: {st.session_state.reconstructed_image_url}") # 이미지 URL이 유효하지 않을 경우 오류 메시지
