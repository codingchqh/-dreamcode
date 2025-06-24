import streamlit as st
import os
from services import stt_service, dream_analyzer_service, image_generator_service, moderation_service
# 오디오 녹음 라이브러리를 임포트합니다.
from st_audiorec import st_audiorec

# --- 페이지 설정 ---
st.set_page_config(
    page_title="보여dream | 당신의 악몽을 재구성합니다",
    page_icon="🌙",
    layout="wide"
)

# --- 세션 상태 초기화 ---
# 사용자의 이전 작업 내용을 기억하기 위해 세션 상태를 사용합니다.
if 'dream_text' not in st.session_state:
    st.session_state.dream_text = ""
if 'nightmare_prompt' not in st.session_state:
    st.session_state.nightmare_prompt = ""
if 'reconstructed_prompt' not in st.session_state:
    st.session_state.reconstructed_prompt = ""
if 'nightmare_image_url' not in st.session_state:
    st.session_state.nightmare_image_url = ""
if 'reconstructed_image_url' not in st.session_state:
    st.session_state.reconstructed_image_url = ""
# 새로운 녹음이 시작되면 이전 결과를 초기화하기 위한 플래그
if 'new_recording' not in st.session_state:
    st.session_state.new_recording = True


# --- UI 구성 ---
st.title("보여dream 🌙")
st.write("아래 녹음 버튼을 눌러 당신의 악몽을 들려주세요. 다시 누르면 녹음이 중지됩니다.")

# --- 오디오 녹음 위젯 ---
# 사용자가 녹음을 멈추면 wav_audio_data에 오디오 데이터(bytes)가 담깁니다.
wav_audio_data = st_audiorec()

# 녹음된 데이터가 새로 들어온 경우
if wav_audio_data is not None:
    # 이전에 처리하던 내용이 있다면, 새로운 녹음이므로 모두 초기화합니다.
    # st.session_state.clear()를 사용하여 모든 이전 데이터를 깨끗이 지웁니다.
    if "new_recording" not in st.session_state or st.session_state.new_recording:
        st.session_state.clear()
        st.session_state.new_recording = False

    # 2. 녹음된 오디오 처리 및 텍스트 변환
    if st.session_state.dream_text == "":
        audio_dir = "user_data/audio"
        audio_path = os.path.join(audio_dir, "recorded_dream.wav")
        os.makedirs(audio_dir, exist_ok=True)

        # 녹음된 오디오 바이트 데이터를 임시 파일로 저장
        with open(audio_path, "wb") as f:
            f.write(wav_audio_data)

        with st.spinner("음성을 텍스트로 변환하고 안전성을 검사하는 중입니다... 🕵️‍♂️"):
            # 2-1. 음성을 텍스트로 변환
            transcribed_text = stt_service.transcribe_audio(audio_path)
            
            # 2-2. 1차 안전성 검사 (변환된 텍스트)
            safety_result = moderation_service.check_text_safety(transcribed_text)

            if safety_result["flagged"]:
                # 문제가 감지되면 사용자에게 알리고 프로세스 중단
                st.error(safety_result["text"])
                st.session_state.dream_text = "" 
            else:
                # 안전하면 다음 단계를 위해 텍스트 저장
                st.session_state.dream_text = safety_result["text"]
        
        # 임시 오디오 파일 삭제
        os.remove(audio_path)

# 3. 변환된 텍스트와 이미지 생성 버튼 표시
if st.session_state.dream_text:
    st.subheader("나의 악몽 이야기")
    st.write(st.session_state.dream_text)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("😱 악몽 이미지 그대로 보기"):
            with st.spinner("악몽을 시각화하는 중... 잠시만 기다려주세요."):
                nightmare_prompt = dream_analyzer_service.create_nightmare_prompt(st.session_state.dream_text)
                st.session_state.nightmare_prompt = nightmare_prompt
                nightmare_image_url = image_generator_service.generate_image_from_prompt(nightmare_prompt)
                st.session_state.nightmare_image_url = nightmare_image_url
                # 새로운 이미지 생성 후, 다시 녹음할 수 있도록 플래그 설정
                st.session_state.new_recording = True

    with col2:
        if st.button("✨ 재구성된 꿈 이미지 보기"):
            with st.spinner("악몽을 긍정적인 꿈으로 재구성하는 중... 🌈"):
                reconstructed_prompt = dream_analyzer_service.create_reconstructed_prompt(st.session_state.dream_text)
                st.session_state.reconstructed_prompt = reconstructed_prompt
                reconstructed_image_url = image_generator_service.generate_image_from_prompt(reconstructed_prompt)
                st.session_state.reconstructed_image_url = reconstructed_image_url
                # 새로운 이미지 생성 후, 다시 녹음할 수 있도록 플래그 설정
                st.session_state.new_recording = True

# 4. 생성된 이미지 표시
if st.session_state.get('nightmare_image_url') or st.session_state.get('reconstructed_image_url'):
    st.markdown("---")
    st.subheader("생성된 꿈 이미지")

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        if st.session_state.get('nightmare_image_url'):
            if st.session_state.nightmare_image_url.startswith("http"):
                st.image(st.session_state.nightmare_image_url, caption="악몽 시각화")
                with st.expander("생성 프롬프트 보기"):
                    st.write(st.session_state.nightmare_prompt)
            else:
                st.error(f"악몽 이미지 생성 실패: {st.session_state.nightmare_image_url}")

    with img_col2:
        if st.session_state.get('reconstructed_image_url'):
            if st.session_state.reconstructed_image_url.startswith("http"):
                st.image(st.session_state.reconstructed_image_url, caption="재구성된 꿈")
                with st.expander("생성 프롬프트 보기"):
                    st.write(st.session_state.reconstructed_prompt)
            else:
                st.error(f"재구성된 꿈 이미지 생성 실패: {st.session_state.reconstructed_image_url}")

