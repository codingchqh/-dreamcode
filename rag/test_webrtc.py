# test_webrtc.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.title("WebRTC 라이브러리 단독 테스트")

try:
    webrtc_streamer(
        key="test-component",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": False, "audio": True},
    )
    st.success("WebRTC 컴포넌트가 성공적으로 로드되었습니다.")
    st.write("웹 화면에 START 버튼이 보이면 정상입니다.")

except Exception as e:
    st.error("WebRTC 컴포넌트 로드 중 오류가 발생했습니다.")
    st.exception(e)