import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import concurrent.futures # 이미지 동시 생성을 위해 추가

# 우리가 만든 모든 서비스들을 가져옵니다.
# (사용자님의 현재 구조에 맞게 'core.' 접두사를 제거했습니다.)
from services.dream_analyzer_service import DreamAnalyzerService
from services.report_generator_service import ReportGeneratorService
from services.image_generator_service import ImageGeneratorService

# --- 1. 페이지 설정 및 서비스 초기화 ---
st.set_page_config(page_title="보여DREAM", page_icon="🌙")

@st.cache_resource # 서비스 및 모델 객체를 캐싱하여 앱 성능 향상
def initialize_services():
    """
    API 키 확인, 모든 서비스 및 모델 객체들을 생성하고 캐싱합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        st.stop()
    
    try:
        # 1. FAISS 인덱스와 RAG 검색기(retriever)를 먼저 준비합니다.
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
        
        # 2. 준비된 retriever를 ReportGeneratorService에 전달하며 객체를 생성합니다.
        report_generator = ReportGeneratorService(api_key=api_key, retriever=retriever)
        
        # 3. 다른 서비스들도 생성합니다.
        dream_analyzer = DreamAnalyzerService(api_key=api_key)
        image_generator = ImageGeneratorService(api_key=api_key)
        
        # 생성된 모든 객체들을 반환합니다.
        return report_generator, dream_analyzer, image_generator

    except Exception as e:
        st.error(f"서비스 초기화 중 오류가 발생했습니다: {e}")
        st.info("core/indexing_service.py를 먼저 실행했는지, faiss_index 폴더가 있는지 확인해주세요.")
        st.stop()

# --- 2. 메인 앱 실행 ---
def main():
    st.title("보여DREAM 🌙")
    st.write("당신의 꿈 이야기를 들려주세요. AI가 악몽을 분석하고 긍정적인 이미지로 재구성해 드립니다.")
    
    st.session_state.report_generator, st.session_state.dream_analyzer, st.session_state.image_generator = initialize_services()

    # TODO: 여기에 음성 입력(파일 업로드, 녹음) UI 추가 가능
    dream_text = st.text_area("어젯밤 어떤 꿈을 꾸셨나요?", height=200, placeholder="여기에 꿈 내용을 자세히 적어주세요...")

    if st.button("분석 및 재구성 시작하기", type="primary", use_container_width=True):
        if dream_text:
            # 1. 리포트 및 프롬프트 생성
            with st.spinner("RAG가 지식 베이스를 참조하여 꿈을 심층 분석 중입니다..."):
                dream_report = st.session_state.report_generator.generate_report_with_rag(dream_text)
                nightmare_prompt = st.session_state.dream_analyzer.create_nightmare_prompt(dream_text)
                reconstructed_prompt, summary, mappings = st.session_state.dream_analyzer.create_reconstructed_prompt_and_analysis(dream_text, dream_report)

            # 2. 이미지 생성 (동시 실행으로 속도 향상)
            with st.spinner("DALL-E 3가 꿈을 이미지로 그리고 있습니다... (1분 정도 소요될 수 있습니다)"):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_nightmare = executor.submit(st.session_state.image_generator.generate_image_from_prompt, nightmare_prompt)
                    future_reconstructed = executor.submit(st.session_state.image_generator.generate_image_from_prompt, reconstructed_prompt)
                    
                    nightmare_image_url = future_nightmare.result()
                    reconstructed_image_url = future_reconstructed.result()

            # 3. 모든 결과 출력
            st.subheader("📝 AI 심층 분석 리포트")
            with st.container(border=True):
                st.markdown("##### 심층 분석 요약")
                st.write(dream_report.get("analysis_summary", "요약 정보 없음"))
                
                st.markdown("##### 주요 감정")
                emotions = dream_report.get("emotions", [])
                if emotions:
                    for emo in emotions:
                        st.progress(emo['score'], text=f"{emo['emotion']} ({int(emo['score']*100)}%)")
                
                st.markdown("##### 핵심 키워드")
                keywords = dream_report.get("keywords", [])
                if keywords:
                    st.write(" &nbsp; ".join(f"`{kw}`" for kw in keywords))

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("악몽의 시각화 (Before)")
                if nightmare_image_url.startswith("http"):
                    st.image(nightmare_image_url, caption="AI가 그린 당신의 악몽")
                else:
                    st.error(f"이미지 생성 실패: {nightmare_image_url}")
            with col2:
                st.subheader("재구성된 꿈 (After)")
                if reconstructed_image_url.startswith("http"):
                    st.image(reconstructed_image_url, caption="AI가 긍정적으로 재구성한 꿈")
                else:
                    st.error(f"이미지 생성 실패: {reconstructed_image_url}")
            
            st.divider()
            st.subheader("✨ 이렇게 바뀌었어요!")
            st.write(summary)
            for mapping in mappings:
                st.markdown(f"- `{mapping['original']}` &nbsp; ➡️ &nbsp; **`{mapping['transformed']}`**")
        else:
            st.warning("꿈 내용을 먼저 입력해주세요.")

if __name__ == "__main__":
    main()