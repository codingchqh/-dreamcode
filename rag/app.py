import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 우리가 만든 모든 서비스들을 가져옵니다.
from services.dream_analyzer_service import DreamAnalyzerService
from services.report_generator_service import ReportGeneratorService
# from core.services.image_generator_service import ImageGeneratorService # TODO: 마지막 단계

# --- 페이지 설정 및 서비스 초기화 ---
st.set_page_config(page_title="보여DREAM", page_icon="🌙")

@st.cache_resource # 서비스 및 모델 객체를 캐싱하여 앱 성능 향상
def initialize_services():
    """
    API 키 확인, 모든 서비스 및 모델 객체들을 생성하고 캐싱합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. 앱을 실행하기 전에 설정해주세요.")
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
        # image_generator = ImageGeneratorService(api_key=api_key) # TODO
        
        # 생성된 모든 객체들을 반환합니다.
        return report_generator, dream_analyzer

    except Exception as e:
        st.error(f"서비스 초기화 중 오류가 발생했습니다: {e}")
        st.info("core/indexing_service.py를 먼저 실행했는지, faiss_index 폴더가 있는지 확인해주세요.")
        st.stop()

# --- 메인 앱 실행 ---
def main():
    st.title("보여DREAM 🌙")
    st.write("당신의 꿈 이야기를 들려주세요. AI가 악몽을 분석하고 긍정적인 이미지로 재구성해 드립니다.")
    
    # 서비스 객체들을 초기화하고 세션 상태에 저장합니다.
    st.session_state.report_generator, st.session_state.dream_analyzer = initialize_services()

    # --- UI 구성 ---
    # TODO: 여기에 음성 입력(파일 업로드, 녹음) UI 추가 가능
    dream_text = st.text_area("어젯밤 어떤 꿈을 꾸셨나요?", height=200, placeholder="여기에 꿈 내용을 자세히 적어주세요...")

    if st.button("분석 및 재구성 시작하기", type="primary", use_container_width=True):
        if dream_text:
            with st.spinner("RAG가 지식 베이스를 참조하여 꿈을 심층 분석 중입니다..."):
                try:
                    # --- 1. RAG 기반 심층 분석 리포트 생성 ---
                    dream_report = st.session_state.report_generator.generate_report_with_rag(dream_text)
                    
                    st.subheader("📝 AI 심층 분석 리포트")
                    with st.container(border=True):
                        st.markdown("##### 심층 분석 요약")
                        st.write(dream_report.get("analysis_summary", "요약 정보를 가져올 수 없습니다."))
                        
                        st.markdown("##### 주요 감정")
                        emotions = dream_report.get("emotions", [])
                        if emotions:
                            for emo in emotions:
                                st.progress(emo['score'], text=f"{emo['emotion']} ({int(emo['score']*100)}%)")
                        
                        st.markdown("##### 핵심 키워드")
                        keywords = dream_report.get("keywords", [])
                        if keywords:
                            # 키워드를 보기 좋게 나열합니다.
                            st.write(" &nbsp; ".join(f"`{kw}`" for kw in keywords))

                    st.divider()

                    # --- 2. 리포트를 바탕으로 Before/After 프롬프트 생성 ---
                    nightmare_prompt = st.session_state.dream_analyzer.create_nightmare_prompt(dream_text)
                    reconstructed_prompt, summary, mappings = st.session_state.dream_analyzer.create_reconstructed_prompt_and_analysis(dream_text, dream_report)

                    # --- 3. 결과 분할하여 보여주기 ---
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("악몽의 시각화 (Before)")
                        st.info(f"**프롬프트:** {nightmare_prompt}")
                        st.warning("TODO: 여기에 '악몽 이미지'가 표시됩니다.")
                    with col2:
                        st.subheader("재구성된 꿈 (After)")
                        st.success(f"**프롬프트:** {reconstructed_prompt}")
                        st.warning("TODO: 여기에 '재구성된 이미지'가 표시됩니다.")
                    
                    st.divider()
                    st.subheader("✨ 이렇게 바뀌었어요!")
                    st.write(summary)
                    for mapping in mappings:
                        st.markdown(f"- `{mapping['original']}` &nbsp; ➡️ &nbsp; **`{mapping['transformed']}`**")

                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {e}")
        else:
            st.warning("꿈 내용을 먼저 입력해주세요.")

if __name__ == "__main__":
    main()