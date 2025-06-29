# indexing_service.py 예시 코드
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS # 또는 Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def build_vector_store():
    # 1. 'data' 폴더의 모든 .txt, .md 파일 불러오기
    loader = DirectoryLoader('./data/', glob="**/*.md")
    documents = loader.load()

    # 2. 문서를 적절한 크기로 나누기
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # 3. OpenAI 임베딩 모델로 벡터화하기
    embeddings = OpenAIEmbeddings()

    # 4. FAISS (또는 Chroma) 벡터 저장소에 저장하기
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index") # 프로젝트 루트에 'faiss_index' 폴더 생성 및 저장
    print("벡터 스토어 생성이 완료되었습니다.")

if __name__ == '__main__':
    build_vector_store()