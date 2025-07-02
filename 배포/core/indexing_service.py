import os # 운영체제 기능 제공
from langchain_community.document_loaders import (
    DirectoryLoader, # 디렉토리에서 문서 로드
    TextLoader, # 텍스트 파일 로드
)
from langchain_community.vectorstores import FAISS # FAISS 벡터 스토어 사용
from langchain_openai import OpenAIEmbeddings # OpenAI 임베딩 사용
from langchain.text_splitter import RecursiveCharacterTextSplitter # 텍스트 재귀 분할

def build_vector_store():
    """
    'data' 디렉토리의 .md 및 .txt 파일을 로드하고,
    텍스트를 분할하여 벡터화한 후 FAISS 벡터 스토어에 저장합니다.
    """
    # OpenAI API 키 환경변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    print("데이터 로드를 시작합니다...")
    
    try:
        # .md 파일 로더 설정
        md_loader = DirectoryLoader(
            './data/',
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        # .txt 파일 로더 설정
        txt_loader = DirectoryLoader(
            './data/',
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        # 문서 로드 및 합치기
        documents = md_loader.load()
        documents.extend(txt_loader.load())

    except Exception as e:
        print(f"❌ 데이터 로딩 중 오류가 발생했습니다: {e}")
        return

    # 로드된 문서가 없는 경우 경고
    if not documents:
        print("경고: 'data' 디렉토리에서 문서를 찾을 수 없습니다.")
        return

    print(f"총 {len(documents)}개의 문서를 불러왔습니다.")

    # 문서를 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"문서를 총 {len(docs)}개의 청크로 나누었습니다.")

    # 분할된 청크가 없는 경우 오류
    if not docs:
        print("\n❌ 오류: 텍스트를 나눈 후 처리할 문서 조각(청크)이 없습니다.")
        print("   data 폴더의 .md 파일과 .txt 파일에 내용이 제대로 저장되어 있는지 확인해주세요.\n")
        return

    print("임베딩 및 벡터 스토어 생성을 시작합니다...")
    
    try:
        # OpenAI 임베딩 모델 초기화
        embeddings = OpenAIEmbeddings()

        # FAISS 벡터 저장소에 문서와 임베딩 저장
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index") # 로컬에 인덱스 저장
    
        print("\n✅ 벡터 스토어 생성이 완료되었습니다. 'faiss_index' 폴더가 생성되었습니다.")
    
    except Exception as e:
        print(f"❌ 임베딩 또는 벡터 스토어 생성 중 오류가 발생했습니다: {e}")
        print("   OpenAI API 키가 유효한지, 인터넷 연결에 문제가 없는지 확인해주세요.")

# 스크립트 직접 실행 시 build_vector_store 함수 호출
if __name__ == '__main__':
    build_vector_store()