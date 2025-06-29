import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_vector_store():
    """
    'data' 디렉토리의 .md 및 .txt 파일을 로드하고,
    텍스트를 분할하여 벡터화한 후 FAISS 벡터 스토어에 저장합니다.
    """
    # 시스템 환경변수에서 OpenAI API 키를 사용하는 것을 전제로 합니다.
    # 키가 설정되어 있는지 확인합니다.
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    print("데이터 로드를 시작합니다...")
    
    try:
        # .md 파일 로더
        md_loader = DirectoryLoader(
            './data/',
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        # .txt 파일 로더
        txt_loader = DirectoryLoader(
            './data/',
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        documents = md_loader.load()
        documents.extend(txt_loader.load())

    except Exception as e:
        print(f"❌ 데이터 로딩 중 오류가 발생했습니다: {e}")
        return

    if not documents:
        print("경고: 'data' 디렉토리에서 문서를 찾을 수 없습니다.")
        return

    print(f"총 {len(documents)}개의 문서를 불러왔습니다.")

    # 2. 문서를 의미 있는 단위(청크)로 나누기 (개선된 방식)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"문서를 총 {len(docs)}개의 청크로 나누었습니다.")

    # 3. 텍스트 분할 후 청크가 비어있는지 최종 확인 (오류 방지)
    if not docs:
        print("\n❌ 오류: 텍스트를 나눈 후 처리할 문서 조각(청크)이 없습니다.")
        print("   data 폴더의 .md 파일과 .txt 파일에 내용이 제대로 저장되어 있는지 확인해주세요.\n")
        return # 내용이 없으면 여기서 실행 중단

    print("임베딩 및 벡터 스토어 생성을 시작합니다...")
    
    try:
        # 4. OpenAI 임베딩 모델로 벡터화하기
        embeddings = OpenAIEmbeddings()

        # 5. FAISS 벡터 저장소에 저장하기
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index") # 프로젝트 루트에 'faiss_index' 폴더 생성 및 저장
    
        print("\n✅ 벡터 스토어 생성이 완료되었습니다. 'faiss_index' 폴더가 생성되었습니다.")
    
    except Exception as e:
        print(f"❌ 임베딩 또는 벡터 스토어 생성 중 오류가 발생했습니다: {e}")
        print("   OpenAI API 키가 유효한지, 인터넷 연결에 문제가 없는지 확인해주세요.")


# 이 스크립트가 직접 실행될 때만 build_vector_store 함수를 호출합니다.
if __name__ == '__main__':
    build_vector_store()