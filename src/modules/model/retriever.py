
import os

from langchain_core.vectorstores import VectorStoreRetriever

from src.modules.model.vectorstore import VectorDBStore

# HTML 파일 경로 지정
# todo 환경에 따라 다름
html_path = '../files/page.html' #os.path.join(os.path.dirname(os.getcwd()), "data/files/page.html")

# VectorDBStore 인스턴스 생성
vectordb_store = VectorDBStore(html_path)

#vectorDB 생성
vectordb_store.load_html()
vectordb_store.load_documents()
vectordb_store.mdtrans()
vectordb_store.split_preprocess_documents()
vectorstore = vectordb_store.create_vectordb()

# retriever 생성(최대 4개의 검색 결과 반환)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#     else:
def get_retriever() -> VectorStoreRetriever:
    """retriever 객체를 반환하는 함수"""
    return retriever
