# vectorDB/store.py
import os
from langchain_community.vectorstores import FAISS

def get_vectorstore(documents, embeddings, index_path="./vectorDB/faiss_index"):
    """
    주어진 문서들을 기반으로 FAISS 벡터스토어를 생성합니다.
    만약 지정한 경로에 인덱스가 존재하면 로드하고, 없으면 새로 생성하여 저장합니다.
    """
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("기존 FAISS 인덱스를 로드했습니다")
    else:
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        vectorstore.save_local(index_path)
        print("새로운 FAISS 인덱스를 생성 후 저장했습니다")
    return vectorstore
