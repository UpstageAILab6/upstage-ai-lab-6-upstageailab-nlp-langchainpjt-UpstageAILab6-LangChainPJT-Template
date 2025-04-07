import os
from .config import FAISS_INDEX_DIR, FAISS_INDEX_SEMANTIC_DIR

def vectorstore(embeddings, documents, index_dir=FAISS_INDEX_DIR):
    """
    주어진 문서를 기반으로 FAISS 벡터스토어를 생성합니다.
    이미 저장된 인덱스가 있으면 불러오고, 없으면 새로 생성합니다.
    """
    from langchain_community.vectorstores import FAISS
    # embed_documents 메서드를 callable로 전달
    if os.path.exists(index_dir):
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(documents=documents, embedding=embeddings)
        vs.save_local(index_dir)
    return vs

def vectorstore_semantic(embeddings, documents, index_dir=FAISS_INDEX_SEMANTIC_DIR):
    """
    시멘틱 청크를 기반으로 FAISS 벡터스토어를 생성합니다.
    이미 저장된 인덱스가 있으면 불러오고, 없으면 새로 생성합니다.
    """
    from langchain_community.vectorstores import FAISS
    if os.path.exists(index_dir):
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(documents=documents, embedding=embeddings)
        vs.save_local(index_dir)
    return vs