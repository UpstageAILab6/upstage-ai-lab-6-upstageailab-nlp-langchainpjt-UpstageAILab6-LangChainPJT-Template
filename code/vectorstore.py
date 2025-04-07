import os
import time
from typing import List
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


class VectorStoreManager:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.vectorstore = None

    def embed_documents_in_batches(self, documents: List[Document], batch_size: int = 100):
        all_embeddings = []
        all_docs = []

        for i in tqdm(range(0, len(documents), batch_size), desc="임베딩 처리 중..."):
            batch = documents[i:i + batch_size]
            try:
                batch_embeddings = self.embeddings.embed_documents([doc.page_content for doc in batch])
            except Exception as e:
                print(f"임베딩 에러 발생. 3초 후 재시도 중... ({e})")
                time.sleep(3)
                batch_embeddings = self.embeddings.embed_documents([doc.page_content for doc in batch])

            all_embeddings.extend(batch_embeddings)
            all_docs.extend(batch)

        return all_docs, all_embeddings

    def create(self, documents: List[Document]) -> FAISS:
        docs, embeddings_list = self.embed_documents_in_batches(documents)
        self.vectorstore = FAISS.from_embeddings(
            embeddings=embeddings_list,
            texts=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs],
        )
        print("FAISS 벡터스토어 생성 완료")
        return self.vectorstore

    def save_local(self, path: str = "faiss_index_v2"):
        if self.vectorstore is None:
            raise ValueError("저장할 벡터스토어가 없습니다. 먼저 생성 또는 로드하세요.")
        self.vectorstore.save_local(path)
        print(f"벡터스토어 저장 완료 → {path}/")

    def load(self, path: str = "faiss_index_v2", allow_dangerous: bool = True) -> FAISS:
        self.vectorstore = FAISS.load_local(
            folder_path=path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=allow_dangerous
        )
        print(f"FAISS 벡터스토어 로드 완료 from {path}/")
        return self.vectorstore

    def load_or_create(self, documents: List[Document], path: str = "faiss_index_v2") -> FAISS:
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            return self.load(path)
        else:
            vs = self.create(documents)
            self.save_local(path)
            return vs

    def get_retriever(self) -> VectorStoreRetriever:
        if not self.vectorstore:
            raise ValueError("VectorStore가 먼저 생성 또는 로드되어야 합니다.")
        return self.vectorstore.as_retriever()
