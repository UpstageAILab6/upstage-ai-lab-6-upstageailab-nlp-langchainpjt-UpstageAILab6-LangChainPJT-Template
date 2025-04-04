from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS


class VectorDB(ABC):
    @abstractmethod
    def create_store(self, docs: List[Document]):
        """문서 리스트를 사용하여 벡터 스토어를 생성합니다."""
        pass

    @abstractmethod
    def add_documents(self, docs: List[Document]):
        """기존 벡터 스토어에 문서를 추가합니다."""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """주어진 쿼리와 유사한 문서를 검색합니다."""
        pass


class FaissDB(VectorDB):
    def __init__(self, embeddings, persist_directory: str = None):
        """
        FAISS 기반 벡터 스토어 구현체입니다.

        Args:
            embeddings: 외부에서 주입된 임베딩 인스턴스.
            persist_directory (str, optional): 벡터 스토어의 상태를 저장할 로컬 경로.
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None

        # persist_directory가 주어졌으면 저장된 인덱스가 있는지 확인 후 로드합니다.
        if self.persist_directory:
            try:
                print(f"'{self.persist_directory}'에 저장된 벡터 스토어를 로드합니다. ", end="\n")
                self.vectorstore = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)
                print("로컬에서 벡터 스토어를 불러왔습니다.")
            except Exception as e:
                print("저장된 벡터 스토어가 없습니다. 새로 생성합니다.")

    def create_store(self, docs: List[Document]):
        """
        주어진 문서 리스트를 사용하여 FAISS 벡터 스토어를 생성합니다.

        Args:
            docs (List[Document]): 문서 객체 리스트.
        """
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        if self.persist_directory:
            self.vectorstore.save_local(self.persist_directory)
            print(f"벡터 스토어를 '{self.persist_directory}'에 저장했습니다.")

    def add_documents(self, docs: List[Document]):
        """
        벡터 스토어에 문서를 추가합니다. 아직 생성되지 않았다면, 먼저 벡터 스토어를 생성합니다.

        Args:
            docs (List[Document]): 문서 객체 리스트.
        """
        if self.vectorstore is None:
            self.create_store(docs)
        else:
            self.vectorstore.add_documents(docs)
            if self.persist_directory:
                self.vectorstore.save_local(self.persist_directory)
                print(f"업데이트된 벡터 스토어를 '{self.persist_directory}'에 저장했습니다.")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        주어진 쿼리와 유사한 문서를 검색합니다.

        Args:
            query (str): 검색할 쿼리 문자열.
            k (int, optional): 반환할 문서 개수 (기본값: 4).

        Returns:
            List[Document]: 유사도 순으로 정렬된 문서 객체 리스트.
        """
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다. 먼저 문서를 추가해주세요.")
        return self.vectorstore.similarity_search(query, k=k)