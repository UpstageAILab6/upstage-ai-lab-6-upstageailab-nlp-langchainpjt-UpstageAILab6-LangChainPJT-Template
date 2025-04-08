from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class DocsLoader(ABC):
    """
    로더의 공통 인터페이스를 정의한 추상 베이스 클래스입니다.
    """

    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        주어진 source(예: URL, Path)로부터 문서를 로드하여 Document 객체로 반환합니다.
        """
        pass
