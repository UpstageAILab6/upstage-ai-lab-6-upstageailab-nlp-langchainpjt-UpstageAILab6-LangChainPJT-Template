import vector_store

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vector_store.vectorstore.as_retriever()

__all__ = ["retriever"]  # 외부에서 사용할 수 있도록 추가