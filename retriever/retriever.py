def retriever(vectorstore):
    """
    벡터스토어로부터 검색기를 생성합니다.
    """
    return vectorstore.as_retriever()