# retriever/retriever.py

def get_retriever(vectorstore, **kwargs):
    """
    주어진 vectorstore를 기반으로 retriever 객체를 생성합니다.
    
    추가적인 설정이 필요한 경우 kwargs로 전달할 수 있습니다.
    """
    return vectorstore.as_retriever(**kwargs)
