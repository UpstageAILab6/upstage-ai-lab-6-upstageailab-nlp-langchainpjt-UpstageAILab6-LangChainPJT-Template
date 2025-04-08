# modules/qa_engine.py

from langchain_upstage import ChatUpstage
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever


def build_qa_chain(retriever: BaseRetriever) -> RetrievalQA:
    """
    주어진 리트리버를 기반으로 LLM QA 체인을 생성합니다.

    Args:
        retriever (BaseRetriever): LangChain 리트리버 객체

    Returns:
        RetrievalQA: 검색 → 응답 생성까지 수행하는 QA 체인
    """
    llm = ChatUpstage()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
