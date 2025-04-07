# model/model.py
from data.pdf_processing import pdf_load_and_split
from embedding.embedding import initialize_embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def initialize_models():
    """
    embedding/embedding.py의 함수를 재사용하여 LLM과 임베딩 모델을 초기화합니다.
    """
    return initialize_embeddings()

def load_chain(pdf_source):
    """
    PDF 파일(또는 폴더)로부터 문서를 처리하고, 
    질문-답변 체인을 구성하여 반환합니다.
    
    Args:
        pdf_source (str): PDF 파일 경로나 폴더 경로.
        index_path (str): FAISS 인덱스 저장 경로.
    
    Returns:
        체인(chain) 객체
    """
    # PDF 처리 및 벡터스토어 구축
    _, _, _, retriever = pdf_load_and_split(pdf_source)
    
    # 모델 초기화
    llm, _ = initialize_models()
    
    # 프롬프트 설정
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Answer in Korean.

        #Context: 
        {context}

        #Question:
        {question}

        #Answer:"""
    )
    
    # 체인 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
