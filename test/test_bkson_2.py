
import os
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_teddynote import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import ChatUpstage, UpstageEmbeddings


def get_upstage_api_key():
    # 환경변수에서 API키 가져오는 함수
    load_dotenv()
    token = os.environ.get("UPSTAGE_API_KEY")
    if not token:
        raise ValueError("UPSTAGE_API_KEY 없음")
    return token
# get_upstage_api_key()


def initialize_models():
    model_name = "solar-mini"
    embedding_model = "solar-embedding-1-large-query"

    # OpenAI API 키 가져오기
    upstages_api_key = get_upstage_api_key()
    
    # Upstages 임베딩 모델
    llm = ChatUpstage(
        model_name=model_name,
        temperature=0.01,
        api_key=upstages_api_key,
    )
    embeddings = UpstageEmbeddings(
        model=embedding_model,
        api_key=upstages_api_key,
    )   
    return llm, embeddings


# Crawler 적용가능???, Reader(?)
def pdf_load_and_split(pdf_path, chunk_size=512, chunk_overlap=50):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    # print(f"문서의 페이지수: {len(docs)}")

    docs = loader.load()
    text_chunks = []

    for i, doc in enumerate(docs):
        # text = docs[page_num].page_content
        if doc.page_content.strip():                                    # 빈 페이지 제외        
            text_chunks.append(f"[페이지 {i+1}] {doc.page_content}")

    # 문서 분할 (청킹)
    text_splitter  = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    split_documents = text_splitter.split_documents(docs)
    # print(f"분할된 청크의 수: {len(split_documents)}")

    # 임베딩 모델을 통해 벡터스토어 구축하기
    _, embeddings = initialize_models()

    # FAISS 인덱스 저장 경로 설정하기
    index_path = "./faiss_index"
    
    # 인덱스가 존재하면 로드해서 불러오기, 없으면 구축 후 저장 (리소스 절약?)
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("기존 FAISS 인덱스를 로드했습니다")
    else:
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore.save_local(index_path)
        print("새로운 FAISS 인덱스를 생성 후 저장했습니다")

    # vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    return text_chunks, split_documents, vectorstore, retriever


def main():
    pdf_path = "24_25_laws_of_the_game_01.pdf"
    if not os.path.exists(pdf_path):
        print("No PDF Data")
        return
    
    text_chunks, split_documents, vectorstore, retriever = pdf_load_and_split(pdf_path)

    llm, _ = initialize_models()

    # 프롬프트 생성
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

    # Chain 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("알고 싶은 경기 규칙을 입력하세요...")
    while True:
        question = input("질문 (종료하려면 '종료' 입력): ").strip()
        if question.lower() in ['종료', 'exit', 'quit']:
            print("프로그램을 종료합니다.")
            break

        # 체인을 실행하여 답변 생성
        answer = chain.invoke(question)
        print("답변:")
        print(answer)


if __name__ == "__main__":
    main()


