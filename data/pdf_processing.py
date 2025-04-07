# data/pdf_processing.py
import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding.embedding import initialize_embeddings  # LLM과 임베딩 초기화를 위한 함수
from vectorDB.store import get_vectorstore             # FAISS 벡터스토어 생성/로드 함수
from retriever.retriever import get_retriever            # 분리된 retriever 생성 함수

def pdf_load_and_split(pdf_source, chunk_size=512, chunk_overlap=50):
    """
    단일 PDF 파일 혹은 PDF 폴더를 입력받아 문서를 로드하고,
    텍스트를 청킹한 후, 임베딩 및 FAISS 벡터스토어를 생성하고 retriever를 반환합니다.
    
    Args:
        pdf_source (str): PDF 파일 경로나 PDF가 포함된 폴더 경로.
        chunk_size (int): 청크 크기.
        chunk_overlap (int): 청크 간 중복 분량.
    
    Returns:
        text_chunks: 페이지별 원본 텍스트 리스트.
        split_documents: 청킹된 문서 리스트.
        vectorstore: 생성된(또는 로드한) FAISS 벡터스토어.
        retriever: 벡터스토어로부터 생성된 retriever 객체.
    """
    all_docs = []
    text_chunks = []
    
    # pdf_source가 폴더인 경우 폴더 내 모든 PDF 파일 처리
    if os.path.isdir(pdf_source):
        pdf_paths = glob.glob(os.path.join(pdf_source, "*.pdf"))
    else:
        pdf_paths = [pdf_source]
    
    if not pdf_paths:
        raise FileNotFoundError("지정된 경로에서 PDF 파일을 찾을 수 없습니다.")
    
    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        for i, doc in enumerate(docs):
            if doc.page_content.strip():
                text_chunks.append(f"[페이지 {i+1}] {doc.page_content}")
        all_docs.extend(docs)
    
    # 문서 분할 (청킹)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    split_documents = text_splitter.split_documents(all_docs)
    
    # 임베딩 초기화
    _, embeddings = initialize_embeddings()
    
    # FAISS 벡터스토어 생성 또는 로드
    vectorstore = get_vectorstore(split_documents, embeddings)
    
    # retriever 생성은 분리된 모듈에서 처리
    retriever = get_retriever(vectorstore)
    
    return text_chunks, split_documents, vectorstore, retriever
