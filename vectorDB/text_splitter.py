from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(all_docs, chunk_size=1000, chunk_overlap=250):
    """
    RecursiveCharacterTextSplitter를 사용하여 문서를 청크 단위로 분할합니다.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(all_docs)