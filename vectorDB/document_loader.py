import os
import glob

def load_documents(data_folder="data/"):
    """
    data_folder 내의 모든 PDF 파일을 로드하여 문서 리스트를 반환합니다.
    """
    pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
    all_docs = []

    for pdf_file in pdf_files:
        # PyMuPDFLoader를 사용해 PDF 문서를 로드합니다.
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(pdf_file)
        docs = loader.load()
        all_docs.extend(docs)
    return pdf_files, all_docs