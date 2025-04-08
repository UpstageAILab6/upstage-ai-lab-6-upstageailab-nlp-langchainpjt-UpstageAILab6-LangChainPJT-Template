import os
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings

class VectorDBStore:

    load_dotenv()

    #API 키 가져오기
    api_key = os.getenv("UPSTAGE_API_KEY")

    def __init__(self, html_file_path):
        self.html_file_path = html_file_path
        self.text_file_path = html_file_path.replace(".html", ".txt")
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large", api_key=self.api_key)
        self.docs = []  # 문서 저장 리스트 초기화
        self.preprocessed_docs = []  # 전처리된 문서 리스트 초기화
        self.split_documents = []  # 분할된 문서 저장
        self.vectorstore = None  # FAISS 저장소

    def load_html(self) -> None:
        """HTML 파일을 읽고 텍스트로 변환하여 저장"""
        with open(self.html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        with open(self.text_file_path, "w", encoding="utf-8") as f:
            f.write(soup.prettify())

    def load_documents(self) -> None:
        """텍스트 문서를 로드"""
        loader = TextLoader(self.text_file_path)
        self.docs = loader.load()

    def mdtrans(self) -> None:
        """Markdown 변환"""
        md = MarkdownifyTransformer()
        self.docs = md.transform_documents(self.docs)

    def preprocess_text(self, text) -> str:
        """텍스트 전처리: 불필요한 태그 및 정보 제거"""
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # 이미지 태깅 제거
        text = re.sub(r'\n\n\d+(\.\d+)?(KB|MB|GB)', '', text)  # 파일 크기 제거
        return text

    def split_preprocess_documents(self) -> None:
        """문서 전처리 수행"""
        self.preprocessed_docs = []
        for doc in self.docs:
            preprocessed_content = self.preprocess_text(doc.page_content)
            preprocessed_doc = Document(page_content=preprocessed_content, metadata=doc.metadata)
            self.preprocessed_docs.append(preprocessed_doc)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=50)
            self.split_documents = text_splitter.split_documents(self.preprocessed_docs)

    '''
    def split_documents(self):
        """문서를 작은 청크로 나눔"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=50)
        self.split_documents = text_splitter.split_documents(self.preprocessed_docs)
    '''
    def create_vectordb(self) -> FAISS:
        """FAISS 벡터DB 생성"""
        if not self.split_documents:
            raise ValueError("문서가 분할되지 않았습니다. 먼저 split_documents()를 실행하세요.")

        self.vectorstore = FAISS.from_documents(documents=self.split_documents, embedding=self.embeddings)
        return self.vectorstore
