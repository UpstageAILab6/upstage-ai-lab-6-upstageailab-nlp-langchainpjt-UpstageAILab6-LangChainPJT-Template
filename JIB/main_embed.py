from dotenv import load_dotenv
load_dotenv(dotenv_path="/data/ephemeral/home/QA/.env")

from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from modules.loader import load_paragraphs_from_txt_dir

TXT_DIR = "/data/ephemeral/home/QA/data"
SAVE_DIR = "/data/ephemeral/home/QA/faiss_store"

# 1. 문단 불러오기
paragraphs = load_paragraphs_from_txt_dir(TXT_DIR)

# 2. 업스테이지 임베딩 모델 로드
embedding = UpstageEmbeddings(model="solar-embedding-1-large")

# 3. LangChain 방식으로 벡터 저장소 생성 및 저장
vectorstore = FAISS.from_texts(texts=paragraphs, embedding=embedding)
vectorstore.save_local(SAVE_DIR)

print(f" 총 {len(paragraphs)}개의 문단을 저장했습니다.")
