from dotenv import load_dotenv
load_dotenv(dotenv_path="/data/ephemeral/home/QA/.env")

from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings

TXT_PATH = "/data/ephemeral/home/QA/data/NationalTomorrowLearningCard_ParagraphBased.txt"
SAVE_DIR = "/data/ephemeral/home/QA/faiss_store"

with open(TXT_PATH, "r", encoding="utf-8") as f:
    paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]

embedding = UpstageEmbeddings(model="solar-embedding-1-large")

#  LangChain 방식으로 저장
vectorstore = FAISS.from_texts(texts=paragraphs, embedding=embedding)
vectorstore.save_local(SAVE_DIR)

print(f" 총 {len(paragraphs)}개의 문단을 저장했습니다.")
