# embedder.py (Upstage 버전 with .env 경로 설정)

import os
from dotenv import load_dotenv
from typing import List
from langchain_upstage import UpstageEmbeddings
import numpy as np
import faiss
import pickle

# ✅ .env 경로 명시적으로 지정
load_dotenv(dotenv_path="/data/ephemeral/home/QA/.env")

# 1. Upstage 임베딩 모델 로드
model = UpstageEmbeddings(model="solar-embedding-1-large")

# 2. 문단(청크) 리스트를 벡터로 임베딩
def embed_chunks(chunks: List[str]) -> np.ndarray:
    embeddings = model.embed_documents(chunks)
    return np.array(embeddings)

# 3. FAISS 벡터 저장소에 저장
def save_embeddings_to_faiss(embeddings: np.ndarray, chunks: List[str], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(save_dir, "faiss.index"))
    with open(os.path.join(save_dir, "texts.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print(f" FAISS 저장소 및 원문 저장 완료: {save_dir}")

# 4. 검색 함수
def load_faiss_and_search(query: str, top_k: int = 5):
    index = faiss.read_index("/data/ephemeral/home/QA/faiss_store/faiss.index")
    with open("/data/ephemeral/home/QA/faiss_store/texts.pkl", "rb") as f:
        texts = pickle.load(f)
    query_vec = np.array([model.embed_query(query)])
    D, I = index.search(query_vec, top_k)
    return [texts[i] for i in I[0]]