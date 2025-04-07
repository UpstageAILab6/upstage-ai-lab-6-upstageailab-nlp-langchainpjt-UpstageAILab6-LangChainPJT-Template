# embedding/embedding.py
import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage, UpstageEmbeddings

def get_upstage_api_key():
    """
    .env 파일에서 UPSTAGE_API_KEY를 로드합니다.
    키가 없으면 에러를 발생시킵니다.
    """
    load_dotenv()
    token = os.environ.get("UPSTAGE_API_KEY")
    if not token:
        raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다.")
    return token

def initialize_embeddings():
    """
    LLM과 임베딩 모델을 초기화합니다.
    """
    model_name = "solar-pro"
    embedding_model = "solar-embedding-1-large-query"
    api_key = get_upstage_api_key()

    # LLM 초기화
    llm = ChatUpstage(
        model_name=model_name,
        temperature=0.01,
        api_key=api_key,
    )
    
    # 임베딩 모델 초기화
    embeddings = UpstageEmbeddings(
        model=embedding_model,
        api_key=api_key,
    )
    return llm, embeddings
