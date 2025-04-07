from langchain_upstage import UpstageEmbeddings

def embedding(upstages_api_key, model="embedding-query", base_url="https://api.upstage.ai/v1"):
    """
    UpstageEmbeddings 인스턴스를 생성하여 반환합니다.
    """
    return UpstageEmbeddings(api_key=upstages_api_key, model=model, base_url=base_url)
