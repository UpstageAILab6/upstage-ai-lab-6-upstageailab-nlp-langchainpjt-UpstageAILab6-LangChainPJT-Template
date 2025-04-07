from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings

class Embedding:
    """
    Upstage Embeddings wrapper for Langchain.
    """

    def __init__(self):
        self.model = UpstageEmbeddings(model="solar-embedding-1-large")

    def embed_documents(self, texts):
        # Implement the logic to embed documents using Upstage API
        pass

    def embed_query(self, text):
        # Implement the logic to embed a query using Upstage API
        pass

def load_vector_store_once(store_path):
    """
    FAISS 벡터스토어를 한 번 로드하여 반환합니다.
    """
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    # vector_store_instance = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

    return Chroma(
        embedding_function=embeddings,
        persist_directory=store_path,
        collection_name="chroma",
    )

