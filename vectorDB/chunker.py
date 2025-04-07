import os
import pickle

def semantic_chunker(embeddings, documents, chunk_size, chunk_overlap, cache_path="semantic_chunks.pkl"):
    """
    LangChain의 SemanticChunker를 사용해 문서를 의미 단위로 재구성합니다.
    캐시가 존재하고 현재의 chunk_size, chunk_overlap과 동일하면 캐시된 결과를 사용합니다.
    """
    regenerate = True
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("chunk_size") == chunk_size and cache.get("chunk_overlap") == chunk_overlap:
            semantic_chunks = cache.get("semantic_chunks")
            regenerate = False

    if regenerate:
        from langchain_experimental.text_splitter import SemanticChunker
        chunker = SemanticChunker(embeddings=embeddings)
        semantic_chunks = chunker.split_documents(documents)
        cache = {
            "semantic_chunks": semantic_chunks,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    return semantic_chunks
