import numpy as np
from typing import List, Dict
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from sentence_splitter import split_text_into_sentences

class VectorstoreGroundingChecker:
    def __init__(self, vectorstore: VectorStore, embeddings: Embeddings, threshold: float = 0.75):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.threshold = threshold

    def run(self, answer: str) -> List[Dict]:
        sentences = split_text_into_sentences(answer)
        results = []

        for sentence in sentences:
            sent_embedding = self.embeddings.embed_query(sentence)
            # 유사한 문서 1개 검색
            top_k_docs = self.vectorstore.similarity_search_with_score_by_vector(sent_embedding, k=1)
            if not top_k_docs:
                continue

            top_doc, score = top_k_docs[0]
            similarity = 1 - score  # FAISS는 distance, 유사도로 변환

            results.append({
                "sentence": sentence,
                "matched_context": top_doc.page_content,
                "similarity": round(similarity, 3),
                "grounded": similarity >= self.threshold
            })

        return results