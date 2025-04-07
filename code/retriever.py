import numpy as np
from typing import List
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


class HybridMMRRetriever:
    def __init__(
        self,
        vectorstore: FAISS,
        embeddings: Embeddings,
        top_k_sim: int = 15,
        top_k_final: int = 5,
        lambda_mult: float = 0.5
    ):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.top_k_sim = top_k_sim
        self.top_k_final = top_k_final
        self.lambda_mult = lambda_mult

    def _maximal_marginal_relevance(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        k: int
    ) -> List[int]:
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        if isinstance(doc_embeddings, list):
            doc_embeddings = np.array(doc_embeddings)

        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        similarity_to_query = np.dot(doc_embeddings, query_embedding)
        similarity_between_docs = np.dot(doc_embeddings, doc_embeddings.T)

        selected = []
        remaining = list(range(len(doc_embeddings)))

        for _ in range(k):
            if not remaining:
                break

            if not selected:
                selected_idx = int(np.argmax(similarity_to_query))
                selected.append(selected_idx)
                remaining.remove(selected_idx)
                continue

            max_score = -np.inf
            selected_idx = -1

            for idx in remaining:
                sim_to_query = similarity_to_query[idx]
                sim_to_selected = max(similarity_between_docs[idx][j] for j in selected)
                score = self.lambda_mult * sim_to_query - (1 - self.lambda_mult) * sim_to_selected

                if score > max_score:
                    max_score = score
                    selected_idx = idx

            selected.append(selected_idx)
            remaining.remove(selected_idx)

        return selected

    def retrieve(
        self,
        question: str,
        top_k_sim: int = None,
        top_k_final: int = None,
        lambda_mult: float = None
    ) -> List[Document]:
        # 외부 입력값 우선, 없으면 인스턴스 기본값 사용
        top_k_sim = top_k_sim or self.top_k_sim
        top_k_final = top_k_final or self.top_k_final
        lambda_mult = lambda_mult or self.lambda_mult

        query_embedding = self.embeddings.embed_query(question)

        sim_docs_and_scores = self.vectorstore.similarity_search_with_score_by_vector(
            query_embedding,
            k=top_k_sim
        )

        docs = [doc for doc, _ in sim_docs_and_scores]
        doc_embeddings = [self.embeddings.embed_query(doc.page_content) for doc in docs]

        selected_indices = self._maximal_marginal_relevance(
            query_embedding=query_embedding,
            doc_embeddings=doc_embeddings,
            k=top_k_final
        )

        return [docs[i] for i in selected_indices]


# 체인 생성 함수 (similarity / mmr 등 공통 구조화용)
def create_retrieval_chain(retriever, prompt, llm, formatter=None):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    if formatter:
        chain |= RunnableLambda(formatter)
    return chain
