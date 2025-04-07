from typing import Any

class GovPolicyQA:
    def __init__(self, vectorstore, embeddings, llm, prompt, formatter):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = llm
        self.prompt = prompt
        self.formatter = formatter

    def run(self, question: str) -> str:
        from retriever import HybridMMRRetriever  # 내부에서 불러오는 방식

        # Step 1: Retrieve documents using Hybrid MMR
        retriever = HybridMMRRetriever(self.vectorstore, self.embeddings)
        docs = retriever.retrieve(question, top_k_sim=15, top_k_final=5, lambda_mult=0.7)

        # Step 2: Construct context from documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 3: Format prompt and invoke LLM
        formatted_prompt = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, "content") else response

        # Step 4: Postprocess the answer to Markdown format
        return self.formatter.format(answer)
