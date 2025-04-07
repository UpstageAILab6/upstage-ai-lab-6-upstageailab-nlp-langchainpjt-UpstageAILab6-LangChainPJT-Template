from langchain_upstage import ChatUpstage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tavily import TavilyClient

def create_models(retriever, prompt, upstages_api_key, use_retriever_context=True):
    """
    검색기와 프롬프트를 기반으로 LLM 체인을 생성합니다.
    - use_retriever_context가 True이면 retriever를 호출해 문서 기반 context를 구성합니다.
    - False이면 retriever 호출 없이 질문과 히스토리만 프롬프트에 전달합니다.
    """
    llm = ChatUpstage(api_key=upstages_api_key, model="solar-pro", temperature=0)
    
    if use_retriever_context:
        def retrieve_and_format(inputs):
            question = str(inputs.get("question", "")).strip()
            history = str(inputs.get("history", "")).strip()
            # retriever를 통해 질문에 대한 관련 문서를 가져옵니다.
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            inputs["context"] = context
            inputs["question"] = question
            inputs["history"] = history
            return inputs

        chain = (
            RunnablePassthrough()
            | retrieve_and_format
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        # retriever context 단계를 생략하고, prompt는 context를 빈 문자열로 채웁니다.
        def add_empty_context(inputs):
            inputs["context"] = ""
            inputs["question"] = str(inputs.get("question", "")).strip()
            inputs["history"] = str(inputs.get("history", "")).strip()
            return inputs

        chain = (
            RunnablePassthrough()
            | add_empty_context
            | prompt
            | llm
            | StrOutputParser()
        )
    return chain

def create_tavily_search_chain(api_key):
    """
    TavilyClient를 활용하여 웹 검색 결과를 가져오고,
    결과를 깔끔한 마크다운 형식으로 반환하는 체인을 생성합니다.
    """
    client = TavilyClient(api_key)

    def search_and_format(inputs):
        question = str(inputs.get("question", "")).strip()
        # history = str(inputs.get("history", "")).strip()
        response = client.search(query=question)
        # response는 dict 형태라고 가정합니다.
        md = f"**검색어:** {response.get('query', '')}\n\n"
        md += f"**응답 시간:** {response.get('response_time', '')}초\n\n"
        results = response.get('results', [])
        if results:
            md += "**검색 결과:**\n\n"
            for res in results:
                title = res.get('title', '제목 없음')
                url = res.get('url', '')
                content = res.get('content', '')
                score = res.get('score', '')
                md += f"- **[{title}]({url})**\n"
                md += f"  - 내용: {content}\n"
                md += f"  - 점수: {score}\n\n"
        else:
            md += "검색 결과가 없습니다."
        return md

    chain = (
        RunnablePassthrough()
        | search_and_format
        | StrOutputParser()
    )
    return chain




# def create_tavily_search_chain(api_key):
#     """
#     TavilyClient를 활용하여 웹 검색 결과를 가져오는 체인을 생성
#     """
#     client = TavilyClient(api_key)

#     def search_and_format(inputs):
#         question = str(inputs.get("question", "")).strip()
#         history = str(inputs.get("history", "")).strip()
#         full_query = f"{history}\n{question}" if history else question
#         response = client.search(query=full_query)
#         # 결과를 문자열로 변환하여 반환 (dict 대신)
#         return str(response)

#     chain = (
#         RunnablePassthrough()
#         | search_and_format
#         | StrOutputParser()
#     )
#     return chain
