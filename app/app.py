import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import streamlit as st
from dotenv import load_dotenv

# 모듈 임포트
from vectorDB.document_loader import load_documents
from vectorDB.text_splitter import split_documents
from embedding.embeddings import embedding
from vectorDB.chunker import semantic_chunker
from vectorDB.vectorstore import vectorstore, vectorstore_semantic
from retriever.retriever import retriever
from model.chain import create_models, create_tavily_search_chain

from langchain_core.prompts import PromptTemplate
from langchain_upstage import UpstageGroundednessCheck
from langchain_community.retrievers import TavilySearchAPIRetriever

def initialize_session():
    if "rag_state" not in st.session_state:
        st.session_state.rag_state = False
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []  # 초기화 추가

def setup_page():
    st.set_page_config(layout="wide")
    st.title("LangChain Project")

def sidebar_settings():
    st.subheader("Settings")
    st.markdown("---")
    st.session_state.chunk_size = st.number_input("chunk_size", value=1000, step=50)
    st.session_state.chunk_overlap = st.number_input("chunk_overlap", value=250, step=50)
    st.session_state.method = st.radio("답변 생성 방식 선택", ("임베딩 기반", "시멘틱 청커 기반"))

    default_prompt = (
        "너는 스포츠 규칙 전문가야.\n"
        "너는 경기 규칙에 대해서 설명해줘야해.\n"
        "너는 답변을 할 때, 반드시 종목을 확인해서 답변해줘야해.\n"
        "너는 답변을 할 때, 반드시 다음과 같은 형식으로 답변해줘야해.\n"
        "1.\n2.\n3.\n\n"
        "#Chat History:\n{history}\n"
        "#Question:\n{question}\n"
        "#Context:\n{context}\n\n"
        "#Answer:"
    )
    st.session_state.new_prompt = st.text_area("프롬프트를 작성해주세요!", value=default_prompt, height=350, key="prompt_write")
    if st.button("실행"):
        st.session_state.rag_state = True

def run_pipeline():
    st.session_state.pdf_files, st.session_state.all_docs = load_documents()
    st.session_state.split_documents = split_documents(
        all_docs=st.session_state.all_docs,
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
    )
    st.session_state.embeddings = embedding(upstages_api_key=st.session_state.upstages_api_key)

    if st.session_state.method == "임베딩 기반":
        st.session_state.vectorstore = vectorstore(st.session_state.embeddings, st.session_state.split_documents)
        st.session_state.retriever = retriever(st.session_state.vectorstore)
    else:  # 시멘틱 청커 기반
        if "semantic_chunks" not in st.session_state:
            st.session_state.semantic_chunks = semantic_chunker(
                st.session_state.embeddings,
                st.session_state.split_documents,
                st.session_state.chunk_size,
                st.session_state.chunk_overlap,
            )
        st.session_state.vector_store_instance = vectorstore_semantic(
            st.session_state.embeddings, st.session_state.semantic_chunks
        )
        st.session_state.retriever = retriever(st.session_state.vector_store_instance)

    st.session_state.prompt = PromptTemplate.from_template(st.session_state.new_prompt)
    st.session_state.chain = create_models(
        st.session_state.retriever,
        st.session_state.prompt,
        st.session_state.upstages_api_key,
    )

def render_result():
    st.subheader("Result")
    st.markdown("---")
    st.write(f"➡️ 총 {len(st.session_state.pdf_files)}개의 PDF 파일에서 {len(st.session_state.all_docs)}개의 문서를 로드했습니다.")
    st.write(f"➡️ 분할된 청크의 수: {len(st.session_state.split_documents)}")
    if st.session_state.method == "시멘틱 청커 기반":
        st.write(f"➡️ 시멘틱 청크의 수: {len(st.session_state.semantic_chunks)}")
    st.text_area("➡️ 프롬프트 내용", st.session_state.prompt.template, height=350)

def render_qa():
    st.subheader("QA")
    st.markdown("---")
    st.session_state.question = st.text_area("질문을 해주세요!", "", height=200)
    if st.button("실행", key="execute"):
        # 최근 10개 대화 내역을 history로 생성 (대화 내역이 없으면 빈 문자열)
        history_str = "\n".join(st.session_state.conversation_history[-10:]) if st.session_state.conversation_history else ""
        response = st.session_state.chain.invoke({"question": st.session_state.question, "history": history_str})
        st.markdown("---")
        st.subheader("Groundedness Check")

        # 관련 문서를 하나의 문자열로 결합
        relevant_docs = st.session_state.retriever.get_relevant_documents(st.session_state.question)
        pdf_contents = "\n\n".join([doc.page_content for doc in relevant_docs])

        request_input = {"context": pdf_contents, "answer": response}
        st.session_state.groundedness_check = UpstageGroundednessCheck(
            api_key=st.session_state.upstages_api_key,
            model="solar-pro",
            temperature=0,
            base_url="https://api.upstage.ai/v1",
        )
        st.session_state.gc_result = st.session_state.groundedness_check.invoke(request_input)

        if st.session_state.gc_result.lower().startswith("grounded"):
            st.text("✅ Groundedness check passed")
            st.markdown(response)
        else:
            st.text("❌ Groundedness check failed")
            st.text("🔍 Tavily API를 사용하여 웹 검색 결과를 가져옵니다.")
            # st.session_state.tavily_retriever = TavilySearchAPIRetriever(k=3)
            tavily_chain = create_tavily_search_chain(st.session_state.tavily_api_key)
            tavily_response = tavily_chain.invoke({"question": st.session_state.question, "history": history_str})
            st.markdown(tavily_response)

        
        # 대화 내역에 현재 대화 추가
        # st.session_state.conversation_history.append("User: " + st.session_state.question)
        # st.session_state.conversation_history.append("Assistant: " + response)

def main():
    load_dotenv()
    st.session_state.upstages_api_key = os.getenv("UPSTAGE_API_KEY")
    st.session_state.tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    # 여기서 session_state 초기화를 먼저 수행합니다.
    initialize_session()

    setup_page()
    col1, col2, col3 = st.columns([1, 1, 2], border=True)

    with col1:
        sidebar_settings()

    if st.session_state.get("rag_state"):
        with col1:
            run_pipeline()
        with col2:
            render_result()
        with col3:
            render_qa()

if __name__ == "__main__":
    main()