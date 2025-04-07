import streamlit as st
from config import ConfigLoader
from loader import DocumentLoader
from vectorstore import VectorStoreManager
from llm import GovPolicyLLM
from prompt import GovPolicyPrompt, MarkdownFormatter
from gov_policy_qa import GovPolicyQA
from chat_history import ChatHistory
from logger import Logger
from langchain_openai import OpenAIEmbeddings

# 페이지 설정
st.set_page_config(page_title="정부 정책 챗봇", layout="wide")
st.title("🇰🇷 정부 지원 정책 질문 챗봇")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatHistory()

if "qa_system" not in st.session_state:
    config = ConfigLoader(project_name="GovPolicyQA")
    config.load()

    st.session_state.logger = Logger()

    # 데이터 로딩 및 벡터스토어 구축
    doc_loader = DocumentLoader(filepath="data/serviceDetail_all.csv")
    documents = doc_loader.load_documents()

    embeddings = OpenAIEmbeddings()
    vs_manager = VectorStoreManager(embeddings=embeddings)
    vectorstore = vs_manager.load_or_create(documents, path="code/faiss_index_v2")

    # LLM & 프롬프트 설정
    llm = GovPolicyLLM(model_name="gpt-4o", temperature=0).get_llm()
    prompt = GovPolicyPrompt().get_prompt()
    formatter = MarkdownFormatter()

    # QA 시스템 구성
    st.session_state.qa_system = GovPolicyQA(vectorstore, embeddings, llm, prompt, formatter)

# 사용자 질문 입력
user_input = st.text_input("정부 지원 정책에 대해 궁금한 점을 입력하세요", "")

# 답변 생성 및 처리
if user_input:
    st.session_state.logger.log_user_input(user_input)
    st.session_state.chat_history.add_user_message(user_input)

    with st.spinner("답변을 생성 중입니다..."):
        answer = st.session_state.qa_system.run(user_input)

    st.session_state.logger.log_assistant_response(answer)
    st.session_state.chat_history.add_assistant_message(answer)

    # 📌 챗봇 답변 출력
    st.markdown("---")
    st.subheader("답변")
    st.markdown(answer, unsafe_allow_html=True)

# 💬 대화 히스토리 출력
if st.session_state.chat_history.get_history():
    st.markdown("---")
    st.subheader("💬 대화 기록")

    for message in st.session_state.chat_history.get_history():
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(f"**👤 사용자:** {content}")
        elif role == "assistant":
            st.markdown(f"**🤖 챗봇:**\n\n{content}", unsafe_allow_html=True)
