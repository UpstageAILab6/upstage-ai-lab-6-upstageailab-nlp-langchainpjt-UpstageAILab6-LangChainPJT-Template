import streamlit as st
from dotenv import load_dotenv

from src.config.env_config import configure_upstage_api
from src.modules.embedding.embedding import Embedding
from src.modules.model.chat import get_response
from src.modules.prompt.template import get_vacation_messages, get_timetable_messages, get_legal_messages

from src.modules.router.question_router import route_question
from src.modules.vector_store.search import handle_vacation_search, handle_timetable_search, handle_legal_search
from src.modules.vector_store.vector_store import ChromaStore, VectorStore


def main(vector_store):
    st.title("AI 부트캠프 매니저 QA봇")
    question = st.text_input("질문을 입력하세요", "카드 발급은 어떻게 하나요?")

    if st.button("Get Answer"):
        with st.spinner("카테고리 라우팅 중..."):
            category = route_question(question)
        st.write(f"**예측된 카테고리:** {category}")

        with st.spinner("답변 생성 중..."):

            if category == "vacation":
                context_text, attached_files = handle_vacation_search(vector_store, question)
                messages = get_vacation_messages(context_text, "", question)
            elif category == "timetable":
                context_text, attached_files = handle_timetable_search(vector_store, question)
                messages = get_timetable_messages(context_text, attached_files, question)
            elif category == "legal":
                context_text, attached_files = handle_legal_search(vector_store, question)
                messages = get_legal_messages(context_text, attached_files, question)
            else:
                # etc인 경우 기본 검색 + 기본 프롬프트
                messages = [
                    (
                        "system",
                        "당신은 학원 QA 봇입니다. (기본)"
                        "질문자는 \"AI+Lab_7기(Upstage_AI+6기)\"입니다."
                    ),
                    ("human", f"질문:\n\n{question}"),
                ]

            response = get_response(messages)

            st.subheader("LLM 응답")
            st.write(response.content)

if __name__ == "__main__":
    load_dotenv()
    configure_upstage_api()
    embeddings = Embedding()
    store_path = "src/chroma"

    vector_store: VectorStore = ChromaStore(embeddings.model, persist_directory=store_path)

    main(vector_store)

