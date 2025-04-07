# app/app.py
import streamlit as st
from model.model import load_chain

def run_app():
    st.title("스포츠 룰 질의응답 시스템")
    # 예시: data/pdf 폴더 내 PDF 파일들을 사용
    pdf_source = "data/pdf/"
    chain = load_chain(pdf_source)

    question = st.text_input("궁금한 내용을 입력하세요:")
    if st.button("질문 전송") and question:
        with st.spinner("답변 생성 중..."):
            answer = chain.invoke(question)
        st.subheader("답변")
        st.write(answer)