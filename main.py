import chat_history

if __name__ == "__main__":
    while True:
        # 실행할 질문
        question = input("입력: ")

        # RAG 실행
        response = chat_history.chain_with_history.invoke(
            {"question": question},
            config={"configurable": {"session_id": "rag123"}},
        )

        print(response)
