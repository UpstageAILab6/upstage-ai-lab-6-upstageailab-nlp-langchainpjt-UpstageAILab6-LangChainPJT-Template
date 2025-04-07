# main.py
import os
import sys

def run_cli():
    """
    CLI 모드: 터미널에서 실행 시 질문을 입력받아 답변을 출력합니다.
    """
    from model.model import load_chain

    pdf_path = "data/pdf/"
    if not os.path.exists(pdf_path):
        print("지정한 PDF 파일이 존재하지 않습니다.")
        return

    chain = load_chain(pdf_path)
    print("알고 싶은 경기 규칙을 입력하세요...")
    while True:
        question = input("질문 (종료하려면 '종료' 입력): ").strip()
        if question.lower() in ['종료', 'exit', 'quit']:
            print("프로그램을 종료합니다.")
            break

        answer = chain.invoke(question)
        print("답변:")
        print(answer)

def run_streamlit():
    """
    Streamlit 모드: app 모듈의 run_app() 함수를 호출하여 웹 인터페이스를 실행합니다.
    """
    from app.app import run_app
    run_app()

def main():
    """
    실행 환경에 따라 CLI 모드 또는 Streamlit 모드를 선택합니다.
    streamlit.runtime.scriptrunner.script_run_context.get_script_run_ctx()를 이용해
    Streamlit 런타임 환경을 감지합니다.
    """
    try:
        from streamlit.runtime.scriptrunner import script_run_context as stctx
        if stctx.get_script_run_ctx() is not None:
            run_streamlit()
        else:
            run_cli()
    except Exception:
        run_cli()


        

if __name__ == "__main__":
    main()



# 모듈 테스트 예시
# if __name__ == "__main__":
#     sample_question = "최근 AAPL 주식 가격은 어떻게 되나요?"
#     try:
#         answer = MCPChain.invoke(sample_question)
#         print("답변:", answer)
#     except Exception as e:
#         print("오류 발생:", e)

