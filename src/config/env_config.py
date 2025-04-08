import os
from dotenv import load_dotenv
import streamlit as st

def configure_upstage_api():
    """
    .env 파일을 로드하고 UPSTAGE_API_KEY 환경 변수를 설정합니다.
    키가 없을 경우 에러 메시지를 출력하고 False를 반환합니다.
    """
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    if not upstage_api_key:
        st.error("UPSTAGE_API_KEY가 설정되어 있지 않습니다. .env 파일에 해당 변수를 추가해주세요.")
        return False
    os.environ["UPSTAGE_API_KEY"] = upstage_api_key
    return True
