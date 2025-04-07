import os
from dotenv import load_dotenv
from langsmith import Client

class ConfigLoader:
    def __init__(self, project_name: str = "Test"):
        self.project_name = project_name

    def load(self):
        load_dotenv()

        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        os.environ["LANGCHAIN_PROJECT"] = self.project_name

        print("OpenAI 키 로드됨:", os.getenv("OPENAI_API_KEY") is not None)
        print("LangSmith 키 로드됨:", os.getenv("LANGSMITH_API_KEY") is not None)
        print("현재 LangSmith 프로젝트:", self.project_name)

        # 필요 시 LangSmith Client도 반환
        return Client()
