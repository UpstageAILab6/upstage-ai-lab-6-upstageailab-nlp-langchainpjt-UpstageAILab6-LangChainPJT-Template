from langchain_openai import ChatOpenAI

class GovPolicyLLM:
    """
    LLM API를 호출하고, response를 받는 역할을 합니다.
    request / response 형식을 정의하며, LangSmith 설정 기반으로 실행됩니다.
    """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def get_llm(self):
        return self.llm
