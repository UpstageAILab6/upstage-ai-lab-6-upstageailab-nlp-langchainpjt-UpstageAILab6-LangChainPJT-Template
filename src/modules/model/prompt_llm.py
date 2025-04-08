import os

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
hb_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")


class PromptLLM:
    def __init__(self, hb_api_key, retriever):
        self.hb_api_key = hb_api_key
        self.retriever = retriever

        self.prompt = PromptTemplate(
            template="""당신은 패스트캠퍼스의 AI LAB 행정 매니저입니다. 
            다음에 제공된 검색된(context) 정보를 사용하여 질문에 답하세요. 
            만약 답을 모른다면, 모른다고 말하세요. 답변은 한국어로 작성하세요.

            # Question:
            {question} 

            # Context:
            {context} 

            # Answer:
            """,
            input_variables=["question", "context"]
        )

        self.llm = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/upstage/SOLAR-10.7B-Instruct",
            # Model upstage/SOLAR-10.7B-Instruct does not exist
            huggingfacehub_api_token=self.hb_api_key,
            temperature=0.2,
            max_new_tokens=100,
        )
        # 체인 생성 (retriever → prompt → LLM → 출력 파싱)
        self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

    def generate_response(self, question) -> str:
        """질문을 입력하면 체인을 실행하고 응답을 반환"""
        return self.chain.invoke(question)
