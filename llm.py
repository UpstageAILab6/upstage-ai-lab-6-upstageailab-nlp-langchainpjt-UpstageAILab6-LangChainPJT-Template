from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from operator import itemgetter
from dotenv import load_dotenv
import os
import retriever
from chat_history import *

# API 키 정보 로드
load_dotenv()

# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Answer in Korean.

#Previous Chat History:
{chat_history}

#Question:
{question}
#Context:
{context}

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatUpstage(api_key=os.getenv('API_KEY'), model="solar-pro")

# 단계 8: 체인(Chain) 생성
chain = (
    {
        "context": itemgetter("question") | retriever.retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

__all__ = ["chain"]