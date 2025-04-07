# model/model.py
from data.pdf_processing import pdf_load_and_split
from embedding.embedding import initialize_embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def initialize_models():
    """
    embedding/embedding.py의 함수를 재사용하여 LLM과 임베딩 모델을 초기화합니다.
    """
    return initialize_embeddings()

def extract_sources(inputs):
    """
    리트리버가 반환한 문서에서 source metadata를 추출하고,
    문서의 page_content를 결합하여 context 텍스트를 생성합니다.
    """
    docs = inputs["context"]
    return {
        "sources": [doc.metadata.get("source", "") for doc in docs],
        "question": inputs["question"],
        "history": inputs["history"],
        "context": "\n\n".join(doc.page_content for doc in docs)
    }

def load_chain(pdf_source):
    """
    PDF 파일(또는 폴더)로부터 문서를 처리하고,
    질문-답변 체인을 구성하여 반환합니다.
    
    Args:
        pdf_source (str): PDF 파일 경로나 폴더 경로.
    
    Returns:
        체인(chain) 객체
    """
    # PDF 처리 및 벡터스토어 구축 (retriever 획득)
    _, _, _, retriever = pdf_load_and_split(pdf_source)
    
    # 모델 초기화
    llm, _ = initialize_models()
    
    # 프롬프트 설정
    prompt = PromptTemplate.from_template(
        """너는 스포츠 룰 전문가야
- 대답은 개조식으로 작성해줘.
- 출력은 마크다운 형식을 반영해줘.
- 별도의 요청이 없으면 답변은 한국어로 작성해줘.

#Context:
{context}

#Chat History:
{history}

#Question:
{question}

#Answer:"""
    )
    
    def retrieve_and_extract(inputs):
        """
        입력된 질문을 기반으로 retriever를 호출하여 관련 문서를 가져오고,
        extract_sources 함수를 통해 context와 source 정보를 구성합니다.
        """
        # retriever를 호출하여 질문에 맞는 문서를 검색
        docs = retriever.invoke(inputs["question"])
        new_inputs = {
            "question": inputs["question"],
            "history": inputs["history"],
            "context": docs
        }
        return extract_sources(new_inputs)
    
    # 체인 생성: 초기 입력은 RunnablePassthrough()를 사용해 그대로 전달합니다.
    chain = (
        RunnablePassthrough()
        | retrieve_and_extract
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
