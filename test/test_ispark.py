import os
import glob
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------

st.set_page_config(layout="wide")

# -----------------------------------------------------------

st.title("RAG project")

# -----------------------------------------------------------

# 환경변수 및 API 키 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

col1, col2 = st.columns(2, border=True)

def document_load():
    pdf_folder = "../data/"
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    all_docs = []

    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(pdf_file)
        docs = loader.load()
        all_docs.extend(docs)

    return pdf_files, all_docs

def document_split(chunk_size=1000, chunk_overlap=250):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    split_documents = text_splitter.split_documents(st.session_state.all_docs)

    return split_documents

def embedding_store_retriever(split_documents):
    embeddings = UpstageEmbeddings(api_key=openai_api_key, model="embedding-query", base_url="https://api.upstage.ai/v1")

    if os.path.exists("faiss_index_dir"):
        vectorstore = FAISS.load_local("faiss_index_dir", embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore.save_local("faiss_index_dir")

    retriever = vectorstore.as_retriever()

    return embeddings, vectorstore, retriever

# 평가 지표 계산 함수들
def ndcg_at_k(relevances, k):
    relevances = np.array(relevances)[:k]
    dcg = np.sum((2**relevances - 1) / np.log2(np.arange(2, relevances.size + 2)))
    ideal_relevances = np.sort(relevances)[::-1]
    idcg = np.sum((2**ideal_relevances - 1) / np.log2(np.arange(2, len(ideal_relevances) + 2)))
    return dcg / idcg if idcg > 0 else 0

def reciprocal_rank(relevances):
    for i, r in enumerate(relevances):
        if r > 0:
            return 1.0 / (i + 1)
    return 0

def average_precision(relevances):
    precisions = []
    num_relevant = 0
    for i, r in enumerate(relevances):
        if r > 0:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    return np.mean(precisions) if precisions else 0

def recall_at_k(relevances, total_relevant, k):
    retrieved_relevant = sum(1 for r in relevances[:k] if r > 0)
    return retrieved_relevant / total_relevant if total_relevant > 0 else 0

def evaluate_query(query, ground_truth_sources, retriever, k=10):
    retrieved_docs = retriever.get_relevant_documents(query, k=k)
    relevances = []
    # 각 문서의 metadata에 저장된 "source"가 ground truth에 있으면 1, 없으면 0
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "")
        relevances.append(1 if source in ground_truth_sources else 0)
    metrics = {
        "NDCG@10": ndcg_at_k(relevances, k),
        "MRR@10": reciprocal_rank(relevances),
        "MAP@10": average_precision(relevances),
        "Recall@10": recall_at_k(relevances, total_relevant=len(ground_truth_sources), k=k)
    }
    return metrics

# -----------------------------------------------------------

# similarity 평가 지표 함수들

def cosine_similarity(vec1, vec2):
    """두 벡터 간의 코사인 유사도 계산"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_similarity(query, retriever, embeddings, k=10):
    """
    쿼리와 검색된 각 문서의 텍스트 간 유사도를 계산합니다.
    여기서는 embeddings.embed_query를 사용해 임베딩하고 cosine similarity를 계산합니다.
    """
    query_embedding = embeddings.embed_query(query)
    retrieved_docs = retriever.get_relevant_documents(query, k=k)
    sims = []
    for doc in retrieved_docs:
        # 각 문서의 page_content를 사용하여 임베딩 계산
        doc_embedding = embeddings.embed_query(doc.page_content)
        sim = cosine_similarity(query_embedding, doc_embedding)
        sims.append(sim)
    avg_similarity = np.mean(sims) if sims else 0
    return {"AvgSimilarity@10": avg_similarity}

# -----------------------------------------------------------

with col1:
    st.header("RAG QA")
    st.write("")

    left, right = st.columns(2, border=True)

    if left.button("Document Load", key="load"):
        with right:
            st.session_state.pdf_files, st.session_state.all_docs = document_load()
            st.write(f"➡️ 총 {len(st.session_state.pdf_files)}개의 PDF 파일에서 {len(st.session_state.all_docs)}개의 문서를 로드했습니다.")

    left.write("⬇️")

    if left.button("Document Split: 문서 청크로 분할", key="split"):
        with right:
            st.session_state.split_documents = document_split(chunk_size=1000, chunk_overlap=250)
            st.write(f"➡️ 분할된 청크의 수: {len(st.session_state.split_documents)}")

    left.write("⬇️")

    if left.button("Embedding 및 벡터스토어 + 검색기(Retriever) 생성", key="embedding_store_retriever"):
        with right:
            st.session_state.embeddings, st.session_state.vectorstore, st.session_state.retriever = embedding_store_retriever(st.session_state.split_documents)
            st.success("성공적으로 실행되었습니다.")

    left.write("⬇️")

    if "prompt_generated" not in st.session_state:
        st.session_state.prompt_generated = False
    if "prompt" not in st.session_state:
        st.session_state.prompt = None

    # "프롬프트 작성" 버튼을 눌렀을 때, 세션 상태에 생성 여부를 기록
    if left.button("프롬프트 작성", key="create_prompt"):
        st.session_state.prompt_generated = True

    # 프롬프트 생성 상태가 True면 오른쪽 영역에 프롬프트 입력창과 저장 버튼 표시
    if st.session_state.prompt_generated:
        with right:
            prompt_text = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    #Question: 
    {question} 
    #Context: 
    {context} 

    #Answer:"""
            # 텍스트 영역에 기본 프롬프트를 보여주고, 사용자가 수정한 내용을 new_prompt에 저장
            new_prompt = st.text_area("프롬프트를 작성해주세요!", value=prompt_text, height=350, key="new_prompt")
            
            # "프롬프트 저장" 버튼 (세션 상태를 이용해 항상 렌더링됨)
            if st.button("프롬프트 저장", key="save_prompt"):
                st.session_state.prompt = new_prompt
                st.success("저장 완료!")
                st.write("저장된 프롬프트:")
                st.text(st.session_state.prompt)
                st.session_state.prompt_generated = False

            

    if left.button("프롬프트 적용", key="prompt_check"):
        with right:
            st.write(PromptTemplate.from_template(st.session_state.prompt))
            st.session_state.prompt_info = PromptTemplate.from_template(st.session_state.prompt)

    left.write("⬇️")

    if left.button("모델 생성 및 체인 생성", key="model_create"):
        with right:
            llm = ChatUpstage(api_key=openai_api_key, model="solar-pro", temperature=0)
            st.session_state.chain = (
                {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                | st.session_state.prompt_info
                | llm
                | StrOutputParser()
            )

            st.success("성공~")
    
    left.write("⬇️")

    if "qa" not in st.session_state:
        st.session_state.qa = False

    if left.button("QA", key="QA"):
        st.session_state.qa = True
    
    if st.session_state.qa:
        with right:
            st.session_state.question = st.text_area("질문을 해주세요!", "농구에서 3점 슛 기준에 대해서 bullet points 형식으로 작성해 주세요.", height=200)

            if st.button("실행", key="execute"):
                response = st.session_state.chain.invoke(st.session_state.question)
                st.text(response)
                # st.session_state.qa = False
        

with col2:
    st.header("RAG 평가")

    st.session_state.query = st.text_input("Query", "농구에서 3점 슛 기준")
    st.session_state.doc = st.selectbox("평가할 문서를 선택해주세요!", ("jiu-jitsu", "kbl", "soccer", "poker"))

    if st.session_state.doc == 'jiu-jitsu':
        st.session_state.ground_truth = "../data/jiu-jitsu.pdf"
    elif st.session_state.doc == 'kbl':
        st.session_state.ground_truth = "../data/kbl.pdf"
    elif st.session_state.doc == 'soccer':
        st.session_state.ground_truth = "../data/soccer.pdf"
    elif st.session_state_doc == 'poker':
        st.session_state.ground_truth == '../data/poker.pdf'

    if st.button("평가하기"):
        answer = st.session_state.chain.invoke(st.session_state.query)
        st.subheader("답변")
        st.write(answer)

        # 평가 지표 계산
        metrics = evaluate_query(st.session_state.query, st.session_state.ground_truth, st.session_state.retriever, k=10)
        similarity_metrics = evaluate_similarity(st.session_state.query, st.session_state.retriever, st.session_state.embeddings, k=10)
        metrics.update(similarity_metrics)
        st.subheader("평가 지표")
        st.write(metrics)

