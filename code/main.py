from config import ConfigLoader
from loader import DocumentLoader
from vectorstore import VectorStoreManager
from llm import GovPolicyLLM
from prompt import GovPolicyPrompt, MarkdownFormatter
from gov_policy_qa import GovPolicyQA
from chat_history import ChatHistory
from logger import Logger
from grounding_checker import VectorstoreGroundingChecker
from langchain_openai import OpenAIEmbeddings

# 1. 환경 설정 및 초기화
config = ConfigLoader(project_name="GovPolicyQA")
client = config.load()

logger = Logger()
history = ChatHistory()

# 2. 데이터 로드 및 벡터스토어 구축
doc_loader = DocumentLoader(filepath="data/serviceDetail_all.csv")
documents = doc_loader.load_documents()

embeddings = OpenAIEmbeddings()
vs_manager = VectorStoreManager(embeddings=embeddings)
vectorstore = vs_manager.load_or_create(documents, path="code/faiss_index_v2")

# 3. LLM 및 Prompt 설정
llm = GovPolicyLLM(model_name="gpt-4o", temperature=0).get_llm()
prompt = GovPolicyPrompt().get_prompt()
formatter = MarkdownFormatter()

# 4. QA 시스템 초기화
qa_system = GovPolicyQA(vectorstore, embeddings, llm, prompt, formatter)

# 5. 벡터스토어 기반 그라운드체커 초기화
grounding_checker = VectorstoreGroundingChecker(vectorstore, embeddings)

# 6. 대화 루프
print("정부 지원 정책 질문 시스템입니다. 'exit' 입력 시 종료됩니다.\n")

while True:
    question = input("질문: ").strip()
    if question.lower() in ["exit", "quit", "종료"]:
        print("종료합니다.")
        break

    logger.log_user_input(question)
    history.add_user_message(question)

    # 답변 생성
    answer = qa_system.run(question)

    # 그라운드체크 실행
    grounding_results = grounding_checker.run(answer)

    # 출력
    print("\n📌 챗봇 답변:\n")
    print(answer)

    print("\n🧠 그라운딩 체크 결과:")
    if grounding_results:
        for idx, r in enumerate(grounding_results, 1):
            print(f"\n[{idx}] 문장: {r['sentence']}")
            print(f"    근거 문맥: {r['matched_context'][:100]}...")
            print(f"    유사도: {r['similarity']:.3f} → {'✅ 근거 있음' if r['grounded'] else '⚠️ 근거 없음'}")
    else:
        print("그라운드체킹 결과가 없습니다.")

    logger.log_assistant_response(answer)
    history.add_assistant_message(answer)