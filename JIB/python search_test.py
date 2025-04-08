from QA.modules.embedder import load_faiss_and_search

query = input(" 질문을 입력하세요: ")
results = load_faiss_and_search(query)

if results:
    print("\n 검색된 문단:")
    for i, r in enumerate(results, 1):
        print(f"\n--- 결과 {i} ---\n{r}")
else:
    print("관련 문단을 찾을 수 없습니다.")

