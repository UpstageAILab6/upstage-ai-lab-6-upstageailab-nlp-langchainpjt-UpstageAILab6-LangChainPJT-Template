from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from grounding_checker import SentenceGroundingChecker
import pandas as pd

# ✅ 임베딩 모델 초기화
embeddings = OpenAIEmbeddings()

# ✅ 그라운드체커 인스턴스 생성
checker = SentenceGroundingChecker(embeddings=embeddings, threshold=0.75)

# ✅ 테스트용 LLM 답변 예시
answer = """
의정부시에 거주하거나 의정부시 소재 학교에 재학 중이거나 졸업한 청년들이 면접 준비를 위한 정장 대여비와 이력서 사진 촬영비를 지원받을 수 있습니다.
창원시의 청년들에게 면접 수당을 지급합니다.
"""

# ✅ 테스트용 context 문서 예시
docs = [
    Document(page_content="의정부시에 거주하거나 의정부시 소재 학교에 재학 중이거나 졸업한 청년"),
    Document(page_content="창원시에 거주하는 미취업 청년에게 면접 수당을 지급"),
    Document(page_content="청년정책 신청은 워크넷을 통해 가능"),
]

# ✅ 문장별 근거 매핑 실행
results = checker.run(answer, docs)

# ✅ 결과 출력
df = pd.DataFrame(results)
print(df.to_markdown(index=False))  # 콘솔에 마크다운 형태로 출력