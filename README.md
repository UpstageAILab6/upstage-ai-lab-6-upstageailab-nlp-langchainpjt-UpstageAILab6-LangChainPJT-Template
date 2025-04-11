[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5BS4k7bR)
# **Langchain 및 RAG를 활용한 스포츠 룰 기반의 QA Engine 시스템 구축**

LangChain 기술을 활용하여, 최신 계정의 스포츠 룰 PDF 문서 기반 Q&A 시스템을 구축하는 프로젝트입니다.  
RAG(Retrieval-Augmented Generation) 구조를 바탕으로 문서 검색 및 응답 시스템을 구현하고, 전체 모델 생애주기를 관리 가능한 파이프라인으로 구성했습니다.

- **프로젝트 기간:** 2025.04.02 ~ 2025.04.15  
- **주제:** LangChain 기반 스포츠룰  + Q&A 자동화 시스템  

---

# **팀원 소개**

| 이름      | 역할             | GitHub                | 담당 기능                                         |
|-----------|------------------|------------------------|--------------------------------------------------|
| **손봉교** | 팀장  | [GitHub 링크](#https://github.com/imsonn94)       | streamlit 초기 적용, Chat History 구현, 코드 리뷰 |
| **박인수** | 팀원 | [GitHub 링크](#https://github.com/insu97)       | streamlit UI 개선, SemanticChuncker 적용, 코드 리펙토링 |


---

# **파이프라인 워크플로우**
![alt text](image.png)

## **1. 비즈니스 문제 정의**
- 최신으로 개정된 스포츠 룰에 대한 대한 빠르고 정확한 자동 응답 시스템 구축

## **2. 데이터 수집 및 전처리**
1. **데이터 수집**
   - PDF, 웹서치, 크롤링 활용
2. **문서 파싱 및 전처리**
   - LangChain의 DocumentLoader
   - PDFLoader
   - Chunking, SemanticChunker, Text Cleaning
3. **임베딩 및 벡터화**
   - Upstages AI / Embedding 모델 사용
   - FAISS / Weaviate / Qdrant 등을 활용한 벡터 DB 구축
4. **데이터 버전 관리**
   - Github를 통한 형상 관리 (v1.x, v2.x)

## **3. LLM 및 RAG 파이프라인 구성**
- LangChain의 RetrievalQA 모듈 활용
- Chain 구성: Embedding → Retriever → Prompt → LLM
- LLM: Upstages AI Solar Pro

## **4. 실행 환경 구성**
1. **터미널 기반 CLI로 즉시 테스트 가능**
2. **로컬에서 Streamlit으로 웹페이지 구현**
3. **MCP의 경우 Cursor AI에 로컬 시스템으로 구축**

## **6. 모니터링 및 재학습 루프**
1. **트래킹 시스템 활용**
   - LangSmith 활용

---

## **프로젝트 실행 방법**

**로컬 환경 또는 터미널 기반으로 실행** 가능합니다.

```bash
# 1. 프로젝트 클론
git clone https://github.com/UpstageAILab6/upstageailab-nlp-langchainpjt-langchain-5.git
cd upstageailab-nlp-langchainpjt-langchain-5

# 2. 가상환경 설정 및 패키지 설치
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 환경 변수 설정
export UPSTAGES_API_KEY=your-api-key

# 4-1. 실행 (CLI 기반)
python main.py

# 4-2. 실행 (Streamlit) -> http://localhost:8501/ 로컬 접속
streamlit run main.py --server.address=0.0.0.0

```

---

## ** 협업 툴**

- **소스 관리:** GitHub
- **프로젝트 관리:** GitHub 칸반보드
- **커뮤니케이션:** Slack
- **버전 관리:** Git

---

## **기대 효과 및 향후 계획**
- 매년 새롭게 추가되거나 삭제되는 스포츠 룰에 대한 포괄적인 이해 가능
- 새롭게 떠오르는 추세인 MCP 적용방안에 대한 고찰
- 향후 다양한 도메인 문서(QA, 정책, 교육자료 등)에 확장 적용 예정

---
## **강사님 피드백 및 프로젝트 회고**

프로젝트 진행 중 담당 강사님과의 피드백 세션을 통해 얻은 주요 인사이트는 다음과 같습니다.

### 📌 **1차 피드백 (2025.04.03)**
- **주제선정**  
  → LLM이 해결하기에 Hallucination이 빈번하거나, 문서를 직접 뜯어보아야 하는 문제에 대한 문제를 파악하여 주제를 선정해야함
- **RAG에 대해 기초적인 지식이 부족함**  
  → 기초적인 강의를 빠르게 습득 후 프로젝트에 수행해야 함

### 📌 **2차 피드백 (2025.04.04)**
- **WorkFlow 정의 필요**  
  - WorkFlow를 명확하게 정의하여 한번 그려보고, 순차적으로 작업할 필요가 있음
- **읽기 어려운 PDF 데이터에 대한 여러 전처리 방안 시도 필요**  
  - 표, 이미지 등을 어떤 방식으로 처리 후 임베딩하여 더 많은 정보를 이끌어 낼 수 있을지에 대한 고민이 필요함
  - PDF -> HTML 변환을 사용하여 밑줄 친 (축구 24/25년도 개정안 규칙) 규칙을 인식할 수 있는 태그 확인 필요
  - OCR 추출, Upstages에서 제공하는 다양한 Chunking 및 DataLoader에 대해 파악해보고 시도해보기
- **정량적 평가의 필요성**
  - HuggingFace 리더보드 참고
  - 스코어 내보기
- **웹 서치 및 GroundCheck 도입**
  - 원하는 답변을 내올 수 없을 시에 LLM이 생성하게 하거나 웹 서치를 활용하는 방안
- **프롬프트 엔지니어링**
  - 다양한 프롬프트 엔지니어링 시도
- **Retriever, Embedding 뜯어보기**
  - 각 단계에서 일어나는 프로세스 직접 파악해보기 

### 📌 **3차 피드백 (2025.04.07 ~ 2025.04.08)**
- **코드 리펙토링 미흡**
  - 각 함수마다 변수타입 힌트 지정
  - 모듈 및 패키지 import 순서 명확하게 하기 ( 빌트인 - 서드파티 - 커스텀모듈 - 로컬모듈)
  - import 중복 피하기
  - 주석 삭제하기
  - 모듈을 이어 붙일 때 Class선언하기
  - 함수명은 동사+명사 형태로 작성 (소문자 시작)
  - Class 선언 시 첫 글자는 대문자, 두 번째 단어의 첫 글자를 대문자로 하여 구분
  - llm 모듈을 main.py에 지정했으면 더 좋았을 것임 (모델을 바꾸기에 용이함)
- **Git Commit**   
  - git commit의 컨벤션은 매우 준수하나 commit 할 때 Enter 친 후 커밋하기
  - git commit시에 소문자로 컨벤션 시작하여 동사+명사 형태로 작성 후 커밋하기
- **발표자료 피드백**  
  - 코드를 보여주기 보다는 WorkFLow 순서로 설명하는 것을 추천함
