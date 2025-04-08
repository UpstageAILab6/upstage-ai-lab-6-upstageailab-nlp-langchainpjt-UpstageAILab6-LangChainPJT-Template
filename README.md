[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5BS4k7bR)
# **LangChain 프로젝트**
LangChain을 활용한 **부트캠프 RAG봇** 구축 프로젝트입니다.  
본 프로젝트는 부트캠프 교육 과정에서 제공되는 다양한 문서(강의 시간표, 강의 리스트, 법령, 슬랙 공지 등)를 기반으로, 사용자 질문에 자동으로 응답할 수 있는 Q&A 시스템을 구현하는 데 목적이 있습니다.

**부트캠프 RAG봇**은 Retrieval-Augmented Generation(RAG) 구조를 기반으로, 사용자 질문을 적절한 도메인(예: 강의, 문서, 일정 등)으로 분류한 뒤, 관련 정보를 검색하고 자연어로 응답을 생성합니다.  
질문 도메인에 따라 라우팅된 체인에서 각기 Search로직을 통해 벡터 저장소 및 프롬프트를 사용함으로써, 보다 정확하고 문맥에 맞는 답변을 제공할 수 있도록 설계되었습니다.

- **프로젝트 기간:** 2025.04.02 ~ 2025.04.08  
- **주제:** LangChain 기반 문서 검색 + Q&A 자동화 시스템  

---

# **팀원 소개**

| 이름      | 역할             | GitHub                | 담당 기능                                         |
|-----------|------------------|------------------------|--------------------------------------------------|
| **강태화** |  팀장 | [GitHub 링크](https://github.com/wooobo)| 아키텍쳐 구조 설게, 휴가/출석대장 작성법과 과정시간표 데이터 수집 및 임베딩, Langchain 통합 |
| **정혜린** |  팀원 | [https://github.com/jhyerin31](#) | 온라인 강의 데이터 수집 및 임베딩, LCEL 구현  |
| **정인복** |  팀원 | [GitHub 링크](#)| 내일배움카드 관련 법령 데이터 수집 및 임베딩, 프롬프트 출력 요약 |
| **진우재** |  팀원 | [GitHub 링크](#)|            |
| **박진신** |  팀원 | [GitHub 링크](#)|               |

---

# **파이프라인 워크플로우**

LangChain 기반 패스트캠퍼스/Upstage AI Lab 6기 과정 전반적인 QA 시스템의 구축 및 운영을 위한 파이프라인입니다.

## **1. 비즈니스 문제 정의**
- 내부 문서가 다양한 채널에 분산되어 있어 정보 탐색에 어려움 존재
- 복잡한 행정/교육 절차를 빠르게 이해할 수 있는 ChatBot 필요
- KPI: 사용자 질문 정확도, 활용성

## **2. 데이터 수집 및 전처리**
1. **데이터 수집**
   - 학원 내 자료(시간표, 강의 리스트, 법령, Notion Page) 수집
   - selenium 을 통한 웹 크롤링
2. **문서 파싱 및 전처리**
   - 팀원별 DocsLoader 개발 (도메인 특화)
   - 공통 규격에 맞춰 Parsing 및 Chunking 처리
3. **임베딩 및 벡터화**
   - 일관된 임베딩 로직 적용 (도메인별 Embedding 전략 사용)
   - Vector Store 재정의로 확장성 확보
4. **데이터 저장**
   - Embedding Vector 저장소 → LLM Context로 활용
     
## **3. LLM 및 RAG 파이프라인 구성**
- LangChain 기반 RetrievalQA 구성
- Domain Routing 체계 구축:
  - `domain_chain` → 질문 분석 및 도메인 분기
  - 도메인별 QA 체인 라우팅 후 응답 생성
- 주요 도메인 예시:
  - `lecture`, `schedule`, `lecal`
- LLM: ChatGPT/Solar API 기반

## **4. 모델 학습 및 실험 추적**
- 파인튜닝보다는 Retrieval 구조 중심 개발
- 프롬프트 설계 및 응답 구조 개선 시도
- Hallucination 개선을 위한 날짜 처리 등 실험 진행

## **5. 실행 환경 구성**
1. **LangChain + Streamlit App**
2. **로컬에서 가상환경 기반 실행 가능**

## **6. 모니터링 및 개선 루프**
1. **응답 예시 검토 및 Hallucination 확인**
2. **법령 응답 요약 기능 제안 → 가독성 개선 방향**
3. **향후 슬랙 피드백 기반 자동 개선 루프 구현 예정**
   
---

## **프로젝트 실행 방법**

본 프로젝트는 웹 서비스 형태로 배포하지 않아도 되며,  
**로컬 환경 또는 클라우드 인스턴스에서 터미널 기반으로 실행** 가능합니다.

```bash
# 1. 프로젝트 클론
git clone https://github.com/UpstageAILab6/upstageailab-nlp-langchainpjt-langchain-1.git
cd langchain-qa-project

# 2. 가상환경 설정 및 패키지 설치
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 환경 변수 설정
**.env.template 파일 참고하여 정의**
export OPENAI_API_KEY=your-api-key
export UPSTAGE_API_KEY=your-api-key
export HUGGINGFACEHUB_API_KEY=your-api-key

# 4. 실행
## 자료 수집
python src/modules/init.py

## 서비스 실행
streamlit run app.py
```

---

## **활용 장비 및 사용 툴**

### **활용 장비**
- **서버:** Local PC
- **개발 환경:** MacOS, Windows, Linux
- 
### **협업 툴**
- **소스 관리:** GitHub
- **프로젝트 관리:** GitHub
- **커뮤니케이션:** Slack
- **버전 관리:** Git

### **사용 도구**
- **LLM 통합:** LangChain, OpenAI API, HuggingFace
- **데이터 관리:** Local file system
- **배포 및 운영:** Local Terminal 
- **도구**: Jupyter Notebook / Terminal 기반 테스트

---

## **기대 효과 및 향후 계획**
- 다양한 도메인에 맞는 문서 응답 자동화 가능성 입증
- 시간표, 강의 리스트, 법령 등 다차원 정보 탐색 지원
- 향후 슬랙 연동, 메타데이터 검색, 요약 기능 등 확장 예정

---
## **강사님 피드백 및 프로젝트 회고**

프로젝트 진행 중 담당 강사님과의 피드백 세션을 통해 얻은 주요 인사이트는 다음과 같습니다.

### 📌 **1차 피드백 (2025.04.02)**
- **주제 선정**  
  → 도메인 분기 기반 Q&A라는 주제의 명확성 확보 및 활용도 고려
- **데이터 Chunking 전략**  
  → 텍스트의 의미 단위로 나누는 전략 필요성 제시
- **Embedding DB 구성 초기 설계**  
  → 단일 Vector Store가 아닌 도메인별 벡터 저장소 구성 제안

### 📌 **2차 피드백 (2025.04.03)**
- **Embedding DB 구성 고도화**  
  → 유연한 검색을 위한 필터링 및 메타데이터 설계 강조
- **실행 가능 코드 구성**  
  → 짧은 시간 내 프로토타입 수준의 실행 가능한 구조 완성 제안

### 📌 **3차 피드백 (2025.04.04)**
- **벡터 DB 다양화**  
  → FAISS 외에도 Qdrant, Milvus 등 비교 실험 제안
- **System Message Prompting**  
  → 프롬프트 설계 시 시스템 메시지를 활용해 문맥 유지 유도
- **Routing 전략 정교화**  
  → 질문의 도메인 분기를 좀 더 유연하게 처리하는 함수 설계 필요
- **응답 품질 요소 제안**  
  - **답변 길이 조절**: 질문 맥락에 맞는 길이로 최적화 필요  
  - **Ground Check 도입**: UpstageAI의 CAG 기반 평가 방법 참조  
    - [CAG_GC Notebook](https://github.com/UpstageAI/cookbook/blob/main/Solar-Fullstack-LLM-101/04_CAG_GC.ipynb) 활용 권장  
    - 실제 응답과 Ground Truth 비교를 통한 평가 체계 수립
