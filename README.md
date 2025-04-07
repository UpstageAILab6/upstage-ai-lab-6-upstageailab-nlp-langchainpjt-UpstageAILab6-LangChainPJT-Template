[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5BS4k7bR)
# **LangChain 프로젝트** *(예시)*

LangChain과 MLOps 기술을 활용하여, 사내 문서 기반 Q&A 시스템을 구축하는 프로젝트입니다.  
RAG(Retrieval-Augmented Generation) 구조를 바탕으로 문서 검색 및 응답 시스템을 구현하고, 전체 모델 생애주기를 관리 가능한 파이프라인으로 구성했습니다.

- **프로젝트 기간:** 2025.03.01 ~ 2025.04.15  
- **주제:** LangChain 기반 문서 검색 + Q&A 자동화 시스템  

---

# **팀원 소개**

| 이름      | 역할             | GitHub                | 담당 기능                                         |
|-----------|------------------|------------------------|--------------------------------------------------|
| **김패캠** | 팀장 / 역할 | [GitHub 링크](#)       | LangChain 통합, FastAPI 백엔드 구성, API 설계 및 구조화 |
| **박패캠** |  역할   | [GitHub 링크](#)       | CI/CD 파이프라인 구축, 도커화, 로컬/클라우드 환경 구성 |
| **이패캠** | 역할 | [GitHub 링크](#)       | 문서 임베딩 처리, 벡터 DB 구축, LLM 파인튜닝           |
| **최패캠** | 역할     | [GitHub 링크](#)       | 데이터 수집, 전처리, DVC 및 S3 데이터 관리            |

---

# **파이프라인 워크플로우**

LangChain 기반 문서 QA 시스템의 구축 및 운영을 위한 파이프라인입니다.

## **1. 비즈니스 문제 정의**
- 내부 문서에 대한 빠르고 정확한 자동 응답 시스템 구축
- 고객지원 및 사내 지식관리의 효율성 증대
- KPI: 응답 정확도, 평균 응답 시간, 사용자 만족도

## **2. 데이터 수집 및 전처리**
1. **데이터 수집**
   - Notion, PDF, 사내 위키 등에서 문서 수집 후 S3 저장
2. **문서 파싱 및 전처리**
   - LangChain의 DocumentLoader 사용
   - Chunking, Text Cleaning
3. **임베딩 및 벡터화**
   - OpenAI / HuggingFace Embedding 모델 사용
   - FAISS / Weaviate / Qdrant 등을 활용한 벡터 DB 구축
4. **데이터 버전 관리**
   - DVC 및 S3로 문서 버전 관리

## **3. LLM 및 RAG 파이프라인 구성**
- LangChain의 RetrievalQA 모듈 활용
- Chain 구성: Embedding → Retriever → LLM(응답)
- LLM: OpenAI GPT-4 / Mistral / Claude 등 선택 가능

## **4. 모델 학습 및 실험 추적**
- 필요 시, 사내 문서로 파인튜닝된 LLM 학습
- MLflow를 통해 실험, 하이퍼파라미터, 모델 버전 관리
- Optuna / Weights & Biases 연동 가능

## **5. 실행 환경 구성**
1. **FastAPI 기반 API 서버 구성 (옵션)**
2. **Docker로 로컬 환경에서 통합 실행 가능**
3. **터미널 기반 CLI로 즉시 테스트 가능**
4. **로컬 또는 클라우드 환경(AWS EC2 등) 모두 지원**

## **6. 모니터링 및 재학습 루프**
1. **모델 성능 모니터링**
   - Prometheus, Grafana를 통한 응답 시간 및 정확도 트래킹
2. **데이터 Drift 탐지**
   - Evidently AI 활용
3. **사용자 피드백 루프**
   - 사용자의 thumbs-up/down 기록을 통해 성능 개선
   - 재학습 조건 충족 시 자동 트리거되는 학습 파이프라인 구성

---

## **프로젝트 실행 방법**

본 프로젝트는 웹 서비스 형태로 배포하지 않아도 되며,  
**로컬 환경 또는 클라우드 인스턴스에서 터미널 기반으로 실행** 가능합니다.

```bash
# 1. 프로젝트 클론
git clone https://github.com/your-org/langchain-qa-project.git
cd langchain-qa-project

# 2. 가상환경 설정 및 패키지 설치
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 환경 변수 설정
export OPENAI_API_KEY=your-api-key

# 4. 실행
python main.py
```

---

## **활용 장비 및 사용 툴**

### **활용 장비**
- **서버:** AWS EC2 (m5.large), S3, ECR
- **개발 환경:** Ubuntu 22.04, Python 3.10+
- **테스트 환경:** NVIDIA V100 GPU 서버 (Lambda Labs 등)

### **협업 툴**
- **소스 관리:** GitHub
- **프로젝트 관리:** Jira, Confluence
- **커뮤니케이션:** Slack
- **버전 관리:** Git

### **사용 도구**
- **CI/CD:** GitHub Actions, Jenkins
- **LLM 통합:** LangChain, OpenAI API, HuggingFace
- **실험 관리:** MLflow, Optuna
- **데이터 관리:** DVC, AWS S3
- **모니터링:** Prometheus, Grafana, ELK Stack
- **배포 및 운영:** Docker, Kubernetes, Helm

---

## **기대 효과 및 향후 계획**
- 문서 기반 질문 응답 자동화로 고객 응대 시간 절감
- 사내 문서 검색 정확도 및 사용성 향상
- 향후 다양한 도메인 문서(QA, 정책, 교육자료 등)에 확장 적용 예정

---
## **강사님 피드백 및 프로젝트 회고**

프로젝트 진행 중 담당 강사님과의 피드백 세션을 통해 얻은 주요 인사이트는 다음과 같습니다.

### 📌 **1차 피드백 (YYYY.MM.DD)**
- **LangChain Retriever 선택 기준 설명이 부족**  
  → 다양한 Retriever 종류에 대해 비교 분석하고, 왜 특정 벡터 DB(RAG with FAISS 등)를 선택했는지 근거 추가.
- **실제 유저 시나리오 고려 부족**  
  → 단순 기술 데모를 넘어서, 사용자의 입력 방식, 오답 처리 UX 흐름까지 고려한 API 설계 제안.

### 📌 **2차 피드백 (YYYY.MM.DD)**
- **MLOps 구성요소 간 연결 시각화 부족**  
  → MLflow, DVC, CI/CD, 모니터링 툴들이 어떻게 유기적으로 연결되는지 다이어그램 추가.
- **재학습(Loop) 조건 불명확**  
  → 어떤 기준으로 재학습이 트리거되는지 수치 기반 조건 정리 필요 (예: 정확도 70% 미만 시 재학습 등).

### 📌 **3차 피드백 (YYYY.MM.DD)**
- **API 보안 및 접근 제어 미흡**  
  → 인증 토큰 기반 접근 제어 및 요청 제한 정책 도입 제안.
- **프롬프트 설계 최적화 피드백**  
  → 단순 질문-응답 프롬프트가 아닌, 문맥 유지형 시스템 메시지 설계 제안.
