import datetime
from typing import Any, List, Tuple

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import BaseModel


def extract_context_and_files(result) -> Tuple[str, str]:
    """
    검색 결과에서 중복 없이 context와 attached_file 정보를 추출합니다.
    """
    unique_context = []
    unique_attached_files = set()

    for chunk in result:
        content_text = chunk.page_content
        source = chunk.metadata.get("source", "Unknown")
        context_entry = f"source: {source} , contents: {content_text}"
        if context_entry not in unique_context:
            unique_context.append(context_entry)

        for file in chunk.metadata.get("attached_file", []):
            unique_attached_files.add(file)

    combined_context = "\n\n".join(unique_context)
    attached_files_str = "\n".join(sorted(unique_attached_files)) if unique_attached_files else "None"

    return combined_context, attached_files_str


def handle_vacation_search(vector_store: Any, question: str) -> Tuple[str, str]:
    """
    실제로 vacation 검색을 실행하고,
    문맥(context) 및 파일 정보까지 추출해 반환.
    여기에 전처리/후처리 로직도 포함할 수 있음.
    """
    result = vector_store.similarity_search(question)
    context_text, attached_files = extract_context_and_files(result)

    return context_text, attached_files


# ------
class DateArray(BaseModel):
    dates: List[str]


def handle_timetable_search(vector_store: Any, question: str) -> Tuple[str, str]:
    # 1. 오늘 날짜를 "YYYY-MM-DD" 형식으로 계산
    # 한국 요일 배열(월, 화, 수, 목, 금, 토, 일)
    weekdays_kr = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

    today = datetime.date.today()
    # '20250407(월)' 형태의 문자열 생성
    today_str = today.strftime("%Y%m%d") + f"({weekdays_kr[today.weekday()]})"
    today_str_short = today.strftime("%Y%m%d")

    # 2. 프롬프트 템플릿 생성
    prompt = PromptTemplate(
        input_variables=["question", "today"],
        template="""
오늘 날짜는 {today}입니다.

# 주석: 모델이 내부적으로 계산 시 참고하세요.
# 하지만 최종 출력 시에는 아래 형식을 정확히 준수해 주세요:
# {{
#   "dates": ["YYYYMMDD", "YYYYMMDD", ...]
# }}

질문: {question}

지금부터는 오직 JSON 형태로만, 아래 구조에 맞게 출력해 주세요:
        """
    )

    # 3. LLM 및 출력 파서 초기화
    # 필요에 따라 ChatUpstage(model="solar-pro", temperature=0.7)로 교체 가능
    llm = ChatOpenAI(
        temperature=1,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o",  # 모델명
    )
    output_parser = PydanticOutputParser(pydantic_object=DateArray)

    # 4. 체인 구성: 프롬프트 -> LLM -> 출력 파서
    chain = prompt | llm | output_parser

    # 5. 입력 데이터 준비 및 체인 실행

    inputs = {
        "question": question,
        "today": today_str
    }
    refined_output = chain.invoke(inputs)

    # 6. 각 날짜별 유사 문서 검색 및 결과 취합
    combined_results = []
    for date in refined_output.dates:
        results = vector_store.similarity_search(date, filter={"search_date": date})
        combined_results.extend(results)

    # 7. 취합된 결과에서 context_text와 files 정보를 추출
    context_text, files = extract_context_and_files(combined_results)

    return context_text, files

def handle_legal_search(vector_store: Any, question: str) -> Tuple[str, str]:
    result = vector_store.similarity_search(question)
    context_text, _ = extract_context_and_files(result)

    return context_text, ""
