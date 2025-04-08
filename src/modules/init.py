import ast
from typing import List

from dotenv import load_dotenv
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.modules.vector_store.vector_store import Faiss, ChromaStore
from src.modules.embedding.embedding import Embedding
from src.modules.loader.notion_loader import NotionLoader, LawLoader, LectureLoader
import csv
from langchain_core.documents import Document


def init():
    notion_url = "https://sincere-nova-ec6.notion.site/a8bbcb69d87c4c19aabee16c6a178286"
    loader = NotionLoader()
    document = loader.load(notion_url)
    md = MarkdownifyTransformer()
    converted_docs = md.transform_documents(document)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
    )
    md_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        # length_function=length_function,
    )
    embeddings = Embedding()
    vector_store = ChromaStore(embeddings.model, persist_directory="../chroma2")

    # Notion 문서 변환 및 분할
    chunked_docs = text_splitter.split_documents(converted_docs)
    if document is not None:
        vector_store.add_documents(chunked_docs)

    # 시간표
    csv_docs: List[Document] = load_schedule_csv_2('../files/schedule.csv')
    vector_store.add_documents(csv_docs)

    # 강의 시간표
    csv_docs = load_schedule_csv('../files/online_lecture.csv')
    vector_store.add_documents(csv_docs)

    # 슬랙 활용법
    md_docs = load_markdown_file('../files/slack.md')
    md_chunks = md_splitter.split_documents([md_docs])
    vector_store.add_documents(md_chunks)
    # input, output 정의가 되어야됨

    # 법
    law_docs = LawLoader().load('law.txt')
    vector_store.add_documents(law_docs)


def load_schedule_csv(csv_path) -> List[Document]:
    documents = []
    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = row.get('date', '').strip() or '날짜 없음'
            timetable = row.get('timetable', '').strip()
            content = f"Date: {date}\nTimetable: {timetable}"
            documents.append(Document(page_content=content))
    return documents


def load_markdown_file(file_path) -> Document:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return Document(page_content=content)



def load_schedule_csv_2(csv_path, pivot_month=11, next_year="2025"):
    """
    CSV 파일로부터 스케줄 데이터를 로드하여 Document 객체 목록을 반환합니다.

    스케줄의 날짜는 "MM.DD(요일)" 형식으로 입력되므로, 괄호 이전의 "MM.DD"만 추출합니다.
    스케줄 범위가 2024.11 ~ 2025.06인 경우,
      - 월이 pivot_month(예: 11) 이상이면 이전 연도(base_year - 1)를 사용 (예: 2025 -> 2024)
      - 그렇지 않으면 기본 연도(base_year)를 사용합니다.

    각 문서의 content는 timetable JSON 문자열에 "date": full_date 를 추가한 형태로 구성되며,
    메타데이터에는 'date' (정규화된 전체 날짜)와 'raw_date' (원본 날짜 문자열)를 저장합니다.
    """
    documents = []
    base_year = int(next_year)

    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            raw_date = row.get('date', '').strip() or '날짜 없음'
            # 괄호 이후의 내용을 제거 (예: "11.14(목)" -> "11.14")
            normalized_date = raw_date.split('(')[0].strip()
            try:
                month_str, day_str = normalized_date.split('.')
                month = int(month_str)
                day = int(day_str)
            except Exception as e:
                month = None
                day = None

            if month is not None:
                # 월이 pivot_month 이상이면 이전 연도 사용, 그렇지 않으면 기본 연도 사용
                full_year = base_year - 1 if month >= pivot_month else base_year
                full_date = f"{full_year}{month:02d}{day:02d}"
            else:
                full_date = "날짜 없음"

            timetable_str = row.get('timetable', '').strip()
            try:
                # timetable 문자열을 안전하게 딕셔너리로 변환
                timetable_dict = ast.literal_eval(timetable_str)
            except Exception as e:
                timetable_dict = {}

            # timetable 딕셔너리에 날짜 정보 추가
            timetable_dict["date"] = full_date

            # content는 timetable 딕셔너리를 문자열로 변환한 값
            content = str(timetable_dict)
            metadata = {
                "search_date": full_date,  # 예: "2024-11-18"
            }

            documents.append(Document(page_content=content, metadata=metadata))
    return documents


if __name__ == "__main__":
    load_dotenv()
    init()
