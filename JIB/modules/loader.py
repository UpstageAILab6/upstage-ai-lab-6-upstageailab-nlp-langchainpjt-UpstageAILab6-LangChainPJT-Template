# modules/loader.py

from pathlib import Path
from typing import List
import fitz  # PyMuPDF

def load_paragraphs_from_txt_dir(txt_dir: str) -> List[str]:
    """
    주어진 디렉토리에서 .txt 파일들을 읽고, 문단 단위로 분리하여 리스트로 반환합니다.
    문단 구분은 두 줄 개행("\n\n")을 기준으로 합니다.
    """
    all_paragraphs = []
    for file in Path(txt_dir).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            chunks = [p.strip() for p in f.read().split("\n\n") if p.strip()]
            all_paragraphs.extend(chunks)
    return all_paragraphs

def load_pdf_to_paragraphs(pdf_path: str) -> List[str]:
    """
    단일 PDF 파일을 열어 텍스트를 추출하고 문단 단위로 분리하여 반환합니다.
    문단 구분은 두 줄 개행("\n\n")을 기준으로 합니다.
    """
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    return paragraphs

def load_all_documents_from_dir(data_dir: str) -> List[str]:
    """
    주어진 디렉토리 내의 모든 .txt 및 .pdf 파일을 로드하여 문단 단위 리스트로 반환합니다.
    """
    all_paragraphs = []

    for path in Path(data_dir).glob("*"):
        if path.suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                chunks = [p.strip() for p in f.read().split("\n\n") if p.strip()]
                all_paragraphs.extend(chunks)

        elif path.suffix == ".pdf":
            doc = fitz.open(path)
            full_text = "\n".join([page.get_text() for page in doc])
            doc.close()
            chunks = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            all_paragraphs.extend(chunks)

    return all_paragraphs
