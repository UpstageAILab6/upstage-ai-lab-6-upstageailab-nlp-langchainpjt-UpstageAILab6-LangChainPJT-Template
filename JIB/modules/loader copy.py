from pathlib import Path
from typing import List

def load_paragraphs_from_txt_dir(txt_dir: str) -> List[str]:
    """
    주어진 디렉토리에서 .txt 파일들을 읽고, 문단 단위로 분리하여 리스트로 반환합니다.
    문단 구분은 두 줄 개행("\n\n")을 기준으로 합니다.

    Args:
        txt_dir (str): .txt 파일들이 들어 있는 디렉토리 경로

    Returns:
        List[str]: 분리된 문단 리스트
    """
    all_paragraphs = []
    for file in Path(txt_dir).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            chunks = [p.strip() for p in f.read().split("\n\n") if p.strip()]
            all_paragraphs.extend(chunks)
    return all_paragraphs
