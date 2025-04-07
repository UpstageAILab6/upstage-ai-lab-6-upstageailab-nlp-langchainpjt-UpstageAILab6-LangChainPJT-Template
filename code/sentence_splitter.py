from typing import List
import re

def split_text_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in sentences if s.strip()]