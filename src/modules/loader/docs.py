import csv
import pandas as pd
from typing import Any, Optional, List


class Docs:
    """
    Document 객체는 크롤링한 내용(content), 원본 데이터(source),
    그리고 연관 파일들(attached_file)을 관리합니다.
    """

    def __init__(self, content: Any, source: Any, document_type: str, attached_file: Optional[List[Any]] = None):
        self.content = content
        self.source = source
        self.document_type = document_type
        self.attached_file = attached_file if attached_file is not None else []

    def __repr__(self):
        return f"Document(content={self.content!r}, source={self.source!r}, attached_file={self.attached_file!r})"

    def to_pandas(self) -> pd.DataFrame:
        """
        Document 객체를 pandas DataFrame으로 변환합니다.
        :return: pandas DataFrame
        """
        data = {
            'content': [self.content],
            'source': [self.source],
            'document_type': [self.document_type],
            'attached_file': [self.attached_file]
        }
        return pd.DataFrame(data)

    def to_csv(self, file_path: str) -> None:
        """
        주어진 파일 경로에 UTF-8 인코딩으로 CSV 파일을 작성합니다.
        :param file_path:
        :return:
        """
        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['content', 'source', 'document_type', 'attached_file'])
            # attached_files_str = '; '.join(str(item) for item in self.attached_file)
            writer.writerow([self.content, self.source, self.document_type, self.attached_file])
