# lecture_loader.py

import os
import pandas as pd
from typing import List
from langchain.schema import Document
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class LectureLoader(CSVLoader):
    def fill_between_non_nan(self, df, column):
        filled_values = []
        temp_value = None
        for value in df[column]:
            if pd.notna(value):
                temp_value = value
            filled_values.append(temp_value)
        df[column] = filled_values
        return df

    def load(self, source: str) -> List[Document]:
        # CSV 원본 경로
        raw_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..",
            "files",
            source
        )

        # 전처리된 CSV 경로
        processed_path = raw_path.replace(".csv", "_v2.csv")

        # 원본 CSV 불러오기
        df = pd.read_csv(raw_path, skiprows=4)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(['Unnamed: 0'], axis=1)

        for col in df.columns:
            df = self.fill_between_non_nan(df, col)

        df.to_csv(processed_path, index=False)

        # CSVLoader로 문서 로드
        loader = CSVLoader(file_path=processed_path, csv_args={'delimiter': ','})
        docs = loader.load()

        # domain 메타데이터 추가
        for doc in docs:
            doc.metadata["domain"] = "lecture_info"

        # 문서 분할
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
        split_docs = splitter.split_documents(docs)

        return split_docs
