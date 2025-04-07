import pandas as pd
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.fields = ['지원대상', '지원내용', '신청방법', '접수기관명', '선정기준', '문의처']

    def _clean_text(self, text):
        return str(text).replace('\r', ' ').replace('\n', ' ').replace('○', '').strip()

    def _extract_chunks_from_row(self, row) -> list[Document]:
        service_name = self._clean_text(row.get("서비스명", ""))
        service_id = row.get("서비스ID", "")
        base_metadata = {"서비스ID": service_id}

        chunks = []
        for field in self.fields:
            value = self._clean_text(row.get(field, ""))
            if value and value.lower() != "nan":
                chunk_text = f"[정책명: {service_name}] [항목: {field}]\n{value}"
                chunks.append(Document(page_content=chunk_text, metadata=base_metadata))

        return chunks

    def load_documents(self) -> list[Document]:
        df = pd.read_csv(self.filepath)
        documents = []
        for _, row in df.iterrows():
            documents.extend(self._extract_chunks_from_row(row))
        print(f"총 {len(documents)}개의 문서 생성 완료")
        return documents
