# 표준 라이브러리
import sys
from pathlib import Path
from pprint import pprint
from typing import List

# 서드파티 라이브러리
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 로컬 애플리케이션
# 프로젝트 루트를 시스템 경로에 추가
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import CAFE_MENU_FILE

# 앞선 "파이썬 코드로 직접 문서 변환" 코드와 동일한 동작
def load() -> List[Document]:
    file_path = CAFE_MENU_FILE
    loader = TextLoader(file_path=file_path, encoding='utf-8')
    docs: List[Document] = loader.load()
    return docs

# `langchain` 라이브러리의 `RecursiveCharacterTextSplitter`를 사용 
def split(src_doc_list: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=140,  # 문서를 나눌 최소 글자 수 (디폴트: 4000)
        chunk_overlap=0,  # 문서를 나눌 때 겹치는 글자 수 (디폴트: 200)
    )
    new_doc_list = text_splitter.split_documents(src_doc_list)
    return new_doc_list

doc_list = load()
print(f"loaded {len(doc_list)} documents")
doc_list = split(doc_list)
print(f"split into {len(doc_list)} documents")
pprint(doc_list)