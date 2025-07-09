# 표준 라이브러리
import sys
from pathlib import Path
from pprint import pprint
from typing import List

# 서드파티 라이브러리
from langchain_core.documents import Document

# 로컬 애플리케이션
# 프로젝트 루트를 시스템 경로에 추가
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import CAFE_MENU_FILE

def load() -> List[Document]:
    file_path = CAFE_MENU_FILE
    with open(file_path, 'r', encoding='utf-8') as file:
        knowledge = file.read()
    
    docs = [
        Document(
            metadata={"source": file_path},
            page_content=knowledge,
        )
    ]
    
    return docs


def split(src_doc_list: List[Document]) -> List[Document]:
    new_doc_list = []
    for doc in src_doc_list:
        for new_page_content in doc.page_content.split("\n\n"):
            new_doc_list.append(
                Document(
                    metadata=doc.metadata.copy(),
                    page_content=new_page_content,
                )
            )
    return new_doc_list

doc_list = load()
print(f"loaded {len(doc_list)} documents")
doc_list = split(doc_list)
print(f"split into {len(doc_list)} documents")
pprint(doc_list)