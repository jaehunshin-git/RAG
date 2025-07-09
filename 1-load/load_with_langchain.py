import sys
from pathlib import Path
from pprint import pprint
from typing import List

# 예전에는 `langchain` 라이브러리 기본에서 다양한 `Loader`를 지원했지만,
# 요즘은 `langchain-community` 라이브러리 등 외부 라이브러리로 지원하는 경우가 많습니다.
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from config import CAFE_MENU_FILE

# 앞선 "파이썬 코드로 직접 문서 변환" 코드와 동일한 동작
def load() -> List[Document]:
    file_path = CAFE_MENU_FILE
    loader = TextLoader(file_path=file_path, encoding='utf-8')
    docs: List[Document] = loader.load()
    return docs

doc_list = load()
print(f"loaded {len(doc_list)} documents")
pprint(doc_list)