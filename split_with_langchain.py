from typing import List
from pprint import pprint
from langchain_core.documents import Document

# 예전에는 `langchain` 라이브러리 기본에서 다양한 `Loader`를 지원했지만,
# 요즘은 `langchain-community` 라이브러리 등 외부 라이브러리로 지원하는 경우가 많습니다.
from langchain_community.document_loaders import TextLoader

# `langchain` 라이브러리의 텍스트 분할 기능을 사용하기 위해
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 앞선 "파이썬 코드로 직접 문서 변환" 코드와 동일한 동작
def load() -> List[Document]:
    file_path = r'C:\Users\JHSHIN\ProgrammingCodes\RAG\cafe_menu.txt'
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