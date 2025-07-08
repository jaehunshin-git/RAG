from typing import List
from pprint import pprint
from langchain_core.documents import Document

def load() -> List[Document]:
    file_path = r'C:\Users\JHSHIN\ProgrammingCodes\RAG\cafe_menu.txt'
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