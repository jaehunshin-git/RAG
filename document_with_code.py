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

doc_list = load()
print(f"loaded {len(doc_list)} documents")
pprint(doc_list)