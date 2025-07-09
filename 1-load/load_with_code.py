import sys
from pathlib import Path
from pprint import pprint
from typing import List

from langchain_core.documents import Document

project_root = str(Path(__file__).resolve().parent.parent)
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

doc_list = load()
print(f"loaded {len(doc_list)} documents")
pprint(doc_list)