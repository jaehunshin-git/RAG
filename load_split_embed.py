from typing import List, Dict
from pprint import pprint
from langchain_core.documents import Document
import openai
import environ

env = environ.Env()
environ.Env.read_env(overwrite=True)  # .env 파일을 환경변수로 로딩합니다.

client = openai.Client()

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




def embed(doc_list: List[Document]) -> List[Dict]:
    vector_store = []

    for doc in doc_list:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=doc.page_content,
        )
        vector_store.append(
            {
                "document": doc.model_copy(),
                "embedding": response.data[0].embedding,
            }
        )

    return vector_store

doc_list = load()
print(f"loaded {len(doc_list)} documents")
doc_list = split(doc_list)
print(f"split into {len(doc_list)} documents")
# pprint(doc_list)

vector_store = embed(doc_list)
print(f"created {len(vector_store)} items in vector store")
for row in vector_store:
    print(
        "{}... => {} 차원, {} ...".format(
            row["document"].page_content[:10],
            len(row["embedding"]),
            row["embedding"][:2],
        )
    )