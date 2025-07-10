import os.path
# 로컬 애플리케이션
# 프로젝트 루트를 시스템 경로에 추가
import sys
from pathlib import Path
from uuid import uuid4

import faiss
from dotenv import load_dotenv
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
from config import CAFE_MENU_FILE

load_dotenv()

faiss_folder_path = project_root / "6-langchain" / "data"/ "faiss_index"

embedding = OpenAIEmbeddings(model="text-embedding-3-small")


def get_vector_store() -> VectorStore:
    if not os.path.exists(faiss_folder_path):
        doc_list = TextLoader(file_path=CAFE_MENU_FILE, encoding="utf-8").load()
        print(f"loaded {len(doc_list)} documents")  # 1

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=140,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=True,
        )
        doc_list = text_splitter.split_documents(doc_list)
        print(f"split into {len(doc_list)} documents")  # 9

        dimension = len(embedding.embed_query("hello"))  # 1536
        # 차원수 = 1536

        index = faiss.IndexFlatL2(dimension)

        vector_store = FAISS(
            embedding_function=embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        uuids = [str(uuid4()) for _ in range(len(doc_list))]
        vector_store.add_documents(documents=doc_list, ids=uuids)

        vector_store.save_local(faiss_folder_path)
    else:
        vector_store = FAISS.load_local(
            faiss_folder_path,
            embedding,
            allow_dangerous_deserialization=True,
        )

    return vector_store


def main():
    vector_store = get_vector_store()

    question = "빽다방 카페인이 높은 음료와 가격은?"

    # 직접 similarity_search 메서드 호출을 통한 유사 문서 검색
    # search_doc_list = vector_store.similarity_search(question)
    # pprint(search_doc_list)

    # retriever 인터페이스를 통한 유사 문서 검색
    # retriever = vector_store.as_retriever()
    # search_doc_list = retriever.invoke(question)
    # pprint(search_doc_list)

    # Chain을 통한 retriever 자동 호출
    # llm = ChatOpenAI(model_name="gpt-4o-mini")
    # retriever = vector_store.as_retriever()
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    # ai_message = qa_chain.invoke(question)
    # print("[AI]", ai_message["result"]) # keys: "query", "result"

    llm = ChatOpenAI(model_name="gpt-4o-mini")
    retriever = vector_store.as_retriever()
    prompt_template = PromptTemplate(
        template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["context", "question"],
    )

    rag_pipeline = (
            RunnableLambda(
                # 아래 invoke를 통해 전달되는 값이 인자로 전달됩니다.
                lambda x: {
                    "context": retriever.invoke(x),
                    "question": x,
                }
            )
            | prompt_template
            | llm
    )
    ai_message: AIMessage = rag_pipeline.invoke(question)
    print("[AI]", ai_message.content)  # AIMessage 타입
    print(ai_message.usage_metadata)


if __name__ == "__main__":
    main()