import pickle
import sys
from pathlib import Path
from typing import List

import openai
from langchain_core.documents import Document

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)
from config import CAFE_MENU_FILE

from dotenv import load_dotenv

load_dotenv()

# OpenAI 클라이언트 초기화
client = openai.Client()

def load() -> List[Document]:
    """
    1단계: 문서 로딩
    - 텍스트 파일을 읽어서 Document 객체로 변환
    """
    file_path = CAFE_MENU_FILE
    knowledge: str = open(file_path, "rt", encoding="utf-8").read()
    docs = [
        Document(
            # 의미있는 메타데이터가 있다면, 맘껏 더 담으시면 됩니다.
            metadata={"source": file_path},
            page_content=knowledge,
        )
    ]
    return docs

def split(src_doc_list: List[Document]) -> List[Document]:
    """
    2단계: 문서 분할
    - 긴 문서를 작은 청크로 분할 (여기서는 \\n\\n 기준으로 분할)
    """
    new_doc_list = []
    for doc in src_doc_list:
        for new_page_content in doc.page_content.split("\\n\\n"):
            new_doc_list.append(
                Document(
                    metadata=doc.metadata.copy(),  # 메타데이터 복사
                    page_content=new_page_content,  # 분할된 내용
                )
            )
    return new_doc_list

class VectorStore(list):
    """
    4단계: 벡터 저장소 클래스
    - list를 상속받아 벡터 데이터를 저장하고 관리
    """
    # 지식에 사용한 임베딩 모델과 질문에 사용할 임베딩 모델은 동일해야만 합니다.
    # 각각 임베딩 모델명을 지정하지 않고, 임베딩 모델명을 클래스 변수로 선언하여
    # 모델명 변경의 용이성을 확보합니다.
    embedding_model = "text-embedding-3-small"

    @classmethod
    def make(cls, doc_list: List[Document]) -> "VectorStore":
        """
        3단계: 임베딩 생성 및 벡터 저장소 생성
        - 문서 리스트를 받아서 벡터 데이터 리스트를 생성
        """
        vector_store = cls()  # VectorStore 인스턴스 생성

        for doc in doc_list:
            # OpenAI API를 사용하여 문서 내용을 벡터로 변환
            response = client.embeddings.create(
                model=cls.embedding_model,
                input=doc.page_content,
            )
            # 벡터 저장소에 문서와 임베딩을 함께 저장
            vector_store.append(
                {
                    "document": doc.model_copy(),  # 원본 문서의 독립적인 복사본
                    "embedding": response.data[0].embedding,  # 생성된 임베딩 벡터
                }
            )

        return vector_store

    def save(self, vector_store_path: Path) -> None:
        """
        벡터 저장소 저장
        - 현재의 벡터 데이터 리스트를 지정 경로에 파일로 저장
        """
        with vector_store_path.open("wb") as f:
            # 리스트(self)를 pickle 포맷으로 파일(f)에 저장
            pickle.dump(self, f)

    @classmethod
    def load(cls, vector_store_path: Path) -> "VectorStore":
        """
        벡터 저장소 로딩
        - 지정 경로에 저장된 파일을 읽어서 벡터 데이터 리스트를 반환
        """
        with vector_store_path.open("rb") as f:
            # pickle 포맷으로 파일(f)에서 리스트(VectorStore)를 로딩
            return pickle.load(f)

    # TODO: 이어서 구현할 예정입니다.
    # 질의 문자열을 받아서, 벡터 스토어에서 유사 문서를 최대 k개 반환
    # def search(self, question: str, k: int = 4) -> List[Document]:

def main():
    """
    메인 실행 함수
    - 벡터 저장소 생성 또는 로딩 후 사용
    """
    vector_store_path = Path(project_root) / "4-store" / "data" / "vector_store.pickle"

    # 지정 경로에 파일이 없으면
    # 문서를 로딩하고 분할하여 벡터 데이터를 생성하고 해당 경로에 저장합니다.
    if not vector_store_path.is_file():
        print("if not 문 진입")
        # 1단계: 문서 로딩
        doc_list = load()
        print(f"loaded {len(doc_list)} documents")

        # 2단계: 문서 분할
        doc_list = split(doc_list)
        print(f"split into {len(doc_list)} documents")

        # 3단계: 임베딩 생성 및 4단계: 벡터 저장소 생성
        vector_store = VectorStore.make(doc_list)

        # 벡터 저장소를 파일로 저장
        vector_store.save(vector_store_path)
        print(f"created {len(vector_store)} items in vector store")

    # 지정 경로에 파일이 있으면, 로딩하여 VectorStore 객체를 복원합니다.
    else:
        print("else 문 진입")
        vector_store = VectorStore.load(vector_store_path)
        print(f"loaded {len(vector_store)} items in vector store")

    # TODO: RAG를 통해 지식에 기반한 AI 답변을 구해보겠습니다.
    question = "빽다방 카페인이 높은 음료와 가격은?"
    print(f"RAG를 통해 '{question}' 질문에 대해서 지식에 기반한 AI 답변을 구해보겠습니다.")

if __name__ == "__main__":
    main()
