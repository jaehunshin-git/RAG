# 표준 라이브러리
import pickle
import sys
from pathlib import Path
from pprint import pprint
from typing import List, cast

import numpy as np
# 서드파티 라이브러리
import openai
from dotenv import load_dotenv
from langchain_community.utils.math import cosine_similarity
from langchain_core.documents import Document
from openai.types.chat import ChatCompletionMessageParam

# 로컬 애플리케이션
# 프로젝트 루트를 시스템 경로에 추가
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import CAFE_MENU_FILE
from print_price import print_prices


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
    - 긴 문서를 작은 청크로 분할 (여기서는 \n\n 기준으로 분할)
    """
    new_doc_list = []
    for doc in src_doc_list:
        for new_page_content in doc.page_content.split("\n\n"):
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

    # 질의 문자열을 받아서, 벡터 스토어에서 유사 문서를 최대 k개 반환
    def search(self, question: str, k: int = 4) -> List[Document]:

        # OpenAI API를 사용하여 질문을 임베딩 벡터로 변환합니다.
        response = client.embeddings.create(
            model=self.embedding_model,  # 클래스 변수에 지정된 임베딩 모델을 사용합니다.
            input=question,  # 사용자의 질문 문자열을 입력으로 넣습니다.
        )
        # API 응답에서 실제 임베딩 벡터를 추출합니다.
        question_embedding = response.data[0].embedding

        # 벡터 저장소(self)에 있는 모든 문서의 임베딩 벡터만 추출하여 리스트를 생성합니다.
        embedding_list = [row["embedding"] for row in self]

        # 질문 임베딩과 저장된 모든 문서 임베딩 간의 코사인 유사도를 계산합니다.
        similarities = cosine_similarity([question_embedding], embedding_list)[0]

        # 계산된 유사도를 내림차순으로 정렬하여 가장 유사한 문서 k개의 인덱스를 가져옵니다.
        top_indices = np.argsort(similarities)[::-1][:k]

        # 가장 유사한 인덱스에 해당하는 문서들의 복사본을 리스트로 만들어 반환합니다.
        return [
            self[idx]["document"].model_copy()  # 원본 데이터 보호를 위해 복사본을 생성합니다.
            for idx in top_indices
        ]


def main():
    vector_store_path = Path(project_root) / "5-search" / "data" / "vector_store.pickle"

    # 첫번째 실행에서는 vector_store.pickle 파일이 없으므로 load, split, make, save 순서로 데이터를 생성하고 저장합니다.
    if not vector_store_path.is_file():
        doc_list = load()
        print(f"loaded {len(doc_list)} documents")
        doc_list = split(doc_list)
        print(f"split into {len(doc_list)} documents")
        vector_store = VectorStore.make(doc_list)
        vector_store.save(vector_store_path)
        print(f"created {len(vector_store)} items in vector store")
    # 이후 실행에서는 vector_store.pickle 파일이 있으므로 load 순서로 데이터를 로딩합니다.
    else:
        vector_store = VectorStore.load(vector_store_path)
        print(f"loaded {len(vector_store)} items in vector store")

    question = "빽다방 카페인이 높은 음료와 가격은?"

    search_doc_list: List[Document] = vector_store.search(question)
    pprint(search_doc_list)

    print("## Knowledge ##")
    knowledge: str = str(search_doc_list)
    print(repr(knowledge))

    res = client.chat.completions.create(
        messages=cast(
            List[ChatCompletionMessageParam],
            [
                {
                    "role": "system",
                    "content": f"넌 AI Assistant. 모르는 건 모른다고 대답.\n\n[[빽다방 메뉴 정보]]\n{knowledge}",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        ),
        model="gpt-4o-mini",
        temperature=0,
    )
    print_prices(res.usage.prompt_tokens, res.usage.completion_tokens)
    ai_message = res.choices[0].message.content

    print("[AI]", ai_message)


if __name__ == "__main__":
    main()