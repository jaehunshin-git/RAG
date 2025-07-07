from typing import List, Dict
import openai
import environ

env = environ.Env()
environ.Env.read_env(overwrite=True)  # .env 파일을 환경변수로 로딩합니다.


def embed_text(text: str) -> List[float]:
    client = openai.Client()
    res = client.embeddings.create(
        model="text-embedding-3-small", input=text  # 1536 차원
    )

    return res.data[0].embedding


text_list = ["오렌지", "설탕 커피", "카푸치노", "coffee"]
vector_list = [embed_text(text) for text in text_list]

for text, vector in zip(text_list, vector_list):
    print(f"{text} => {len(vector)} 차원 : {vector[:2]}")
