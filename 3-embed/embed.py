# 표준 라이브러리
from typing import List

# 서드파티 라이브러리
import numpy as np
import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

def embed_text(text: str) -> List[float]:
    client = openai.Client()
    res = client.embeddings.create(
        model="text-embedding-3-small", input=text  # 1536 차원
    )

    return res.data[0].embedding

# 데이터 셋
text_list = ["오렌지", "설탕 커피", "카푸치노", "coffee"]
vector_list = [embed_text(text) for text in text_list]

for text, vector in zip(text_list, vector_list):
    print(f"{text} => {len(vector)} 차원 : {vector[:2]}")

# 질문 예시
question = "커피"
question_vector = embed_text(question)
print(f"{question} => {len(question_vector)} 차원 : {question_vector[:2]}")


similarity_list = cosine_similarity(np.array([question_vector]), np.array(vector_list))[0]

for text, similarity in zip(text_list, similarity_list):
    print(text, similarity)