import environ
import openai

env = environ.Env() # 환경변수 설정을 위한 객체 생성
environ.Env.read_env() # .env 파일을 읽어 환경변수 설정


def print_prices(input_tokens: int, output_tokens: int) -> None:
    """
    입력 및 출력 토큰 수를 받아, 각각의 토큰 가격(원화 기준)을 계산하여 출력합니다.
    Args:
        input_tokens (int): 입력에 사용된 토큰 수.
        output_tokens (int): 출력에 사용된 토큰 수.
    출력 예시:
        input: tokens 1000, krw 0.2250
        output: tokens 500, krw 0.450000
    참고:
        - 입력 토큰 가격: 0.150 USD / 1,000,000 tokens, 환율 1,500원/USD 적용
        - 출력 토큰 가격: 0.600 USD / 1,000,000 tokens, 환율 1,500원/USD 적용
    """
    input_price = (input_tokens * 0.150 / 1_000_000) * 1_500
    output_price = (output_tokens * 0.600 / 1_000_000) * 1_500
    print("input: tokens {}, krw {:.4f}".format(input_tokens, input_price))
    print("output: tokens {}, krw {:4f}".format(output_tokens, output_price))


# make_ai_message 함수는 OpenAI API를 사용하여 질문에 대한 답변을 생성합니다.
def make_ai_message(question: str) -> str:
    client = openai.Client()  # OPENAI_API_KEY 환경변수를 디폴트로 참조

    res = client.chat.completions.create(
        messages=[
            { "role": "user", "content": question },
        ],
        model="gpt-4o-mini",
        temperature=0,
    )
    if res.usage is not None:
        print_prices(res.usage.prompt_tokens, res.usage.completion_tokens)
    else:
        print("Usage information is not available.")
    return res.choices[0].message.content or ""


def main():
    with open("data/cafe_menu.txt", "r", encoding="utf-8") as file:
        knowledge = file.read()    
    
    question = f"""
    넌 AI Assistant. 모르는 건 모른다고 대답.빽다방 카페인이 높은 음료와 가격은?

    [[빽다방 메뉴 정보]]
    {knowledge}"
    
    질문: 빽다방 카페인이 높은 음료와 가격은?
    """
    ai_message = make_ai_message(question)
    print(ai_message)

if __name__ == "__main__":
    main()