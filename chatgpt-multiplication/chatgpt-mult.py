import csv

from pydantic import BaseModel
from openai import OpenAI


class MathResponse(BaseModel):

    final_answer: float

    
def run_mult(a, b):

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful math assistant."
            },
            {
                "role": "user",
                "content": f"Calculate the product of {a} and {b}"
            }
        ],
        response_format=MathResponse,
    )

    message = completion.choices[0].message
    if not message.parsed:
        answer = -1
    else:
        answer = message.parsed.final_answer

    return answer


# generic numbers
a = [
    "0.0099660",
    "0.0374368",
    "0.9783497",
    "5.9283700",
    "12.484789",
    "908.17286"
]
# special numbers
b = [
    "0.09",
    "1.96",
    "0.96",
    "10.9",
    "123.0",
]

answers = []
for n1 in a:
    for n2 in b:
        answer = run_mult(n1, n2)
        print(n1, n2, answer, float(n1) * float(n2))
        answers.append((n1, n2, answer))
        
with open("answers-float.csv", "w") as fp:
    writer = csv.writer(fp)
    writer.writerow(["a", "b", "a*b"])
    writer.writerows(answers)
