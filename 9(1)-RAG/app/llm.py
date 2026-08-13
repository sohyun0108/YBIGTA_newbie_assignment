"""Solar Pro3 LLM utility for RAG answer generation.

Uses Upstage Solar API (OpenAI-compatible) with solar-pro3 model.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 상위 경로 모듈 인식 문제 방지
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

BASE_URL = "https://api.upstage.ai/v1/solar"
MODEL = "solar-mini"

NO_RAG_PROMPT = "Answer the following question concisely.\n\nQuestion: {question}"

RAG_PROMPT = """\
Answer the question based ONLY on the provided context.
If the answer is not found in the context, reply exactly: "The provided context does not contain this information."
Do NOT use any outside knowledge.

Context:
{context}

Question: {question}"""


def _get_api_key() -> str:
    """Get the first available Upstage API key."""
    key = os.getenv("UPSTAGE_API_KEY1") or os.getenv("UPSTAGE_API_KEY", "")
    return key.strip()


def generate(question: str, context: str | None = None) -> str:
    """Generate an answer using Solar LLM."""
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("Upstage API Key not found in .env file.")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # Context 유무에 따른 프롬프트 분기
    if context:
        prompt = RAG_PROMPT.format(context=context, question=question)
    else:
        prompt = NO_RAG_PROMPT.format(question=question)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # 테스트 1: No RAG
    print("--- No RAG Test ---")
    print(generate("Who suggested Lincoln grow a beard?"))

    # 테스트 2: RAG
    print("\n--- RAG Test ---")
    sample_context = "Grace Bedell wrote a letter to Abraham Lincoln suggesting he grow a beard."
    print(generate("Who suggested Lincoln grow a beard?", context=sample_context))