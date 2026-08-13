"""Vector retriever using Pinecone (cosine similarity)."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

# 상위 경로 모듈 인식 문제 방지
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ingest.embedding import embed_query

load_dotenv()


def search(query: str, top_k: int = 10) -> list[dict]:
    """Vector cosine similarity search."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "rag-index")

    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # 1. 쿼리 벡터 변환
    query_vector = embed_query(query)

    # 2. Pinecone Vector 검색 실행
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    results = []
    for match in response["matches"]:
        metadata = match.get("metadata", {})
        results.append({
            "id": match["id"],
            "text": metadata.get("text", ""),
            "score": float(match["score"]),
            "method": "Vector",
        })

    return results


if __name__ == "__main__":
    # 간단한 작동 테스트
    sample_results = search("Who suggested Lincoln grow a beard?", top_k=3)
    print(sample_results)