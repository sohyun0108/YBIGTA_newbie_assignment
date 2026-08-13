"""BM25 retriever using Elasticsearch."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# 상위 경로 모듈 인식 문제 방지
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

load_dotenv()

INDEX_NAME = "wiki-bm25"


def get_es_client() -> Elasticsearch:
    return Elasticsearch(
        os.getenv("ELASTIC_ENDPOINT"),
        api_key=os.getenv("ELASTIC_API_KEY"),
        request_timeout=30,
    )


def search(query: str, top_k: int = 10) -> list[dict]:
    """BM25 match search."""
    es = get_es_client()

    # BM25 match 쿼리 작성
    search_query = {
        "query": {
            "match": {
                "text": query
            }
        },
        "size": top_k,
    }

    response = es.search(index=INDEX_NAME, body=search_query)

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "id": hit["_id"],
            "text": hit["_source"]["text"],
            "score": float(hit["_score"]),
            "method": "BM25",
        })

    return results


if __name__ == "__main__":
    # 간단한 작동 테스트
    sample_results = search("Who suggested Lincoln grow a beard?", top_k=3)
    print(sample_results)