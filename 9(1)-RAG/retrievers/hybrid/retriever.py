"""Hybrid retriever using Elasticsearch RRF (Reciprocal Rank Fusion).

Combines BM25 text search with dense vector kNN search.
Uses ES 8.14+ RRF support with rank_constant=60.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# 상위 경로 모듈 인식 문제 방지
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ingest.embedding import embed_query

load_dotenv()

INDEX_NAME = "wiki-hybrid"


def get_es_client() -> Elasticsearch:
    return Elasticsearch(
        os.getenv("ELASTIC_ENDPOINT"),
        api_key=os.getenv("ELASTIC_API_KEY"),
        request_timeout=30,
    )


def search(query: str, top_k: int = 10, candidate_size: int = 50) -> list[dict]:
    """RRF hybrid search combining BM25 + kNN."""
    es = get_es_client()

    # 1. 쿼리 벡터화
    query_vector = embed_query(query)

    # 2. ES RRF Retriever 쿼리 구성
    retriever_query = {
        "rrf": {
            "retrievers": [
                {
                    "standard": {
                        "query": {
                            "match": {
                                "text": query
                            }
                        }
                    }
                },
                {
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_vector,
                        "k": candidate_size,
                        "num_candidates": candidate_size * 2,
                    }
                },
            ],
            "rank_constant": 60,
            "rank_window_size": candidate_size,
        }
    }

    response = es.search(index=INDEX_NAME, retriever=retriever_query, size=top_k)

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "id": hit["_id"],
            "text": hit["_source"]["text"],
            "score": float(hit["_score"]),
            "method": "Hybrid (RRF)",
        })

    return results


if __name__ == "__main__":
    # 간단한 작동 테스트
    sample_results = search("Who suggested Lincoln grow a beard?", top_k=3)
    print(sample_results)