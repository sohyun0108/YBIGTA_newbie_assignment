"""Ingest corpus into Elasticsearch BM25 index (wiki-bm25).

Index mapping: text field only (no vectors).
Bulk chunk_size=500 (lightweight without vectors).
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

# 상위 경로 모듈 인식 문제 방지
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

load_dotenv()

INDEX_NAME = "wiki-bm25"
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

INDEX_MAPPINGS = {
    "properties": {
        "text": {"type": "text", "analyzer": "standard"},
    }
}


def get_es_client() -> Elasticsearch:
    return Elasticsearch(
        os.getenv("ELASTIC_ENDPOINT"),
        api_key=os.getenv("ELASTIC_API_KEY"),
        request_timeout=60,
    )


def _generate_actions(corpus_path: Path):
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            yield {
                "_index": INDEX_NAME,
                "_id": doc["id"],
                "_source": {
                    "text": doc["text"],
                },
            }


def ingest(progress_callback=None):
    """Create BM25 index and bulk-ingest corpus into Elasticsearch."""
    es = get_es_client()
    corpus_path = RAW_DIR / "corpus.jsonl"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}. Run download.py first.")

    # 1. 기존 인덱스가 있으면 삭제 후 재생성
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")

    es.indices.create(index=INDEX_NAME, mappings=INDEX_MAPPINGS)
    print(f"Created index: {INDEX_NAME}")

    # 2. 문서 개수 확인 (tqdm용)
    total_docs = sum(1 for _ in open(corpus_path, encoding="utf-8"))

    # 3. Bulk 적재
    print("Ingesting corpus into Elasticsearch (BM25)...")
    success_count, _ = bulk(
        es,
        _generate_actions(corpus_path),
        chunk_size=500,
        stats_only=True,
    )

    # 4. 인덱스 새로고침
    es.indices.refresh(index=INDEX_NAME)
    print(f"Successfully indexed {success_count} / {total_docs} documents.")

    if progress_callback:
        progress_callback(success_count)

    return success_count


if __name__ == "__main__":
    ingest()