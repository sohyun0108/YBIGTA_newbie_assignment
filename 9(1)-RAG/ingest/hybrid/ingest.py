"""Ingest corpus into Elasticsearch Hybrid index (wiki-hybrid).

Index mapping: text field + dense_vector(4096, cosine).
Bulk chunk_size=100 (heavier with 4096-dim vectors).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

# 상위 경로 모듈 인식 문제 방지
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

load_dotenv()

INDEX_NAME = "wiki-hybrid"
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

INDEX_MAPPINGS = {
    "properties": {
        "text": {"type": "text", "analyzer": "standard"},
        "embedding": {
            "type": "dense_vector",
            "dims": 4096,
            "index": True,
            "similarity": "cosine",
        },
    }
}


def get_es_client() -> Elasticsearch:
    return Elasticsearch(
        os.getenv("ELASTIC_ENDPOINT"),
        api_key=os.getenv("ELASTIC_API_KEY"),
        request_timeout=120,
    )


def _generate_actions(corpus_path: Path, embeddings: np.ndarray, ids: list[str]):
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["id"]
            idx = id_to_idx.get(doc_id)
            if idx is None:
                continue
            yield {
                "_index": INDEX_NAME,
                "_id": doc_id,
                "_source": {
                    "text": doc["text"],
                    "embedding": embeddings[idx].tolist(),
                },
            }


def ingest(progress_callback=None):
    """Create hybrid index (text + dense_vector) and bulk-ingest corpus."""
    es = get_es_client()
    corpus_path = RAW_DIR / "corpus.jsonl"
    embeddings_path = PROCESSED_DIR / "embeddings.npy"
    ids_path = PROCESSED_DIR / "embedding_ids.json"

    # 1. 파일 존재 여부 확인
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}")
    if not embeddings_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Embeddings or IDs file not found. Run ingest/embedding.py first.")

    # 2. 임베딩 데이터 및 ID 로드
    print("Loading cached embeddings and IDs...")
    embeddings = np.load(embeddings_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))

    # 3. 기존 인덱스 삭제 후 재생성
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")

    es.indices.create(index=INDEX_NAME, mappings=INDEX_MAPPINGS)
    print(f"Created index: {INDEX_NAME}")

    # 4. Bulk 적재 (벡터가 포함되어 가벼운 chunk_size=100 설정)
    print("Ingesting corpus with embeddings into Elasticsearch (Hybrid)...")
    success_count, _ = bulk(
        es,
        _generate_actions(corpus_path, embeddings, ids),
        chunk_size=100,
        stats_only=True,
    )

    # 5. 인덱스 새로고침
    es.indices.refresh(index=INDEX_NAME)
    print(f"Successfully indexed {success_count} documents into {INDEX_NAME}.")

    if progress_callback:
        progress_callback(success_count)

    return success_count


if __name__ == "__main__":
    ingest()