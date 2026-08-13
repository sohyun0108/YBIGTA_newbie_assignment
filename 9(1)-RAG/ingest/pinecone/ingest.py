"""Ingest embeddings into Pinecone vector index.

Batch upsert: 100 vectors per call.
Metadata: text truncated to 1000 chars (40KB limit).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

# 상위 경로 모듈 인식 문제 방지
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

load_dotenv()

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

BATCH_SIZE = 100
TEXT_LIMIT = 1000  # metadata text truncation


def ingest(progress_callback=None):
    """Batch upsert embeddings into Pinecone vector index."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "ragsession")

    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    corpus_path = RAW_DIR / "corpus.jsonl"
    embeddings_path = PROCESSED_DIR / "embeddings.npy"
    ids_path = PROCESSED_DIR / "embedding_ids.json"

    # 파일 존재 확인
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}")
    if not embeddings_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Embeddings or IDs file not found. Run ingest/embedding.py first.")

    print("Loading cached embeddings and corpus...")
    embeddings = np.load(embeddings_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))

    # ID를 key로 하는 본문 텍스트 매핑 생성 (메타데이터용)
    id_to_text = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            id_to_text[doc["id"]] = doc["text"][:TEXT_LIMIT]

    total_vectors = len(ids)
    print(f"Upserting {total_vectors} vectors to Pinecone index '{index_name}'...")

    # Batch Upsert
    for i in tqdm(range(0, total_vectors, BATCH_SIZE), desc="Pinecone Upsert"):
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_vectors = embeddings[i : i + BATCH_SIZE]

        vectors_to_upsert = []
        for doc_id, vector in zip(batch_ids, batch_vectors):
            text_metadata = id_to_text.get(doc_id, "")
            vectors_to_upsert.append({
                "id": doc_id,
                "values": vector.tolist(),
                "metadata": {"text": text_metadata}
            })

        index.upsert(vectors=vectors_to_upsert)

        if progress_callback:
            progress_callback(min(i + BATCH_SIZE, total_vectors), total_vectors)

    print(f"Successfully upserted {total_vectors} vectors into Pinecone.")
    return total_vectors


if __name__ == "__main__":
    ingest()