"""Upstage Solar embedding utility with disk caching and parallel API keys.

Models:
  - solar-embedding-1-large-passage  (document encoding)
  - solar-embedding-1-large-query    (query encoding)

Uses multiple API keys (UPSTAGE_API_KEY1..N) for parallel embedding.
Each key gets its own thread with independent RPM/TPM limits.
Saves progress incrementally so crashes don't lose work.
Cache: data/processed/embeddings.npy (float32) + embedding_ids.json
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
IDS_PATH = PROCESSED_DIR / "embedding_ids.json"

BATCH_SIZE = 100
RPM_LIMIT = 100
MIN_INTERVAL = 60.0 / RPM_LIMIT
DIM = 4096
BASE_URL = "https://api.upstage.ai/v1/solar"
MAX_CHARS = 12000  # ~3000 tokens, safely under 4000 token limit
MAX_RETRIES = 3


def _get_api_keys() -> list[str]:
    """Collect all UPSTAGE_API_KEY* from env."""
    keys = []
    for i in range(1, 100):
        key = os.getenv(f"UPSTAGE_API_KEY{i}")
        if key:
            keys.append(key.strip())
        else:
            break
    if not keys:
        single = os.getenv("UPSTAGE_API_KEY", "")
        if single:
            keys.append(single.strip())
    return keys


def _truncate(text: str) -> str:
    """Truncate text to stay within token limits."""
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS]
    return text


def _embed_batch_safe(client: OpenAI, batch: list[str]) -> list[list[float]]:
    """Embed a batch with retry and fallback to smaller sub-batches."""
    truncated = [_truncate(t) for t in batch]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model="solar-embedding-1-large-passage",
                input=truncated,
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            err_msg = str(e)
            if "maximum context length" in err_msg or "4000 tokens" in err_msg:
                # Split batch in half and process separately
                mid = len(truncated) // 2
                if mid == 0:
                    # Single text too long, truncate more aggressively
                    truncated = [t[: MAX_CHARS // 2] for t in truncated]
                    continue
                left = _embed_batch_safe(client, truncated[:mid])
                time.sleep(MIN_INTERVAL)
                right = _embed_batch_safe(client, truncated[mid:])
                return left + right
            elif attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                raise


def embed_passages(texts: list[str], ids: list[str], progress_callback=None) -> np.ndarray:
    """Embed passages using parallel API keys."""
    # 디렉토리가 없으면 생성
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    api_keys = _get_api_keys()
    if not api_keys:
        raise ValueError("No Upstage API key found in .env file.")

    clients = [OpenAI(api_key=key, base_url=BASE_URL) for key in api_keys]
    num_clients = len(clients)

    # 배치 단위 분할
    batches = []
    for i in range(0, len(texts), BATCH_SIZE):
        batches.append((i, texts[i : i + BATCH_SIZE]))

    all_embeddings = [None] * len(texts)
    lock = Lock()
    completed_count = 0

    def process_batch(batch_tuple, client_idx):
        nonlocal completed_count
        start_idx, batch_texts = batch_tuple
        client = clients[client_idx % num_clients]

        embeddings = _embed_batch_safe(client, batch_texts)

        with lock:
            for j, emb in enumerate(embeddings):
                all_embeddings[start_idx + j] = emb
            completed_count += len(batch_texts)
            if progress_callback:
                progress_callback(completed_count, len(texts))

        time.sleep(MIN_INTERVAL)

    # 병렬 처리 실행
    with ThreadPoolExecutor(max_workers=num_clients) as executor:
        futures = [
            executor.submit(process_batch, batch_tuple, i)
            for i, batch_tuple in enumerate(batches)
        ]
        for future in tqdm(as_completed(futures), total=len(batches), desc="Embedding passages"):
            future.result()

    embeddings_np = np.array(all_embeddings, dtype=np.float32)

    # 결과 파일 저장
    np.save(EMBEDDINGS_PATH, embeddings_np)
    IDS_PATH.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Successfully saved embeddings to {EMBEDDINGS_PATH}")
    return embeddings_np


def embed_query(query: str) -> list[float]:
    """Embed a single query using the query model."""
    api_keys = _get_api_keys()
    if not api_keys:
        raise ValueError("No Upstage API key found in .env file.")

    client = OpenAI(api_key=api_keys[0], base_url=BASE_URL)
    truncated_query = _truncate(query)

    response = client.embeddings.create(
        model="solar-embedding-1-large-query",
        input=truncated_query,
    )
    return response.data[0].embedding


def load_cached_embeddings() -> tuple[np.ndarray, list[str]] | None:
    """Load cached embeddings from disk. Returns (embeddings, ids) or None."""
    if EMBEDDINGS_PATH.exists() and IDS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        ids = json.loads(IDS_PATH.read_text())
        return embeddings, ids
    return None


if __name__ == "__main__":
    from data.download import RAW_DIR

    corpus_path = RAW_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        print("Run data/download.py first.")
        raise SystemExit(1)

    texts, ids = [], []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc["id"])
            texts.append(doc["text"])

    embed_passages(texts, ids)