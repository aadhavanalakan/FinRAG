"""
embedding.py — Nebius (Token Factory) embedder with on-disk caching.

Turns text into vectors via the Nebius /embeddings endpoint (OpenAI-compatible).
Used in two places: the semantic chunker (embed sentences to find topic breaks)
and ingest (embed chunks before storing in Pinecone).

Cost control: every embedding is cached on disk keyed by (model, text). Re-running
ingest during development re-embeds ONLY text it hasn't seen — first run costs a few
cents, later runs are basically free and much faster. Raw vectors are cached;
L2-normalization (for cosine) is applied on return, so toggling that flag needs no
re-embedding.

Smoke test:  python -m finrag.embedding
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
from openai import OpenAI

from finrag.config import EmbeddingConfig, load_config

_CACHE_PATH = Path(".cache/embeddings.db")

# Nebius caps a single /embeddings request at the model's context window (40960
# tokens for Qwen3-Embedding-8B). We batch by TOKEN budget (with margin), not a
# fixed item count, and truncate any single input that alone exceeds the budget.
_MAX_TOKENS_PER_REQUEST = 20000           # well under the 40960 hard cap, for margin
_CHARS_PER_TOKEN = 3                       # conservative: dense financial text tokenizes ~2.9 chars/token


class Embedder:
    def __init__(self, cfg: EmbeddingConfig, cache_path: Path = _CACHE_PATH) -> None:
        self.cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key())
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Streamlit serves reruns from a pool of threads,
        # and one cached Embedder is shared across them; the CLI uses a single thread.
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)"
        )
        self._db.commit()

    # --- cache plumbing ------------------------------------------------------
    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.cfg.model}\n{text}".encode()).hexdigest()

    def _get_cached(self, key: str) -> np.ndarray | None:
        row = self._db.execute("SELECT vec FROM emb WHERE key=?", (key,)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    def _put_cached(self, key: str, vec: np.ndarray) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
            (key, vec.astype(np.float32).tobytes()),
        )

    # --- API call ------------------------------------------------------------
    def _batches(self, texts: list[str]) -> list[list[str]]:
        """Pack texts into requests under both the item-count and token-budget caps."""
        char_budget = _MAX_TOKENS_PER_REQUEST * _CHARS_PER_TOKEN
        batches: list[list[str]] = []
        cur: list[str] = []
        cur_chars = 0
        for t in texts:
            t = t if len(t) <= char_budget else t[:char_budget]   # truncate giant single inputs
            if cur and (len(cur) >= self.cfg.batch_size or cur_chars + len(t) > char_budget):
                batches.append(cur)
                cur, cur_chars = [], 0
            cur.append(t)
            cur_chars += len(t)
        if cur:
            batches.append(cur)
        return batches

    def _call_api(self, texts: list[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for batch in self._batches(texts):
            resp = self._client.embeddings.create(model=self.cfg.model, input=batch)
            for d in resp.data:
                v = np.asarray(d.embedding, dtype=np.float32)
                if v.shape[0] != self.cfg.dimension:
                    raise ValueError(
                        f"{self.cfg.model} returned {v.shape[0]}-dim vectors, but config "
                        f"says {self.cfg.dimension}. Fix config/models.yaml + Pinecone index."
                    )
                out.append(v)
        return out

    # --- public API ----------------------------------------------------------
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts (cache-first), returning an (n, dim) float32 array of RAW vectors."""
        keys = [self._key(t) for t in texts]
        vecs: list[np.ndarray | None] = [self._get_cached(k) for k in keys]

        miss_idx = [i for i, v in enumerate(vecs) if v is None]
        if miss_idx:
            fresh = self._call_api([texts[i] for i in miss_idx])
            for i, v in zip(miss_idx, fresh):
                vecs[i] = v
                self._put_cached(keys[i], v)
            self._db.commit()

        return np.vstack(vecs)

    def _maybe_normalize(self, arr: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize_embeddings:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages/chunks (no query prefix). Returns (n, dim)."""
        if not texts:
            return np.empty((0, self.cfg.dimension), dtype=np.float32)
        return self._maybe_normalize(self._embed(texts))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query, applying the configured query prefix. Returns (dim,)."""
        prefixed = self.cfg.query_prefix + text
        return self._maybe_normalize(self._embed([prefixed]))[0]

    def close(self) -> None:
        self._db.close()


if __name__ == "__main__":
    cfg = load_config()
    emb = Embedder(cfg.models.embedding)

    samples = [
        "Verizon total operating revenues were $138,191 million in 2025.",
        "The company's capital expenditures decreased year over year.",
    ]
    import time

    t0 = time.time()
    v1 = emb.embed_documents(samples)
    t1 = time.time()
    v2 = emb.embed_documents(samples)   # should be a cache hit -> much faster
    t2 = time.time()

    print(f"shape         : {v1.shape}  (expect (2, {cfg.models.embedding.dimension}))")
    print(f"L2 norm row 0 : {np.linalg.norm(v1[0]):.4f}  (expect ~1.0 if normalized)")
    print(f"1st call      : {t1 - t0:.3f}s  (API)")
    print(f"2nd call      : {t2 - t1:.3f}s  (cache)")
    q = emb.embed_query("What were Verizon's operating revenues?")
    print(f"query shape   : {q.shape}  (expect ({cfg.models.embedding.dimension},))")
    emb.close()
