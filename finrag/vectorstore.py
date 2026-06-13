"""
vectorstore.py — Pinecone wrapper for the fixed/semantic vector index.

One serverless index (dimension from models.yaml, cosine). The fixed vs semantic
A/B is isolated by NAMESPACE; company/section/etc. ride as metadata so retrieval
can filter to a company within either namespace. The chunk's text is stored in
metadata (under the Pinecone 40 KB cap) so retrieval returns text without a second
lookup.

Key from PINECONE_API_KEY (never hardcoded). Code creates the index at the right
dimension — never create it by hand.
"""

from __future__ import annotations

import time

import numpy as np
from pinecone import Pinecone, ServerlessSpec

from finrag.config import AppConfig, resolve_api_key
from finrag.chunking import Chunk


class VectorStore:
    def __init__(self, cfg: AppConfig, api_key: str | None = None) -> None:
        self.cfg = cfg
        self.vs = cfg.pipeline.vectorstore
        self.dimension = cfg.models.embedding.dimension
        self._pc = Pinecone(api_key=api_key or resolve_api_key("PINECONE_API_KEY"))
        self._index = None

    # --- index lifecycle -----------------------------------------------------
    def ensure_index(self):
        """Create the index if missing (at the embedding dimension), wait until ready."""
        names = [ix["name"] for ix in self._pc.list_indexes()]
        if self.vs.index_name not in names:
            self._pc.create_index(
                name=self.vs.index_name,
                dimension=self.dimension,
                metric=self.vs.metric,
                spec=ServerlessSpec(cloud=self.vs.cloud, region=self.vs.region),
            )
            while not self._pc.describe_index(self.vs.index_name).status["ready"]:
                time.sleep(1)
        self._index = self._pc.Index(self.vs.index_name)
        return self._index

    @property
    def index(self):
        return self._index if self._index is not None else self.ensure_index()

    def namespace_for(self, strategy: str) -> str:
        return strategy if self.vs.namespace_by_strategy else ""

    # --- writes --------------------------------------------------------------
    def upsert_chunks(self, chunks: list[Chunk], vectors: np.ndarray, strategy: str) -> int:
        """Upsert (id, vector, metadata+text) for a strategy's namespace, in batches."""
        ns = self.namespace_for(strategy)
        items = []
        for ch, vec in zip(chunks, vectors):
            md = ch.metadata()
            md["text"] = ch.text                       # store text for retrieval
            items.append({"id": ch.id, "values": vec.tolist(), "metadata": md})
        bs = self.vs.upsert_batch_size
        for i in range(0, len(items), bs):
            self.index.upsert(vectors=items[i : i + bs], namespace=ns)
        return len(items)

    def delete_namespace(self, strategy: str) -> None:
        """Clear a namespace for a clean rebuild (no-op if it doesn't exist yet)."""
        ns = self.namespace_for(strategy)
        try:
            self.index.delete(delete_all=True, namespace=ns)
        except Exception:                              # namespace absent on first ingest
            pass

    def delete_ids(self, ids: list[str], strategy: str) -> int:
        """Delete specific vector ids from a strategy's namespace, in batches.
        Serverless Pinecone has no delete-by-metadata, so removing one company means
        deleting its (deterministic) chunk ids — sourced from the BM25 corpus file."""
        ns = self.namespace_for(strategy)
        bs = self.vs.upsert_batch_size
        for i in range(0, len(ids), bs):
            self.index.delete(ids=ids[i : i + bs], namespace=ns)
        return len(ids)

    # --- reads ---------------------------------------------------------------
    def query(self, vector: np.ndarray, top_k: int, strategy: str, metadata_filter: dict | None = None):
        """Cosine top-k within a strategy's namespace, optional metadata filter."""
        res = self.index.query(
            vector=vector.tolist(),
            top_k=top_k,
            namespace=self.namespace_for(strategy),
            include_metadata=True,
            filter=metadata_filter or None,
        )
        return res["matches"]

    def stats(self) -> dict:
        return self.index.describe_index_stats().to_dict()
