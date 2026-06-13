"""
config.py — typed, validated loader for the pipeline's YAML configuration.

Turns the six files under config/ into one validated `AppConfig` object that the
rest of the package imports. Three responsibilities:

  1. LOAD + VALIDATE   Read every YAML into a typed Pydantic model. Unknown keys
                       are rejected (extra="forbid") so a typo fails loudly.
  2. STAMP THE RUN     Build a RunManifest — each file's `version` plus a content
                       hash — so any metric shift is attributable to a config change.
  3. ONE ENTRY POINT   Everyone calls load_config(); nobody else touches YAML.

Secrets are resolved LAZILY: config stores the NAME of the env var holding an API
key (e.g. NEBIUS_API_KEY), never the key. resolve_api_key() reads it only when a
model is actually called — so parsing, chunking, and tests run with no keys set.

Smoke test:  python -m finrag.config
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Load .env (if present) so API keys are available via os.environ. Optional dep;
# no-op if python-dotenv isn't installed or there's no .env file.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# --- base: strict + immutable ------------------------------------------------
class _Base(BaseModel):
    # extra="forbid": reject unknown YAML keys (catches typos).
    # frozen=True: config is read-only after load — no accidental mutation.
    # protected_namespaces=(): allow fields literally named `model`.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


# =============================================================================
# models.yaml
# =============================================================================
class GenerationConfig(_Base):
    provider: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_s: int = Field(default=60, gt=0)

    def api_key(self) -> str:
        """Resolve the generation API key from the environment (lazy)."""
        return resolve_api_key(self.api_key_env)


class EmbeddingConfig(_Base):
    provider: str                       # "local" or a hosted provider (e.g. "nebius")
    model: str
    dimension: int = Field(gt=0)        # MUST match the Pinecone index; discovered/confirmed at ingest
    normalize_embeddings: bool
    query_prefix: str = ""              # some models (bge) need a query prefix; "" if not
    batch_size: int = Field(default=32, gt=0)
    base_url: str | None = None         # required for hosted providers
    api_key_env: str | None = None      # required for hosted providers

    @model_validator(mode="after")
    def _hosted_needs_endpoint(self) -> "EmbeddingConfig":
        if self.provider != "local" and (not self.base_url or not self.api_key_env):
            raise ValueError(
                f"embedding provider {self.provider!r} is hosted and requires both "
                "base_url and api_key_env"
            )
        return self

    def api_key(self) -> str:
        """Resolve the embedding API key (hosted providers only)."""
        if not self.api_key_env:
            raise RuntimeError("local embedding model has no API key to resolve")
        return resolve_api_key(self.api_key_env)


class RerankerConfig(_Base):
    provider: str
    model: str


class HostedModelConfig(_Base):
    enabled: bool
    provider: str
    model: str
    api_key_env: str

    def api_key(self) -> str:
        return resolve_api_key(self.api_key_env)


class HostedCeilingConfig(_Base):
    openai: HostedModelConfig
    claude: HostedModelConfig


class ModelsConfig(_Base):
    version: int
    generation: GenerationConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    hosted_ceiling: HostedCeilingConfig


# =============================================================================
# chunking.yaml
# =============================================================================
class SharedChunkingConfig(_Base):
    table_open_marker: str
    table_close_marker: str
    max_chunk_chars: int = Field(gt=0)
    min_chunk_chars: int = Field(ge=0)
    min_alpha_ratio: float = Field(ge=0.0, le=1.0)


class FixedChunkingConfig(_Base):
    protect_tables: bool        # False for the naive baseline (shreds tables)
    splitter: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    separators: list[str]

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> "FixedChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"fixed.chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class SemanticChunkingConfig(_Base):
    protect_tables: bool        # True — each table is one atomic chunk
    breakpoint_percentile: int = Field(ge=1, le=99)
    breakpoint_sweep: list[int]
    buffer_size: int = Field(ge=1)
    min_chunk_chars: int = Field(ge=0)


class ChunkingConfig(_Base):
    version: int
    shared: SharedChunkingConfig
    fixed: FixedChunkingConfig
    semantic: SemanticChunkingConfig


# =============================================================================
# pipeline.yaml
# =============================================================================
class MetadataFilterConfig(_Base):
    enabled: bool
    fields: list[str]


class RetrievalConfig(_Base):
    dense_top_k: int = Field(gt=0)
    bm25_top_k: int = Field(gt=0)
    fusion: str
    rrf_k: int = Field(gt=0)
    fused_top_k: int = Field(gt=0)
    metadata_filter: MetadataFilterConfig


class RerankingConfig(_Base):
    enabled: bool
    input_top_n: int = Field(gt=0)
    output_top_n: int = Field(gt=0)


class ContextAssemblyConfig(_Base):
    reorder: str
    question_position: str
    context_open_tag: str
    context_close_tag: str
    include_chunk_ids: bool
    max_context_chars: int = Field(gt=0)


class VectorStoreConfig(_Base):
    index_name: str
    metric: str
    cloud: str
    region: str
    namespace_by_strategy: bool
    upsert_batch_size: int = Field(default=100, gt=0)


class PipelineConfig(_Base):
    version: int
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    context_assembly: ContextAssemblyConfig
    vectorstore: VectorStoreConfig


# =============================================================================
# thresholds.yaml
# =============================================================================
class RetrievalThresholds(_Base):
    k: int = Field(gt=0)
    min_recall_at_k: float = Field(ge=0.0, le=1.0)
    min_mrr: float = Field(ge=0.0, le=1.0)
    min_ndcg_at_k: float = Field(ge=0.0, le=1.0)


class GenerationThresholds(_Base):
    min_faithfulness: float = Field(ge=0.0, le=1.0)
    min_citation_coverage: float = Field(ge=0.0, le=1.0)


class GatingConfig(_Base):
    mode: Literal["hard", "warn"]
    tolerance: float = Field(ge=0.0, le=1.0)


class ThresholdsConfig(_Base):
    version: int
    gated_config: str
    retrieval: RetrievalThresholds
    generation: GenerationThresholds
    gating: GatingConfig


# =============================================================================
# prompts/*.yaml
# =============================================================================
class RagAnswerPrompt(_Base):
    version: int
    system: str
    user: str


class PromptStage(_Base):
    system: str
    user: str


class FaithfulnessPrompt(_Base):
    version: int
    extract: PromptStage
    verdict: PromptStage


class PromptsConfig(_Base):
    rag_answer: RagAnswerPrompt
    faithfulness: FaithfulnessPrompt


# =============================================================================
# run manifest — the version stamp
# =============================================================================
class RunManifest(_Base):
    run_id: str                 # timestamp + short signature hash, unique per run
    created_at: str             # ISO-8601 UTC
    versions: dict[str, int]    # {config_name: version}
    hashes: dict[str, str]      # {config_name: sha256[:12] of raw file bytes}

    def summary(self) -> str:
        vers = "  ".join(f"{k}=v{v}" for k, v in sorted(self.versions.items()))
        return f"run_id={self.run_id}\n  {vers}"


# =============================================================================
# top-level config + cross-field validation
# =============================================================================
class AppConfig(_Base):
    models: ModelsConfig
    chunking: ChunkingConfig
    pipeline: PipelineConfig
    thresholds: ThresholdsConfig
    prompts: PromptsConfig
    manifest: RunManifest

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "AppConfig":
        rr = self.pipeline.reranking
        rt = self.pipeline.retrieval
        # The reranker can only see candidates the fusion stage produced.
        if rr.input_top_n > rt.fused_top_k:
            raise ValueError(
                f"reranking.input_top_n ({rr.input_top_n}) cannot exceed "
                f"retrieval.fused_top_k ({rt.fused_top_k})"
            )
        # Can't keep more reranked results than were fed in.
        if rr.output_top_n > rr.input_top_n:
            raise ValueError(
                f"reranking.output_top_n ({rr.output_top_n}) cannot exceed "
                f"input_top_n ({rr.input_top_n})"
            )
        # NOTE: embedding.dimension must match the Pinecone index dimension. With a
        # hosted embedder the value is set by the model (confirmed at ingest), so we
        # no longer pin it to a literal here — vectorstore.py creates the index from
        # this value and asserts the live embedder agrees.
        return self


# =============================================================================
# loader
# =============================================================================
# config name -> path relative to the config directory
CONFIG_FILES: dict[str, str] = {
    "models": "models.yaml",
    "chunking": "chunking.yaml",
    "pipeline": "pipeline.yaml",
    "thresholds": "thresholds.yaml",
    "rag_answer": "prompts/rag_answer.yaml",
    "faithfulness": "prompts/faithfulness.yaml",
}


def resolve_api_key(env_var: str) -> str:
    """Read an API key from the environment, with a clear error if it's missing."""
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"Environment variable {env_var!r} is not set. Export it before "
            f"running this step, e.g.  export {env_var}=..."
        )
    return key


def _read_yaml(path: Path) -> tuple[dict, str]:
    """Parse a YAML file and return (data, sha256[:12] of its raw bytes)."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping (got {type(data).__name__})")
    digest = hashlib.sha256(raw).hexdigest()[:12]
    return data, digest


def _build_manifest(versions: dict[str, int], hashes: dict[str, str]) -> RunManifest:
    now = datetime.now(timezone.utc)
    signature = ",".join(f"{k}:v{v}:{hashes[k]}" for k, v in sorted(versions.items()))
    sig_hash = hashlib.sha256(signature.encode()).hexdigest()[:6]
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + sig_hash
    return RunManifest(
        run_id=run_id,
        created_at=now.isoformat(),
        versions=versions,
        hashes=hashes,
    )


def load_config(config_dir: str | Path = "config") -> AppConfig:
    """
    Load and validate every config file under `config_dir` into one AppConfig.

    Raises on any unknown key, wrong type, or cross-field inconsistency. Does NOT
    require API keys — those are resolved lazily at call time.
    """
    base = Path(config_dir)
    data: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for name, rel in CONFIG_FILES.items():
        data[name], hashes[name] = _read_yaml(base / rel)

    models = ModelsConfig(**data["models"])
    chunking = ChunkingConfig(**data["chunking"])
    pipeline = PipelineConfig(**data["pipeline"])
    thresholds = ThresholdsConfig(**data["thresholds"])
    prompts = PromptsConfig(
        rag_answer=RagAnswerPrompt(**data["rag_answer"]),
        faithfulness=FaithfulnessPrompt(**data["faithfulness"]),
    )

    versions = {
        "models": models.version,
        "chunking": chunking.version,
        "pipeline": pipeline.version,
        "thresholds": thresholds.version,
        "rag_answer": prompts.rag_answer.version,
        "faithfulness": prompts.faithfulness.version,
    }
    manifest = _build_manifest(versions, hashes)

    return AppConfig(
        models=models,
        chunking=chunking,
        pipeline=pipeline,
        thresholds=thresholds,
        prompts=prompts,
        manifest=manifest,
    )


if __name__ == "__main__":
    # Smoke test: load the real config and print the run manifest.
    cfg = load_config()
    print("Config loaded and validated OK.\n")
    print(cfg.manifest.summary())
    print(f"\n  generation : {cfg.models.generation.model} via {cfg.models.generation.provider}")
    print(f"  embedding  : {cfg.models.embedding.model} ({cfg.models.embedding.dimension}-dim)")
    print(f"  reranker   : {cfg.models.reranker.model}")
