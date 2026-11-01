"""Central configuration, read from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]          # backend/
FRONTEND_DIR = ROOT.parent / "frontend"
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int, lo: int = 1) -> int:
    try:
        return max(lo, int(_env(name, str(default))))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


def _gemini_keys() -> tuple[str, ...]:
    """GEMINI_API_KEY plus any backups in GEMINI_API_KEYS (comma-separated), first one first, no duplicates."""
    keys = [_env("GEMINI_API_KEY")] + _env("GEMINI_API_KEYS").split(",")
    return tuple(dict.fromkeys(k.strip() for k in keys if k.strip()))


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    # Source document
    pdf_path: Path
    source_name: str
    source_edition: str

    # Storage
    processed_dir: Path
    index_dir: Path
    cache_path: Path

    # LLMs
    gemini_api_key: str                 # the first key (embeddings, health checks)
    gemini_model: str
    gemini_light_model: str
    fallback_base_url: str
    fallback_api_key: str
    fallback_model: str
    llm_timeout: float          # seconds to wait for one LLM response
    llm_max_wait: float         # longest rate-limit back-off we sit through before falling back

    # Retrieval
    embedding_backend: str
    local_embedding_model: str
    gemini_embedding_model: str
    reranker_model: str
    bm25_top_k: int
    dense_top_k: int
    rerank_pool: int            # candidates reranked per sub-question (before xref / neighbours)
    rerank_budget: int          # max (query, passage) pairs scored per question, across sub-questions
    rerank_max_chars: int       # passage length fed to the cross-encoder
    context_k: int
    min_relevance: float

    # Runtime
    warmup: bool                # load models when the API starts, not on the first question
    gemini_api_keys: tuple[str, ...] = ()   # every Gemini key, GEMINI_API_KEY first then GEMINI_API_KEYS; failover order

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def sections_path(self) -> Path:
        return self.processed_dir / "sections.json"


def load_settings() -> Settings:
    return Settings(
        pdf_path=_path(_env("PDF_PATH", "data/raw/income-tax-act-1961-as-amended-by-finance-act-2026.pdf")),
        source_name=_env("SOURCE_NAME", "Income-tax Act, 1961 (as amended by Finance Act, 2026)"),
        source_edition=_env(
            "SOURCE_EDITION",
            "As amended by the Finance Act, 2026 - CBDT consolidated text, incometaxindia.gov.in",
        ),
        processed_dir=_path(_env("PROCESSED_DIR", "data/processed")),
        index_dir=_path(_env("INDEX_DIR", "data/index")),
        cache_path=_path(_env("LLM_CACHE_PATH", "data/cache/llm_cache.sqlite")),
        gemini_api_key=(_gemini_keys() or ("",))[0],
        gemini_api_keys=_gemini_keys(),
        gemini_model=_env("GEMINI_MODEL", "gemini-2.5-flash,gemini-3.6-flash,gemini-flash-latest"),
        gemini_light_model=_env("GEMINI_LIGHT_MODEL", "gemini-3.5-flash-lite,gemini-flash-lite-latest"),
        fallback_base_url=_env("FALLBACK_BASE_URL", "https://api.groq.com/openai/v1"),
        fallback_api_key=_env("FALLBACK_API_KEY"),
        fallback_model=_env("FALLBACK_MODEL", "llama-3.3-70b-versatile"),
        llm_timeout=_float("LLM_TIMEOUT", 90.0),
        llm_max_wait=_float("LLM_MAX_WAIT", 20.0),
        embedding_backend=_env("EMBEDDING_BACKEND", "local").lower(),
        local_embedding_model=_env("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        gemini_embedding_model=_env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        reranker_model=_env("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        bm25_top_k=_int("BM25_TOP_K", 30),
        dense_top_k=_int("DENSE_TOP_K", 30),
        rerank_pool=_int("RERANK_POOL", 16),
        rerank_budget=_int("RERANK_BUDGET", 72),
        rerank_max_chars=_int("RERANK_MAX_CHARS", 1100, lo=200),
        context_k=_int("CONTEXT_K", 10),
        min_relevance=_float("MIN_RELEVANCE", 0.10),
        warmup=_bool("WARMUP", True),
    )


settings = load_settings()
