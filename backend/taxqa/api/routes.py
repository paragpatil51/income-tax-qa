"""REST endpoints under /api."""

from __future__ import annotations

import json
import logging
import threading

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from taxqa.answering.schemas import Answer
from taxqa.core.config import settings
from taxqa.retrieval.section_refs import section_id
from taxqa.services.qa_pipeline import QAPipeline

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class PipelineUnavailable(RuntimeError):
    """The index has not been built (or is out of date); surfaced as HTTP 503."""


_lock = threading.Lock()
_pipeline: QAPipeline | None = None
_sections_cache: tuple[float, dict[str, dict]] | None = None


def get_pipeline() -> QAPipeline:
    """Build the pipeline once, thread-safely. Failures are not cached, so running
    ingest.py while the server is up makes it ready without a restart."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is None:
            try:
                _pipeline = QAPipeline(settings)
            except FileNotFoundError as e:
                raise PipelineUnavailable(
                    f"Index not built ({getattr(e, 'filename', e)} missing). Run: python scripts/ingest.py"
                ) from e
            except RuntimeError as e:
                raise PipelineUnavailable(str(e)) from e
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(max_length=2000, examples=["Can I deduct interest on a housing loan for a let-out house?"])

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 5:
            raise ValueError("question must be at least 5 characters")
        return v


@router.get("/health")
def health() -> dict:
    try:
        p = get_pipeline()
    except PipelineUnavailable as e:
        return {"ready": False, "detail": str(e), "llm_configured": bool(settings.gemini_api_key or settings.fallback_api_key)}
    except Exception as e:  # health must never fail
        log.exception("health check failed")
        return {"ready": False, "detail": f"Backend failed to start: {e}"}
    return {
        "ready": True,
        "source_document": settings.source_name,
        "edition": settings.source_edition,
        "chunks": len(p.index.chunks),
        "sections": len({c.section for c in p.index.chunks if c.kind == "section"}),
        "schedules": len({c.section for c in p.index.chunks if c.kind == "schedule"}),
        "index_built_at": p.index.meta.get("built_at"),
        "embeddings": f"{p.embedder.backend}:{p.embedder.name}",
        "reranker": settings.reranker_model if p.reranker.available else "unavailable",
        "llm_configured": p.llm.configured,
        "llm_model": settings.gemini_model if settings.gemini_api_key else settings.fallback_model,
        # real API calls per model (UTC day); cached answers are free
        "llm_usage_today": p.llm.cache.usage_today() if hasattr(p.llm, "cache") else {},
    }


@router.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    return get_pipeline().ask(req.question)


@router.get("/search")
def search(q: str = Query(min_length=3, max_length=2000), llm: bool = True) -> dict:
    """Retrieval only (no answer generation). `llm=false` also skips the LLM query planner."""
    r = get_pipeline().retrieve(q, use_llm=llm)
    return {
        "sub_queries": [vars(s) for s in r.plan.sub_queries],
        "explicit_sections": r.plan.explicit_sections,
        "in_scope": r.plan.in_scope,
        "context": [h.to_dict() for h in r.context],
        "top_hits": [h.to_dict() for h in r.hits[:25]],
        "timings_ms": r.timings_ms,
    }


def _sections() -> dict[str, dict]:
    """sections.json keyed by canonical section id; reloaded when the file changes."""
    global _sections_cache
    path = settings.sections_path
    if not path.exists():
        raise PipelineUnavailable("Index not built. Run: python scripts/ingest.py")
    mtime = path.stat().st_mtime
    if _sections_cache is None or _sections_cache[0] != mtime:
        data = json.loads(path.read_text(encoding="utf-8"))
        _sections_cache = (mtime, {section_id(s["num"]): s for s in data})
    return _sections_cache[1]


@router.get("/section/{num}")
def section(num: str) -> dict:
    """Full text of one section, for the 'view full section' link in the UI."""
    s = _sections().get(section_id(num))
    if s is None:
        raise HTTPException(404, f"Section {num} not found in the ingested Act")
    return s
