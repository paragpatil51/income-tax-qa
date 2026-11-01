"""End-to-end question answering: plan -> retrieve -> generate -> verify -> score."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from taxqa.answering import generator
from taxqa.answering.confidence import compute_confidence
from taxqa.answering.schemas import Answer, SourceView
from taxqa.answering.verifier import apply_removals, finalize_text, judge_support, verify_citations
from taxqa.core.config import Settings, settings as default_settings
from taxqa.core.llm_client import LLMClient, LLMError
from taxqa.retrieval.embeddings import Embedder
from taxqa.retrieval.hybrid_index import HybridIndex
from taxqa.retrieval.query_planner import QueryPlan, plan_query
from taxqa.retrieval.reranker import Reranker
from taxqa.retrieval.retriever import Hit, Retriever

log = logging.getLogger(__name__)


@dataclass
class Retrieval:
    plan: QueryPlan
    hits: list[Hit]          # every reranked candidate, best first
    context: list[Hit]       # the excerpts given to the LLM
    timings_ms: dict[str, int] = field(default_factory=dict)


class QAPipeline:
    def __init__(self, settings: Settings = default_settings, *, embedder: Embedder | None = None,
                 index: HybridIndex | None = None, reranker: Reranker | None = None,
                 llm: LLMClient | None = None):
        self.s = settings
        self.embedder = embedder or Embedder(settings)
        self.index = index or HybridIndex(settings, self.embedder)
        self.reranker = reranker or Reranker(settings.reranker_model)
        self.retriever = Retriever(self.index, settings, self.reranker)
        self.llm = llm or LLMClient(settings)

    def warmup(self) -> None:
        """Load the ONNX models now so the first question does not pay for it."""
        t0 = time.perf_counter()
        self.embedder.warmup()
        self.reranker.warmup()
        log.info("Models warmed up in %.1fs", time.perf_counter() - t0)

    def _sources(self, hits: list[Hit]) -> list[SourceView]:
        return [
            SourceView(
                source_id=sid,
                chunk_id=h.chunk.id,
                label=h.chunk.label,
                section_title=h.chunk.section_title,
                chapter=h.chunk.chapter,
                pages=h.chunk.pages,
                page_start=h.chunk.page_start,
                page_end=h.chunk.page_end,
                relevance=round(h.score, 3),
                via=sorted(h.via),
                text=h.chunk.text,
                context=h.chunk.context,
            )
            for sid, h in generator.source_ids(hits).items()
        ]

    def retrieve(self, question: str, *, use_llm: bool = True) -> Retrieval:
        """Plan and retrieve. `use_llm=False` skips the LLM planner (zero API calls)."""
        t0 = time.perf_counter()
        plan = plan_query(question, self.llm if use_llm else None)
        t1 = time.perf_counter()
        hits = self.retriever.search(plan)
        context = self.retriever.select_context(hits, len(plan.sub_queries), plan.explicit_sections)
        t2 = time.perf_counter()
        return Retrieval(plan, hits, context, {"planning": int((t1 - t0) * 1000), "retrieval": int((t2 - t1) * 1000)})

    def ask(self, question: str) -> Answer:
        question = " ".join(question.split())
        t0 = time.perf_counter()
        ans = Answer(question=question, status="error", source_document=self.s.source_name, edition=self.s.source_edition)

        r = self.retrieve(question)
        plan, context = r.plan, r.context
        timings = dict(r.timings_ms)
        ans.sub_questions = [sq.question for sq in plan.sub_queries]
        ans.sources = self._sources(context)
        ans.models = {
            "planner": plan.planner,
            "embeddings": f"{self.embedder.backend}:{self.embedder.name}",
            "reranker": self.s.reranker_model if self.reranker.available else "unavailable (RRF fallback)",
        }
        if not self.reranker.available:
            ans.warnings.append("Reranker unavailable - relevance scores are approximate.")
        missing = [num for num in plan.explicit_sections if not self.index.has_section(num)]
        if missing:
            ans.warnings.append(
                f"Section(s) {', '.join(missing)} named in the question were not found in the ingested Act "
                "(they may not exist, may be omitted, or may be numbered differently)."
            )

        best = context[0].score if context else 0.0
        if not plan.in_scope and best < 0.5 and not (set(plan.explicit_sections) - set(missing)):
            ans.status = "out_of_scope"
            ans.summary = "This question does not appear to concern the Income-tax Act, 1961, so it cannot be answered from this source."
            return self._finish(ans, t0, timings)
        if best < self.s.min_relevance:
            ans.status = "insufficient"
            ans.summary = (
                "No provision of the Act retrieved for this question is relevant enough to support an answer. "
                "The closest provisions found are listed below for reference, but they should not be read as answering the question."
            )
            return self._finish(ans, t0, timings)
        if not self.llm.configured:
            ans.status = "error"
            ans.summary = "No LLM API key configured. Showing the retrieved provisions only."
            return self._finish(ans, t0, timings)

        t1 = time.perf_counter()
        try:
            draft = generator.generate(
                question, ans.sub_questions, context, self.llm, self.s.source_name, self.s.source_edition
            )
        except LLMError as e:
            ans.summary = "The language model could not be reached. The retrieved provisions are shown below."
            ans.warnings.append(f"LLM error: {e}")
            return self._finish(ans, t0, timings)
        ans.models["generator"] = self.llm.last_model
        timings["generation"] = int((time.perf_counter() - t1) * 1000)

        ans.not_covered = [str(x) for x in draft.get("not_covered") or [] if str(x).strip()]
        ans.points = generator.parse_points(draft)

        t2 = time.perf_counter()
        sources = generator.source_ids(context)
        ans.warnings += verify_citations(ans.points, sources, self.s.source_name)
        by_chunk = {h.chunk.id: h for h in context}
        judged = judge_support(ans.points, by_chunk, self.llm, question=question)
        if not judged:
            ans.warnings.append("Claim-level entailment check could not run (LLM unavailable); citations were checked mechanically only.")
        else:
            ans.models["verifier"] = self.llm.last_model
        apply_removals(ans.points, judge_ran=judged)
        # Summary and caveats may only repeat verified content (removed claims cannot leak back in).
        final = finalize_text(draft, ans.points, sources)
        ans.summary, ans.summary_removed, ans.caveats = final.summary, final.summary_removed, final.caveats
        ans.warnings += final.warnings
        timings["verification"] = int((time.perf_counter() - t2) * 1000)

        kept = [p for p in ans.points if not p.removed]
        model_status = str(draft.get("status", "answered"))
        if not kept:
            ans.status = "insufficient"
            if ans.points:
                ans.summary = (
                    "A draft answer was generated, but none of its statements could be verified against the "
                    "retrieved provisions, so it has been withheld."
                )
        else:
            ans.status = "partial" if model_status == "partial" or ans.not_covered or len(kept) < len(ans.points) else "answered"
        ans.confidence = compute_confidence(ans.points, by_chunk, len(plan.sub_queries),
                                            judge_ran=judged, not_covered=len(ans.not_covered))
        if ans.confidence.band == "low":
            ans.warnings.append("Low confidence: read the cited provisions directly before relying on this answer.")
        return self._finish(ans, t0, timings)

    @staticmethod
    def _finish(ans: Answer, t0: float, timings: dict[str, int]) -> Answer:
        timings["total"] = int((time.perf_counter() - t0) * 1000)
        ans.timings_ms = timings
        return ans
