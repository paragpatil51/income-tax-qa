"""Evidence-based answer confidence.

The score is computed from measurable signals, never from the LLM's opinion
of itself:

  support    (35%)  per statement: supported = 1, partial = 0.5, unsupported = 0.
                    Removed statements count 0, and a statement with a figure not
                    found in its cited text counts at most 0.5.
  citations  (25%)  share of citations passing all deterministic checks
                    (section retrieved + pinpoint present + quote matches)
  retrieval  (25%)  mean cross-encoder relevance of the provisions actually
                    cited by the kept statements
  coverage   (15%)  share of sub-questions answered by a kept, cited statement,
                    reduced when the answer reports parts as not covered

If the entailment judge could not run, its weight is redistributed and the
score is capped at 70 (medium), since no semantic check of the statements
took place.

Handling: below 50 the band is "low" and the pipeline adds a warning to read
the cited provisions directly. When no statement survives verification the
score is "n/a" and the answer is reported as insufficient.
"""

from __future__ import annotations

from taxqa.answering.schemas import Confidence, Point
from taxqa.retrieval.retriever import Hit

WEIGHTS = {"support": 0.35, "citations": 0.25, "retrieval": 0.25, "coverage": 0.15}
SUPPORT_VALUE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
UNJUDGED_CAP = 70
COVERAGE_RELEVANCE = 0.25   # reranker relevance at which a cited provision counts as addressing a sub-question


def _support_value(p: Point) -> float:
    if p.removed:
        return 0.0
    value = SUPPORT_VALUE.get(p.support, 0.5)   # judge ran but returned no verdict for it -> 0.5
    return min(value, 0.5) if p.unverified_figures else value


def compute_confidence(points: list[Point], hits_by_chunk: dict[str, Hit], n_sub_queries: int, *,
                       judge_ran: bool | None = None, not_covered: int = 0) -> Confidence:
    kept = [p for p in points if not p.removed]
    if not points or not kept:
        return Confidence(score=None, band="n/a", explanation=["No verified statements to score."])
    if judge_ran is None:
        judge_ran = any(p.support in SUPPORT_VALUE for p in points)

    comps: dict[str, float] = {}
    notes: list[str] = []

    if judge_ran:
        comps["support"] = sum(_support_value(p) for p in points) / len(points)
        notes.append(
            f"{sum(p.support == 'supported' and not p.removed for p in points)}/{len(points)} statements "
            f"fully supported by their cited text; {len(points) - len(kept)} removed."
        )
    else:
        notes.append(f"Entailment check unavailable: score uses mechanical signals only and is capped at {UNJUDGED_CAP}.")
    figures = sum(1 for p in kept if p.unverified_figures)
    if figures:
        notes.append(f"{figures} kept statement(s) contain figures not found in the cited text.")

    cites = [c for p in points for c in p.citations]
    comps["citations"] = sum(c.valid for c in cites) / len(cites) if cites else 0.0
    notes.append(f"{sum(c.valid for c in cites)}/{len(cites)} citations passed section, pinpoint and quote checks.")

    cited_chunks = {c.chunk_id for p in kept for c in p.citations if c.found_in_sources and c.chunk_id}
    rel = [hits_by_chunk[cid].score for cid in cited_chunks if cid in hits_by_chunk]
    comps["retrieval"] = sum(rel) / len(rel) if rel else 0.0
    notes.append(f"Mean reranker relevance of cited provisions: {comps['retrieval']:.2f}.")

    n_sub = max(1, n_sub_queries)
    # A sub-question is covered when a cited provision is relevant to it (one provision,
    # e.g. section 24, can answer several sub-questions).
    covered = set()
    for cid in cited_chunks:
        h = hits_by_chunk.get(cid)
        if h is None:
            continue
        covered.add(h.sub_query)
        covered.update(q for q, s in getattr(h, "per_query", {}).items() if s >= COVERAGE_RELEVANCE)
    coverage = min(1.0, len(covered) / n_sub)
    if not_covered:
        coverage = min(coverage, 1 - not_covered / (not_covered + max(1, len(covered))))
    comps["coverage"] = coverage
    notes.append(f"{len(covered)}/{n_sub} sub-questions backed by a cited provision"
                 + (f"; {not_covered} part(s) reported as not covered." if not_covered else "."))

    total_w = sum(WEIGHTS[k] for k in comps)
    pct = round(100 * sum(WEIGHTS[k] * v for k, v in comps.items()) / total_w)
    if not judge_ran:
        pct = min(pct, UNJUDGED_CAP)
    band = "high" if pct >= 75 else "medium" if pct >= 50 else "low"
    return Confidence(score=pct, band=band, components={k: round(v, 3) for k, v in comps.items()}, explanation=notes)
