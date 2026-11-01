"""Hybrid retrieval pipeline.

For every sub-question:
  1. BM25 and dense search, each with the user phrasing and the statutory phrasing
  2. Reciprocal Rank Fusion of the ranked lists
  3. + chunks of sections the user named explicitly (preferring the named sub-division)
  4. + chunks of sections cross-referenced by the best hits (e.g. 24 -> 23)
  5. + neighbouring chunks of the best hits (a proviso may sit in the next chunk)
  6. cross-encoder reranking against the sub-question, within a per-question pair budget

Reranker scores are calibrated probabilities (0-1) and are reused downstream
for abstention and for the confidence score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from taxqa.core.config import Settings
from taxqa.core.models import Chunk
from taxqa.retrieval.hybrid_index import HybridIndex
from taxqa.retrieval.query_planner import QueryPlan
from taxqa.retrieval.reranker import Reranker
from taxqa.retrieval.section_refs import section_id

RRF_K = 60
EXPAND_FROM_TOP = 3        # hits whose cross-references / neighbours are added
REFS_PER_HIT = 3
CHUNKS_PER_REF = 2
XREF_CONTEXT_FROM = 3       # top context passages whose referenced sections are always included
XREF_CONTEXT_MAX = 2        # at most this many sections added that way
CONDITIONS_FROM = 6         # top context passages whose section's conditions sub-section is kept with them
CONDITIONS_MAX = 3
CONDITIONS_RE = re.compile(
    r"subject to the following conditions|shall not be allowed unless|shall be allowed only if|"
    r"shall apply only if|following conditions are fulfilled",
    re.I,
)
EXPLICIT_PER_SECTION = 4
CONTEXT_PREVIEW_CHARS = 240


@dataclass
class Hit:
    chunk: Chunk
    score: float                           # reranker relevance probability, 0-1
    rrf: float = 0.0
    via: set[str] = field(default_factory=set)   # bm25 / dense / explicit / xref / neighbour
    sub_query: int = 0                     # which sub-question it best serves
    per_query: dict[int, float] = field(default_factory=dict)   # relevance to every sub-question it was scored for

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.id,
            "label": self.chunk.label,
            "score": round(self.score, 4),
            "via": sorted(self.via),
            "sub_query": self.sub_query,
            "per_query": {q: round(s, 4) for q, s in self.per_query.items()},
        }


def rerank_text(c: Chunk, max_chars: int) -> str:
    """Header + a short preview of the section preamble + the unit's own text.

    The preamble can be ~120 words; passing it whole would push the statutory
    text itself past the truncation point."""
    ctx = c.context
    if len(ctx) > CONTEXT_PREVIEW_CHARS:
        ctx = ctx[:CONTEXT_PREVIEW_CHARS].rsplit(" ", 1)[0] + " ..."
    return "\n".join(p for p in (c.header, ctx, c.text) if p)[:max_chars]


class Retriever:
    def __init__(self, index: HybridIndex, settings: Settings, reranker: Reranker):
        self.index = index
        self.s = settings
        self.reranker = reranker

    def _fused(self, texts: list[str]) -> list[tuple[str, float, set[str]]]:
        fused: dict[str, tuple[float, set[str]]] = {}
        for text in dict.fromkeys(t for t in texts if t):
            for name, results in (
                ("bm25", self.index.bm25_search(text, self.s.bm25_top_k)),
                ("dense", self.index.dense_search(text, self.s.dense_top_k)),
            ):
                for rank, (chunk, _) in enumerate(results):
                    score, via = fused.get(chunk.id, (0.0, set()))
                    fused[chunk.id] = (score + 1.0 / (RRF_K + rank + 1), via | {name})
        return sorted(((cid, s, v) for cid, (s, v) in fused.items()), key=lambda x: -x[1])

    def _explicit_chunks(self, plan: QueryPlan, fused_order: list[str]) -> list[Chunk]:
        """Chunks of sections the user named: the named sub-division first, then the best-ranked ones."""
        pinpoints = {f"({p.lower()})" for p in re.findall(r"\(\s*([0-9A-Za-z]{1,6})\s*\)", plan.original)}
        rank = {cid: i for i, cid in enumerate(fused_order)}
        out: list[Chunk] = []
        for num in plan.explicit_sections:
            chunks = self.index.section_chunks(num)
            if len(chunks) <= EXPLICIT_PER_SECTION:
                out.extend(chunks)
                continue
            pinned = [c for c in chunks if pinpoints and any(p in c.path.lower() for p in pinpoints)]
            ranked = sorted(chunks, key=lambda c: rank.get(c.id, len(rank) + self.index.position[c.id]))
            picked = {c.id: c for c in pinned[:2] + ranked}   # dedupe, keeping priority order
            out.extend(list(picked.values())[:EXPLICIT_PER_SECTION])
        return out

    def _pool(self, plan: QueryPlan, qi: int, cap: int) -> list[Hit]:
        sq = plan.sub_queries[qi]
        # The user's own wording is always one of the queries, so retrieval never depends on the planner alone.
        fused = self._fused([sq.question, sq.statutory_query, plan.original])
        pool: dict[str, Hit] = {}

        def add(c: Chunk, via: str, rrf: float = 0.0) -> None:
            if len(pool) < cap or c.id in pool:
                h = pool.setdefault(c.id, Hit(c, 0.0, sub_query=qi))
                h.via.add(via)
                h.rrf = max(h.rrf, rrf)

        for c in self._explicit_chunks(plan, [cid for cid, _, _ in fused]):
            add(c, "explicit")
        for cid, score, via in fused[: self.s.rerank_pool]:
            for v in via:
                add(self.index.by_id[cid], v, score)
        top = [self.index.by_id[cid] for cid, _, _ in fused[:EXPAND_FROM_TOP]]
        for c in top:
            for ref in c.refs[:REFS_PER_HIT]:
                for rc in self.index.section_chunks(ref)[:CHUNKS_PER_REF]:
                    add(rc, "xref")
        for c in top:
            for nc in self.index.neighbours(c):
                add(nc, "neighbour")
        return list(pool.values())

    def search(self, plan: QueryPlan) -> list[Hit]:
        n = max(1, len(plan.sub_queries))
        cap = max(8, self.s.rerank_budget // n)
        best: dict[str, Hit] = {}
        scored: dict[tuple[str, str], float] = {}   # (rerank query, chunk id) -> score, within this request
        per_query: dict[str, dict[int, float]] = {}

        for qi, sq in enumerate(plan.sub_queries):
            hits = self._pool(plan, qi, cap)
            query = sq.rerank_text
            todo = [h for h in hits if (query, h.chunk.id) not in scored]
            scores = self.reranker.scores(query, [rerank_text(h.chunk, self.s.rerank_max_chars) for h in todo])
            if scores is None:
                top = max((h.rrf for h in hits), default=0.0) or 1.0
                scores = [h.rrf / top * 0.5 for h in todo]   # crude; flagged via reranker.available
            for h, sc in zip(todo, scores):
                scored[(query, h.chunk.id)] = sc
            for h in hits:
                h.score = scored[(query, h.chunk.id)]
                per_query.setdefault(h.chunk.id, {})[qi] = h.score
                prev = best.get(h.chunk.id)
                if prev is None or h.score > prev.score:
                    if prev:
                        h.via |= prev.via
                    best[h.chunk.id] = h
                else:
                    prev.via |= h.via

        for cid, h in best.items():
            h.per_query = per_query[cid]
        return sorted(best.values(), key=lambda h: -h.score)

    def select_context(self, hits: list[Hit], n_sub: int, explicit: list[str] | None = None) -> list[Hit]:
        """Best passage of each named section, then top hits per sub-question, then best overall."""
        k = self.s.context_k
        per_q = max(2, k // max(1, n_sub))
        floor = self.s.min_relevance * 0.5
        chosen: dict[str, Hit] = {}
        for num in (explicit or [])[:3]:
            sid = section_id(num)
            h = next((h for h in hits if section_id(h.chunk.section) == sid), None)
            if h:
                chosen.setdefault(h.chunk.id, h)
        for qi in range(n_sub):
            for h in [h for h in hits if h.sub_query == qi and h.score >= floor][:per_q]:
                chosen.setdefault(h.chunk.id, h)
        for h in hits:
            if len(chosen) >= k:
                break
            if h.score >= floor:
                chosen.setdefault(h.chunk.id, h)
        if not chosen:
            chosen = {h.chunk.id: h for h in hits[:3]}   # shown as "closest provisions" when abstaining
            return sorted(chosen.values(), key=lambda h: -h.score)[:k]
        selected = sorted(chosen.values(), key=lambda h: -h.score)[:k]
        selected += self._cross_references(selected, hits)
        return selected + self._conditions(selected, hits)

    def _conditions(self, selected: list[Hit], hits: list[Hit]) -> list[Hit]:
        """The sub-section that states a provision's conditions ("(3) The deduction under sub-section (1)
        shall be subject to the following conditions") is often a separate chunk that scores low against
        the question. Without it an answer can describe a deduction but not who qualifies for it."""
        have = {h.chunk.id for h in selected}
        by_id = {h.chunk.id: h for h in hits}
        extra: list[Hit] = []
        done: set[str] = set()
        for h in selected[:CONDITIONS_FROM]:
            sid = section_id(h.chunk.section)
            if sid in done or h.chunk.kind != "section":
                continue
            done.add(sid)
            for c in self.index.section_chunks(h.chunk.section):
                if c.id in have or not CONDITIONS_RE.search(c.text):
                    continue
                hit = by_id.get(c.id) or Hit(c, 0.0, sub_query=h.sub_query)
                hit.via.add("conditions")
                extra.append(hit)
                have.add(c.id)
                break   # one conditions chunk per section
            if len(extra) >= CONDITIONS_MAX:
                break
        return extra

    def _cross_references(self, selected: list[Hit], hits: list[Hit]) -> list[Hit]:
        """Sections the best passages themselves refer to ("... section 23"), added whatever their
        reranker score: a provision often only makes sense with the one it points to (24 -> 23 for
        "annual value"), and a lay question rarely shares words with that second provision."""
        have = {section_id(h.chunk.section) for h in selected}
        by_section: dict[str, Hit] = {}
        for h in hits:                       # hits are sorted best first
            by_section.setdefault(section_id(h.chunk.section), h)
        extra: list[Hit] = []
        for h in selected[:XREF_CONTEXT_FROM]:
            for ref in h.chunk.refs[:REFS_PER_HIT]:
                sid = section_id(ref)
                if sid in have or len(extra) >= XREF_CONTEXT_MAX:
                    continue
                hit = by_section.get(sid)
                if hit is None:
                    chunks = self.index.section_chunks(ref)
                    if not chunks or chunks[0].omitted:
                        continue
                    hit = Hit(chunks[0], 0.0, via={"xref"}, sub_query=h.sub_query)
                hit.via.add("xref")
                extra.append(hit)
                have.add(sid)
        return extra
