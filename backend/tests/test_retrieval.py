"""Retrieval behaviour with fake embedder / reranker (no model downloads, no network)."""

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from taxqa.core.config import settings
from taxqa.core.models import Chunk
from taxqa.retrieval.hybrid_index import HybridIndex, tokenize
from taxqa.retrieval.query_planner import explicit_sections, plan_query
from taxqa.retrieval.retriever import Retriever

DIM = 256


class FakeEmbedder:
    backend, name = "fake", "hashing-bow"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)

    def warmup(self):
        pass


class FakeReranker:
    """Relevance = share of query tokens found in the passage; records batch sizes."""

    def __init__(self, available: bool = True):
        self.available = available
        self.batches: list[int] = []

    def scores(self, query, docs):
        if not self.available:
            return None
        self.batches.append(len(docs))
        q = set(tokenize(query))
        return [len(q & set(tokenize(d))) / max(1, len(q)) for d in docs]

    def warmup(self):
        pass


def _c(cid, section, path, text, title="", context="", refs=None):
    return Chunk(id=cid, section=section, section_title=title, chapter="IV", chapter_title="", path=path,
                 page_start=10, page_end=10, text=text, context=context, refs=refs or [])


CHUNKS = [
    _c("s2-1", "2", "2(14)", '(14) "capital asset" means property of any kind held by an assessee', "Definitions"),
    _c("s10-1", "10", "10(1)", "(1) agricultural income;", "Incomes not included in total income"),
    _c("s10-2", "10", "10(2)", "(2) any sum received by a member of a Hindu undivided family", "Incomes not included in total income"),
    _c("s10-3", "10", "10(5)", "(5) leave travel concession", "Incomes not included in total income"),
    _c("s10-4", "10", "10(10)", "(10) any death-cum-retirement gratuity", "Incomes not included in total income"),
    _c("s10-5", "10", "10(10D)", "(10D) any sum received under a life insurance policy, including bonus", "Incomes not included in total income"),
    _c("s23-1", "23", "23(1)", "(1) For the purposes of section 22, the annual value of any property shall be deemed to be the sum "
       "for which the property might reasonably be expected to let from year to year", "Annual value how determined", refs=["22"]),
    _c("s24-1", "24", "24(a)-(b)", "(a) a sum equal to thirty per cent of the annual value; (b) where the property has been "
       "acquired or constructed with borrowed capital, the amount of any interest payable on such capital:",
       "Deductions from income from house property",
       context="Income chargeable under the head Income from house property shall be computed after making the following deductions",
       refs=["23"]),
    _c("s24-2", "24", "24(b)", "Provided that the amount of deduction for a self-occupied property shall not exceed two lakh rupees",
       "Deductions from income from house property"),
    _c("s80C-1", "80C", "80C(1)", "(1) there shall be deducted the amount paid as life insurance premium or contribution to a "
       "provident fund, as does not exceed one lakh fifty thousand rupees", "Deduction in respect of life insurance premia"),
    _c("s80-IA-1", "80-IA", "80-IA(1)", "(1) profits and gains derived by an undertaking from an eligible business of developing "
       "infrastructure facility", "Deductions in respect of profits and gains from infrastructure undertakings"),
]


@pytest.fixture
def env(tmp_path):
    s = dataclasses.replace(settings, processed_dir=tmp_path, index_dir=tmp_path, rerank_budget=72, context_k=6,
                            min_relevance=0.10, bm25_top_k=10, dense_top_k=10, rerank_pool=8)
    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in CHUNKS:
            f.write(json.dumps(c.to_dict()) + "\n")
    emb = FakeEmbedder()
    np.save(tmp_path / "embeddings.npy", emb.embed_documents([c.index_text() for c in CHUNKS]))
    (tmp_path / "meta.json").write_text(json.dumps(
        {"backend": emb.backend, "model": emb.name, "dim": DIM, "count": len(CHUNKS), "chunk_ids": [c.id for c in CHUNKS]}))
    index = HybridIndex(s, emb)
    return s, index


def test_tokenizer_handles_section_forms():
    toks = tokenize("Deduction u/s 80-IA and exemption under 10(10D) and 115BAC")
    assert "80ia" in toks and "10_10d" in toks and "10d" in toks and "115bac" in toks
    assert "u" not in toks and "s" not in toks


def test_explicit_section_parsing():
    assert explicit_sections("sections 23 and 24 apply") == ["23", "24"]
    assert explicit_sections("section 24 and 25") == ["24", "25"]
    assert explicit_sections("What does section 80-IA say? Compare 80IA and u/s 80C") == ["80IA", "80C"]
    assert explicit_sections("I filed on the 10th, my 1st return in 2024") == []


def test_hybrid_search_with_xref_and_neighbour_expansion(env):
    s, index = env
    r = Retriever(index, s, FakeReranker())
    plan = plan_query("Can I deduct thirty per cent of the annual value and interest on borrowed capital?", None)
    hits = r.search(plan)
    assert hits[0].chunk.section == "24"
    by_id = {h.chunk.id: h for h in hits}
    assert "s23-1" in by_id                          # pulled in via 24's cross-reference
    assert "s24-2" in by_id                          # proviso in the neighbouring chunk
    assert {"xref"} <= by_id["s23-1"].via or {"bm25", "dense"} & by_id["s23-1"].via
    ctx = r.select_context(hits, 1, plan.explicit_sections)
    assert ctx[0].chunk.id == "s24-1" and len(ctx) <= s.context_k


def test_explicit_hyphenated_section_and_pinpoint(env):
    s, index = env
    r = Retriever(index, s, FakeReranker())
    assert index.has_section("80-IA") and index.has_section("80IA")
    plan = plan_query("What does section 80-IA provide?", None)
    ctx = r.select_context(r.search(plan), 1, plan.explicit_sections)
    assert any(h.chunk.section == "80-IA" for h in ctx)

    plan = plan_query("Explain section 10(10D)", None)
    picked = r._explicit_chunks(plan, [])
    assert picked[0].path == "10(10D)"
    assert len(picked) <= 4


def test_rerank_budget_caps_pairs_per_sub_question(env):
    s, index = env
    s = dataclasses.replace(s, rerank_budget=20)
    rr = FakeReranker()
    r = Retriever(index, s, rr)
    from taxqa.retrieval.query_planner import QueryPlan, SubQuery
    plan = QueryPlan("q", [SubQuery(q, q) for q in ["annual value", "borrowed capital interest",
                                                    "life insurance premium", "agricultural income"]])
    r.search(plan)
    assert rr.batches and max(rr.batches) <= max(8, 20 // 4)


def test_reranker_unavailable_falls_back_to_fused_scores(env):
    s, index = env
    r = Retriever(index, s, FakeReranker(available=False))
    hits = r.search(plan_query("life insurance premium provident fund deduction", None))
    assert hits[0].chunk.section == "80C"
    assert all(0.0 <= h.score <= 0.5 for h in hits)


def test_planner_strips_sections_the_user_did_not_write():
    class FakeLLM:
        configured, last_model = True, "fake:model"

        def __init__(self, reply):
            self.reply = reply

        def json(self, system, user, light=False, temperature=0.0):
            return self.reply

    reply = {"in_scope": True, "sub_questions": [
        {"question": "Is interest on a housing loan deductible under section 24(b)?",
         "statutory_query": "interest payable on capital borrowed, section 24"},
        {"question": "Is interest on a housing loan deductible under section 24(b)?", "statutory_query": "dup"},
    ]}
    plan = plan_query("Can I deduct my home loan interest?", FakeLLM(reply))
    assert len(plan.sub_queries) == 1
    assert "24" not in plan.sub_queries[0].question and "24" not in plan.sub_queries[0].statutory_query
    plan = plan_query("Does section 24 allow home loan interest?", FakeLLM(reply))
    assert "section 24" in plan.sub_queries[0].statutory_query
    plan = plan_query("GST on hotels?", FakeLLM({"in_scope": "false", "sub_questions": [{"question": "GST rate"}]}))
    assert plan.in_scope is False and plan.planner == "fake:model"
