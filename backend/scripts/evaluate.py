"""Evaluate retrieval (cheap, default) or the full pipeline (--full, uses LLM quota).

    python scripts/evaluate.py --no-llm         # retrieval metrics, zero LLM calls
    python scripts/evaluate.py                  # retrieval metrics, LLM query planner on
    python scripts/evaluate.py --full --sleep 6 # end-to-end answers, citations, abstention

Writes eval/results_<mode>.md and .json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from taxqa.core.config import ROOT, settings
from taxqa.retrieval.section_refs import section_id
from taxqa.services.qa_pipeline import QAPipeline

EVAL = ROOT / "eval"


def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def eval_retrieval(p: QAPipeline, qs: list[dict], use_llm: bool) -> tuple[list[dict], dict]:
    rows = []
    for q in qs:
        t0 = time.perf_counter()
        r = p.retrieve(q["question"], use_llm=use_llm)
        latency = int((time.perf_counter() - t0) * 1000)
        ranked = list(dict.fromkeys(section_id(h.chunk.section) for h in r.hits))
        in_ctx = list(dict.fromkeys(section_id(h.chunk.section) for h in r.context))
        expected = {section_id(s) for s in q["expected_sections"]}
        first = next((i + 1 for i, s in enumerate(ranked) if s in expected), None)
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "expected": q["expected_sections"],
            "top_sections": ranked[:8],
            "context_sections": in_ctx,
            "sub_queries": [sq.question for sq in r.plan.sub_queries],
            "best_score": round(r.context[0].score, 3) if r.context else 0.0,
            "first_relevant_rank": first,
            "in_context": bool(expected & set(in_ctx)),
            "all_expected_in_context": expected <= set(in_ctx) if expected else None,
            "latency_ms": latency,
        })
        print(f"{q['id']}: rank={first} best={rows[-1]['best_score']:.2f} {latency} ms  top={ranked[:4]}")
    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]

    def hit_at(k: int) -> float:
        return round(sum(1 for r in ans if r["first_relevant_rank"] and r["first_relevant_rank"] <= k) / max(1, len(ans)), 3)

    metrics = {
        "mode": "retrieval" + ("" if use_llm else " (no LLM)"),
        "answerable_questions": len(ans),
        "unanswerable_questions": len(neg),
        "section_hit@1": hit_at(1),
        "section_hit@3": hit_at(3),
        "section_hit@5": hit_at(5),
        "section_hit@10": hit_at(10),
        "MRR": round(sum(1 / r["first_relevant_rank"] for r in ans if r["first_relevant_rank"]) / max(1, len(ans)), 3),
        "expected_section_in_llm_context": round(sum(r["in_context"] for r in ans) / max(1, len(ans)), 3),
        "all_expected_sections_in_context": round(sum(bool(r["all_expected_in_context"]) for r in ans) / max(1, len(ans)), 3),
        "abstention_correct_on_unanswerable": round(
            sum(r["best_score"] < settings.min_relevance for r in neg) / max(1, len(neg)), 3) if neg else None,
        "false_abstention_on_answerable": round(
            sum(r["best_score"] < settings.min_relevance for r in ans) / max(1, len(ans)), 3),
        "best_score_answerable_median": _median([r["best_score"] for r in ans]),
        "best_score_answerable_min": min((r["best_score"] for r in ans), default=None),
        "best_score_unanswerable_max": max((r["best_score"] for r in neg), default=None),
        "min_relevance_threshold": settings.min_relevance,
        "median_latency_ms": _median([r["latency_ms"] for r in rows]),
        "p90_latency_ms": sorted(r["latency_ms"] for r in rows)[int(0.9 * (len(rows) - 1))] if rows else None,
    }
    return rows, metrics


def eval_full(p: QAPipeline, qs: list[dict], sleep: float) -> tuple[list[dict], dict]:
    rows = []
    for i, q in enumerate(qs):
        if i and sleep:
            time.sleep(sleep)
        try:
            a = p.ask(q["question"])
        except Exception as e:  # one failure must not lose the whole run
            rows.append({"id": q["id"], "question": q["question"], "answerable": q["answerable"],
                         "expected": q["expected_sections"], "status": "exception", "error": str(e),
                         "cited_sections": [], "expected_cited": False if q["expected_sections"] else None,
                         "citations": 0, "valid_citations": 0, "untraceable_citations": 0, "statements": 0,
                         "removed_statements": 0, "confidence": None, "latency_ms": None, "summary": ""})
            print(f"{q['id']}: EXCEPTION {e}")
            continue
        kept = [pt for pt in a.points if not pt.removed]
        cites = [c for pt in a.points for c in pt.citations]
        cited = sorted({section_id(c.section) for pt in kept for c in pt.citations if c.found_in_sources and c.section})
        expected = {section_id(s) for s in q["expected_sections"]}
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "expected": q["expected_sections"],
            "status": a.status,
            "cited_sections": cited,
            "expected_cited": bool(expected & set(cited)) if expected else None,
            "citations": len(cites),
            "valid_citations": sum(c.valid for c in cites),
            "untraceable_citations": sum(not c.found_in_sources for c in cites),
            "statements": len(a.points),
            "removed_statements": len(a.points) - len(kept),
            "confidence": a.confidence.score,
            "latency_ms": a.timings_ms.get("total"),
            "summary": a.summary,
        })
        print(f"{q['id']}: {a.status:<12} cited={cited} conf={a.confidence.score} {a.timings_ms.get('total')} ms")
    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    total_c = sum(r["citations"] for r in rows)
    metrics = {
        "mode": "full",
        "answerable_questions": len(ans),
        "unanswerable_questions": len(neg),
        "answered_or_partial_on_answerable": round(sum(r["status"] in ("answered", "partial") for r in ans) / max(1, len(ans)), 3),
        "expected_section_cited": round(sum(bool(r["expected_cited"]) for r in ans) / max(1, len(ans)), 3),
        "citation_validity_rate": round(sum(r["valid_citations"] for r in rows) / max(1, total_c), 3),
        "untraceable_citation_rate": round(sum(r["untraceable_citations"] for r in rows) / max(1, total_c), 3),
        "statements_removed_by_verifier": sum(r["removed_statements"] for r in rows),
        "correct_refusal_on_unanswerable": round(
            sum(r["status"] in ("insufficient", "out_of_scope") for r in neg) / max(1, len(neg)), 3) if neg else None,
        "exceptions": sum(r["status"] == "exception" for r in rows),
        "median_confidence_answerable": _median([r["confidence"] for r in ans]),
        "median_latency_ms": _median([r["latency_ms"] for r in rows]),
    }
    return rows, metrics


def write_report(mode: str, rows: list[dict], metrics: dict) -> None:
    EVAL.mkdir(exist_ok=True)
    (EVAL / f"results_{mode}.json").write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=1), encoding="utf-8")
    md = [f"# Evaluation: {mode}", "", f"Run: {time.strftime('%Y-%m-%d %H:%M')}", "", "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in metrics.items()]
    md += ["", "| ID | Expected | Result | Latency (ms) |", "|---|---|---|---|"]
    for r in rows:
        result = r.get("cited_sections", r.get("top_sections"))
        flag = r.get("status") or (f"rank {r['first_relevant_rank']}" if r.get("first_relevant_rank") else "miss")
        md.append(f"| {r['id']} | {', '.join(r['expected']) or '(none)'} | {flag}: {', '.join(result or [])} | {r.get('latency_ms')} |")
    (EVAL / f"results_{mode}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="end-to-end evaluation (uses LLM quota)")
    ap.add_argument("--no-llm", "--no-planner", dest="no_llm", action="store_true",
                    help="retrieval only, without the LLM query planner (zero API calls)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between questions (free-tier rate limits)")
    ap.add_argument("--questions", type=Path, default=EVAL / "questions.jsonl")
    args = ap.parse_args()

    p = QAPipeline(settings)
    p.warmup()
    qs = load_questions(args.questions)
    if args.full:
        mode = "full"
        rows, metrics = eval_full(p, qs, args.sleep)
    else:
        mode = "retrieval" + ("_nollm" if args.no_llm else "")
        rows, metrics = eval_retrieval(p, qs, use_llm=not args.no_llm)
    write_report(mode, rows, metrics)
    print(json.dumps(metrics, indent=1))
    print(f"Report written to eval/results_{mode}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
