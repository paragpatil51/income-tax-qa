"""Ask a question from the command line.

    python scripts/ask.py "Can I claim deduction for interest on a housing loan?"
    python scripts/ask.py --json "..."
"""

from __future__ import annotations

import argparse
import sys

from taxqa.services.qa_pipeline import QAPipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--json", action="store_true", help="print the full JSON response")
    args = ap.parse_args()

    pipeline = QAPipeline()
    ans = pipeline.ask(args.question)
    if args.json:
        print(ans.model_dump_json(indent=2))
        return 0

    print(f"\nSTATUS: {ans.status}    CONFIDENCE: {ans.confidence.score if ans.confidence.score is not None else 'n/a'}"
          f" ({ans.confidence.band})\n")
    print(ans.summary, "\n")
    for p in ans.points:
        if p.removed:
            continue
        print(f"* {p.heading}: {p.statement}")
        for c in p.citations:
            ok = "OK " if c.valid else "?? "
            pages = f"PDF p.{c.page_start}" + (f"-{c.page_end}" if c.page_end != c.page_start else "") if c.page_start else ""
            print(f"    [{ok}] {c.provision}  {pages}  \"{c.quote[:90]}\"")
    for w in ans.warnings:
        print(f"! {w}")
    removed = [p for p in ans.points if p.removed]
    if removed:
        print(f"\n{len(removed)} statement(s) removed by verification:")
        for p in removed:
            print(f"  - {p.statement[:100]}  ({p.removal_reason})")
    print(f"\nTimings (ms): {ans.timings_ms}")
    print(f"Models: {ans.models}")
    usage = pipeline.llm.cache.usage_today()
    print("LLM API calls today (UTC): " + (", ".join(f"{m} {u['calls']} ({u['errors']} failed)" for m, u in usage.items()) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
