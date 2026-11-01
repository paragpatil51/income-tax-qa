"""Parse the Act PDF, chunk it, and build the search indexes.

    python scripts/ingest.py                 # uses PDF_PATH from .env
    python scripts/ingest.py --pdf path.pdf
    python scripts/ingest.py --skip-embeddings   # parse + chunk only (fast, for tuning the parser)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import logging

from taxqa.core.config import settings
from taxqa.ingestion.chunker import chunk_sections
from taxqa.ingestion.pdf_reader import load_clean_lines
from taxqa.ingestion.section_parser import parse_sections
from taxqa.retrieval.embeddings import Embedder
from taxqa.retrieval.hybrid_index import build_dense_index

SPOT_CHECK = ["1", "2", "10", "23", "24", "80C", "80-IA", "115BAC", "139", "FIRST SCHEDULE"]

# Report entries that point at parsing problems; printed under "Check these".
WARNINGS = {
    "integer_gaps_in_section_numbers": "section numbers with no section found (some are genuinely not printed)",
    "strong_starts_rejected": "bold section-number lines NOT used as section starts",
    "toc_sections_missing_in_body": "listed in the table of contents but not found in the body",
    "untitled_sections": "sections without a heading",
    "near_empty_sections": "non-omitted sections with almost no text",
    "very_long_sections": "unusually long sections (check for a missed section start)",
    "pages_without_assigned_text": "pages whose text belongs to no section, schedule or footnote",
}


def print_report(report: dict) -> None:
    for k in ("mode", "pages", "lines", "body_start_page", "schedule_start_page", "toc_entries",
              "sections_found", "omitted_sections", "footnotes", "footnote_lines", "parts_found"):
        print(f"      {k}: {report.get(k)}")
    print(f"      chapters ({len(report['chapters_found'])}): {', '.join(report['chapters_found'])}")
    print(f"      schedules: {len(report['schedules_found'])}")
    flagged = [(k, report[k]) for k in WARNINGS if report.get(k)]
    if not flagged:
        print("      No structural warnings.")
        return
    print("      Check these:")
    for k, v in flagged:
        sample = v[:12]
        print(f"        - {WARNINGS[k]} ({len(v)}): {json.dumps(sample, ensure_ascii=False)[:300]}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=settings.pdf_path)
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="      %(message)s")   # model download / embedding progress
    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}\nDownload the official PDF and set PDF_PATH in .env (see README).")
        return 1

    t0 = time.time()
    print(f"[1/4] Extracting text from {args.pdf.name} ...")
    lines = load_clean_lines(args.pdf)
    if not lines:
        print("No extractable text in this PDF (scanned image?). Use the text PDF from incometaxindia.gov.in.")
        return 1
    print(f"      {len(lines)} lines across {max(l.page for l in lines)} pages")

    print("[2/4] Reconstructing chapters and sections ...")
    sections, report = parse_sections(lines)
    print_report(report)

    print("[3/4] Chunking ...")
    chunks = chunk_sections(sections)
    words = [len(c.text.split()) for c in chunks]
    print(f"      {len(chunks)} chunks (avg {sum(words) // max(1, len(words))} words, max {max(words, default=0)})")

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with settings.chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    settings.sections_path.write_text(
        json.dumps([asdict(s) for s in sections], ensure_ascii=False), encoding="utf-8"
    )
    report["source_pdf"] = args.pdf.name
    report["source_sha256"] = sha256(args.pdf)
    report["source_name"] = settings.source_name
    report["source_edition"] = settings.source_edition
    report["chunks"] = len(chunks)
    (settings.processed_dir / "ingest_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("      Spot check:")
    by_sec = {}
    for c in chunks:
        by_sec.setdefault(c.section, []).append(c)
    for num in SPOT_CHECK:
        cs = by_sec.get(num, [])
        if not cs:
            print(f"        Section {num}: NOT FOUND")
            continue
        chapter = f" ch.{cs[0].chapter}" if cs[0].chapter else ""
        print(f"        {num}{chapter} '{cs[0].section_title[:50]}' pp.{cs[0].page_start}-{cs[-1].page_end}: "
              f"{len(cs)} chunk(s) -> {', '.join(c.path for c in cs[:6])}{' ...' if len(cs) > 6 else ''}")

    if args.skip_embeddings:
        print("Skipping embeddings (--skip-embeddings).")
        return 0

    embedder = Embedder(settings)
    print(f"[4/4] Embedding {len(chunks)} chunks with {embedder.backend}:{embedder.name} "
          "(unchanged chunks reuse their vectors; the first run downloads the model) ...")
    embedded = build_dense_index(chunks, settings, embedder)
    print(f"      embedded {embedded} new or changed chunk(s), reused {len(chunks) - embedded}")
    print(f"Done in {time.time() - t0:.0f}s. Start the app with:  uvicorn taxqa.api.main:app --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
