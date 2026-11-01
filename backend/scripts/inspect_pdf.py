"""Debugging aid for the parser: show extracted lines with their layout features.

    python scripts/inspect_pdf.py --pages 1-3
    python scripts/inspect_pdf.py --find "^24\\."
    python scripts/inspect_pdf.py --section 24          # parsed chunks (needs ingest first)
    python scripts/inspect_pdf.py --footnotes --pages 130-135

Flags column: B/I = line mostly bold/italic, b/i = first span bold/italic,
F = footnote (excluded from statute text), S = bold section-number line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from taxqa.core.config import settings
from taxqa.core.section_numbers import normalize_section
from taxqa.ingestion.pdf_reader import body_font_size, load_clean_lines
from taxqa.ingestion.section_parser import is_strong_start, parse_sections


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(settings.pdf_path))
    ap.add_argument("--pages", help="page range, e.g. 40-42")
    ap.add_argument("--find", help="regex to search for in lines")
    ap.add_argument("--section", help="print the parsed chunks of a section (needs ingest first)")
    ap.add_argument("--footnotes", action="store_true", help="only show lines flagged as footnotes")
    args = ap.parse_args()

    if args.section:
        want = normalize_section(args.section) if re.match(r"\d", args.section) else args.section.upper()
        for line in settings.chunks_path.open(encoding="utf-8"):
            c = json.loads(line)
            if c["section"].upper() == want:
                print(f"--- {c['id']}  path={c['path']}  pages={c['page_start']}-{c['page_end']}  refs={c['refs']}")
                if c["context"]:
                    print(f"CONTEXT: {c['context']}")
                print(c["text"], "\n")
        return 0

    lines = load_clean_lines(args.pdf)
    parse_sections(lines)  # flags footnote blocks in place
    print(f"body font size: {body_font_size(lines)}")
    lo, hi = 1, 10**6
    if args.pages:
        a, _, b = args.pages.partition("-")
        lo, hi = int(a), int(b or a)
    pattern = re.compile(args.find) if args.find else None
    for ln in lines:
        if not lo <= ln.page <= hi:
            continue
        if pattern and not pattern.search(ln.text):
            continue
        if args.footnotes and not ln.footnote:
            continue
        flags = (
            ("B" if ln.bold else "-") + ("I" if ln.italic else "-")
            + ("b" if ln.first_bold else "-") + ("i" if ln.first_italic else "-")
            + ("F" if ln.footnote else "-") + ("S" if is_strong_start(ln) else "-")
        )
        print(f"p{ln.page:>4} x{ln.x0:>5.0f} y{ln.y0:>5.0f} {ln.size:>4} {flags} | {ln.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
