"""Stage 1: extract clean, page-tagged text lines from the statute PDF.

PyMuPDF gives us font size/flags and position per span, which we use to
  * re-join spans and same-baseline fragments into one logical line (the CBDT
    PDF prints "(50)" and the clause text as separate pieces, and puts no space
    after italic markers such as "(a)the sum"),
  * drop amendment/footnote markers: superscripts, and the small digits in
    "12[(2) The annual value ...",
  * drop running headers/footers that repeat on many pages,
  * keep the first span's style (bold/italic) - a bold, upright "24." is the
    strongest evidence that a line starts a section.

Footnote *blocks* ("12. Sub. by Act No. 7 of 2025 ...") are recognised later by
the section parser, because that needs knowledge of section starts.

PDFs whose text is drawn with Type3 bitmap fonts and no Unicode map (the
official FA 2026 print) are decoded first by type3_decoder.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3; the "fitz" name is deprecated
except ImportError:  # pragma: no cover
    import fitz

from taxqa.ingestion.type3_decoder import Type3Decoder, clean_type3_text

SUPERSCRIPT = 1
ITALIC = 2
BOLD = 16
SAME_BASELINE = 2.5            # points; fragments closer than this are one line
SMALL_FONT_SHARE = 0.01        # size-based footnotes only if >1% of text is small

_CHAR_FIXES = str.maketrans({
    "‐": "-", "‑": "-", " ": " ", "­": "", "​": "",
    "‘": "'", "’": "'", "“": '"', "”": '"',
})


@dataclass
class Line:
    page: int          # 1-based PDF page number
    text: str
    size: float        # dominant font size (by characters)
    bold: bool         # most of the line is bold
    italic: bool       # most of the line is italic
    x0: float
    y0: float
    page_height: float
    footnote: bool = False
    first_bold: bool = False    # style of the first visible span
    first_italic: bool = False
    x1: float = 0.0


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & BOLD) or "Bold" in span.get("font", "")


def _is_italic(span: dict) -> bool:
    font = span.get("font", "")
    return bool(span["flags"] & ITALIC) or "Italic" in font or "Oblique" in font


def _strip_marker(span: dict, body: float) -> str:
    """Return span text without footnote/amendment markers ('' if the span is only a marker)."""
    text = span["text"]
    small = span["size"] <= body - 1.5
    if span["flags"] & SUPERSCRIPT and re.fullmatch(r"\s*[\d*†‡]{1,4}\s*", text):
        return ""
    if small:
        if re.fullmatch(r"\s*\d{1,3}\s*", text):
            return ""
        text = re.sub(r"^\s*\d{1,3}(?=\[)", "", text)   # "71[***]" -> "[***]"
    return text


def _join_spans(spans: list[dict], body: float) -> tuple[str, list[dict]]:
    parts: list[str] = []
    kept: list[dict] = []
    prev_x1 = None
    for s in spans:
        text = _strip_marker(s, body)
        if not text.strip():
            continue
        if parts and prev_x1 is not None:
            gap = s["bbox"][0] - prev_x1
            if gap > 0.15 * s["size"] and not parts[-1].endswith(" ") and not text.startswith(" "):
                parts.append(" ")
        parts.append(text)
        kept.append(s)
        prev_x1 = s["bbox"][2]
    return "".join(parts), kept


def _merge_baselines(raw: list[tuple[tuple, list[dict]]]) -> list[tuple[tuple, list[dict]]]:
    """Merge fragments on the same baseline (left to right) into one line."""
    raw = sorted(raw, key=lambda r: (r[0][1], r[0][0]))
    out: list[tuple[tuple, list[dict]]] = []
    for bbox, spans in raw:
        if out and abs(bbox[1] - out[-1][0][1]) < SAME_BASELINE:
            prev_bbox, prev_spans = out[-1]
            items = sorted([(prev_bbox[0], prev_spans), (bbox[0], spans)], key=lambda t: t[0])
            merged = [s for _, ss in items for s in ss]
            out[-1] = (
                (min(prev_bbox[0], bbox[0]), min(prev_bbox[1], bbox[1]), max(prev_bbox[2], bbox[2]), max(prev_bbox[3], bbox[3])),
                sorted(merged, key=lambda s: s["bbox"][0]),
            )
        else:
            out.append((bbox, spans))
    return out


def extract_lines(pdf_path: Path) -> list[Line]:
    doc = fitz.open(pdf_path)
    # Pass 1: discover Type3 fonts document-wide, so bold/italic can be judged
    # against the regular face of the whole document.
    decoder = Type3Decoder(doc)
    has_type3 = False
    for page in doc:
        has_type3 = bool(decoder.page_fonts(page)) or has_type3
    if has_type3:
        decoder.classify_styles()

    pages: list[tuple[int, float, list]] = []
    sizes: Counter = Counter()
    for pno, page in enumerate(doc, start=1):
        raw = []
        if has_type3:
            # rawdict gives per-character positions, needed to drop filler glyphs.
            blocks = page.get_text("rawdict", sort=True)["blocks"]
            decoder.rewrite_spans(page, blocks)
        else:
            blocks = page.get_text("dict", sort=True)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for ln in block["lines"]:
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if spans:
                    raw.append((tuple(ln["bbox"]), spans))
                    for s in spans:
                        sizes[round(s["size"], 1)] += len(s["text"].strip())
        pages.append((pno, page.rect.height, _merge_baselines(raw)))
    doc.close()
    fix = clean_type3_text if has_type3 else (lambda t: t)

    body = sizes.most_common(1)[0][0] if sizes else 10.0
    lines: list[Line] = []
    for pno, height, raw in pages:
        for bbox, spans in raw:
            text, kept = _join_spans(spans, body)
            text = fix(re.sub(r"\s+", " ", text.translate(_CHAR_FIXES)).strip())
            # A leading marker glued to its text with no visual gap: "(2AD)Where" -> "(2AD) Where"
            text = re.sub(r"^(\[?\([0-9A-Za-z]{1,5}\))(?=[A-Za-z\"'])", r"\1 ", text)
            if not text or not kept:
                continue
            weights: Counter = Counter()
            bold_chars = italic_chars = 0
            for s in kept:
                n = len(s["text"].strip())
                weights[round(s["size"], 1)] += n
                bold_chars += n if _is_bold(s) else 0
                italic_chars += n if _is_italic(s) else 0
            total = sum(weights.values()) or 1
            lines.append(
                Line(
                    page=pno,
                    text=text,
                    size=weights.most_common(1)[0][0],
                    bold=bold_chars / total > 0.6,
                    italic=italic_chars / total > 0.6,
                    x0=kept[0]["bbox"][0],
                    y0=bbox[1],
                    page_height=height,
                    first_bold=_is_bold(kept[0]),
                    first_italic=_is_italic(kept[0]),
                    x1=bbox[2],
                )
            )
    return lines


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def body_font_size(lines: list[Line]) -> float:
    sizes: Counter = Counter()
    for ln in lines:
        sizes[ln.size] += len(ln.text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def clean_lines(lines: list[Line], band: float = 0.08, repeat_ratio: float = 0.3) -> list[Line]:
    """Remove repeated headers/footers and page numbers; flag small-font footnotes."""
    if not lines:
        return []
    n_pages = max(ln.page for ln in lines)

    def in_band(ln: Line) -> bool:
        return ln.y0 < ln.page_height * band or ln.y0 > ln.page_height * (1 - band)

    pages_with: Counter = Counter()
    for key in {(ln.page, _norm(ln.text)) for ln in lines if in_band(ln)}:
        pages_with[key[1]] += 1
    repeated = {k for k, c in pages_with.items() if c >= max(3, repeat_ratio * n_pages)}

    body = body_font_size(lines)
    total = sum(len(ln.text) for ln in lines) or 1
    small_share = sum(len(ln.text) for ln in lines if ln.size <= body - 1.0) / total
    size_footnotes = small_share > SMALL_FONT_SHARE

    out: list[Line] = []
    for ln in lines:
        if in_band(ln):
            if _norm(ln.text) in repeated:
                continue
            if re.fullmatch(r"[-–\s]*\d{1,4}[-–\s]*", ln.text):
                continue
        # Layouts with a separate footnote font: small text in the lower part of the page.
        if size_footnotes and ln.size <= body - 1.0 and ln.y0 > ln.page_height * 0.45:
            ln.footnote = True
        out.append(ln)
    return out


def load_clean_lines(pdf_path: Path) -> list[Line]:
    return clean_lines(extract_lines(pdf_path))
