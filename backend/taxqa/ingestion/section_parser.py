"""Stage 2: rebuild the Chapter -> Section structure of the Act from text lines.

The hard part is telling a real section start ("24. Income chargeable ...")
apart from lines that merely begin with a number: amendment footnotes
("24. Sub. by Act No. 08 of 2023 ..."), items inside Explanations, and wrapped
sentences. We:
  1. read the "Arrangement of Sections" (table of contents) when present,
  2. mark amendment-footnote blocks (italic "N. Sub./Ins. by ..." lines and the
     old text they quote) so they never count as statute text,
  3. collect every remaining line that *looks* like a section start and weight
     it by evidence: printed in bold (CBDT layout), listed in the ToC, preceded
     by a heading or a finished sentence, not preceded by "... section",
  4. keep the maximum-weight subsequence whose section numbers strictly
     increase in statutory order (10 < 10A < 10AA < 10B < 11; 80HHF < 80-I).
Step 4 discards stray matches because they break the ordering.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from taxqa.core.models import Section
from taxqa.core.section_numbers import SECTION_NUM, normalize_section, section_key
from taxqa.ingestion.pdf_reader import Line

SECTION_START = re.compile(rf"^\[?\s*(?P<num>{SECTION_NUM})\s*\.\s*(?P<rest>.*)$")
TOC_ENTRY = re.compile(rf"^\[?\s*(?P<num>{SECTION_NUM})\s*\.\s*(?P<title>\S.*)$")
# "81. to 85C. [Omitted by ...]" - one line standing for a range of omitted sections
SECTION_RANGE = re.compile(rf"^(?:to|and|&)\s+(?P<end>{SECTION_NUM})\s*\.")
CHAPTER = re.compile(r"^(?:CHAPTER|Chapter)\s+(?P<num>[IVXLC]+(?:\s*-?\s*[A-Z]{1,3})?)\s*$")
PART = re.compile(r"^(?P<letter>[A-Z]{1,2})\.\s*[—–-]+\s*(?P<title>\S.{0,120})$")
SCHEDULE = re.compile(
    r"^(?:THE\s+)?(?P<ord>FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    r"ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH)\s+SCHEDULE\b"
)
# Amendment footnotes: "12. Sub. by Act No. 7 of 2025 ...", "17a. Ins. by ...", "*Amendments made by ...".
# A pure-digit number (optionally with a lower-case letter) not followed by "[":
# "39. [Omitted by ...]" and "35B. ..." are sections.
FOOTNOTE_START = re.compile(r"^(?:\d{1,3}[a-z]?(?:\s*[-–,&]\s*\d{1,3}[a-z]?)?\.\s+(?!\[)\S|\*\s*\S)")
AMENDMENT_NOTE = re.compile(
    r"^\S+\s+(?:Sub\.|Subs\.|Ins\.|Omtt\.|Omitted|Substituted|Inserted|Renumbered|Re-numbered|"
    r"Italicised|Words?\b|Figures?\b|Brackets?\b|Letters?\b|Earlier\b|Prior\b|Original\b|Amended\b|"
    r"Items?\b|Clauses?\b|Provisos?\b|Explanations?\b|Sections?\b|Sub-sections?\b)"
)
# Footnotes cite the amending law; statute text in a numbered list or table does not.
AMENDMENT_SOURCE = re.compile(r"\bAct\.? No\.|\bw\.r?\.e\.f\b|\bibid\b|\bFinance (?:\(No\. ?\d\) )?Act\b|\bAmendments made\b")
BODY_START_PHRASE = re.compile(r"may be called the Income-?tax Act", re.I)
WRAPPED_TAIL = re.compile(
    r"(section|sections|clause|clauses|sub-section|sub-sections|sub-clause|item|paragraph|rule|"
    r"and|or|of|to|under|the|in|with|,)$",
    re.I,
)

# Candidates with weaker evidence than this (e.g. not bold in a bold layout, or
# following "... section") are discarded even if they would fit the numbering order.
MIN_WEIGHT = 0.5
MIN_BOLD_STARTS = 5      # bold "N." lines needed before the bold layout is assumed
BOLD_SHARE = 0.3         # ... and they must be at least this share of numbered lines
MARGIN_TOLERANCE = 4.0
SMALL_WORDS = {"of", "to", "and", "in", "the", "for", "on", "from", "by", "a", "an", "or", "under", "with"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def nice_title(text: str) -> str:
    """'COMPUTATION OF TOTAL INCOME' -> 'Computation of Total Income'."""
    words = text.lower().split()
    return " ".join(w if (i and w in SMALL_WORDS) else w[:1].upper() + w[1:] for i, w in enumerate(words))


def chapter_number(raw: str) -> str:
    """'VIA' / 'VI-A' / 'XII - BA' -> 'VI-A' / 'XII-BA'."""
    s = re.sub(r"\s+", "", raw)
    m = re.fullmatch(r"([IVXLC]+)-?([A-Z]{0,3})", s)
    if not m:
        return s
    roman, suffix = m.groups()
    return f"{roman}-{suffix}" if suffix else roman


@dataclass
class _Candidate:
    idx: int
    num: str
    key: tuple[int, str]
    weight: float
    strong: bool


def clean_title(text: str) -> str:
    """Strip amendment brackets from a heading: 'Tax on income of individuals [, HUF]' -> '... individuals, HUF'."""
    t = re.sub(r"\[\*+\]|[\[\]]", "", text)
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip(" .")


def is_footnote_start(ln: Line, margin: float | None = None) -> bool:
    """An amendment footnote: "12. Sub. by Act No. 7 of 2025, w.e.f. 1-4-2025."

    Style alone is not enough (footnotes are printed upright or italic, and
    amended statute text is italic too), so the line must also read like an
    amendment note. Numbered statute rows such as the 115BAC rate table
    ("1. Upto Rs. 3,00,000 Nil") therefore stay in the section.
    """
    t = ln.text
    if not FOOTNOTE_START.match(t):
        return False
    if ln.first_bold and not ln.first_italic:
        return False  # a bold upright number is a section start
    if AMENDMENT_NOTE.match(t) or AMENDMENT_SOURCE.search(t[:160]):
        return True
    # A run-on footnote line without the vocabulary: accept only where the CBDT layout puts
    # footnote numbers (indented ~3pt from section numbers) *and* in footnote italics.
    return ln.first_italic and margin is not None and margin + 1.5 <= ln.x0 <= margin + 8 and len(t) < 60


def is_strong_start(ln: Line) -> bool:
    """Bold section number, as printed in the CBDT layout ("24." in bold, upright or bold-italic)."""
    return bool(SECTION_START.match(ln.text)) and ln.first_bold and not is_footnote_start(ln)


def _is_structural(ln: Line, margin: float | None) -> bool:
    t = ln.text.strip()
    return (
        is_strong_start(ln)
        or bool(CHAPTER.match(t))
        or bool(SCHEDULE.match(t) and t.isupper())
        or _is_part_heading(ln, margin)
    )


def _is_part_heading(ln: Line, margin: float | None) -> bool:
    return bool(PART.match(ln.text.strip())) and ln.x0 > (margin if margin is not None else 45.0) + 60


def _left_margin(lines: list[Line]) -> float | None:
    xs = Counter(round(ln.x0) for ln in lines if is_strong_start(ln))
    return float(xs.most_common(1)[0][0]) if xs else None


def _is_heading_like(ln: Line, margin: float | None) -> bool:
    """Bold section heading at the left margin; amended headings start with "[" and may be italic."""
    t = ln.text
    if not ln.bold or t.isupper() or FOOTNOTE_START.match(t) or SECTION_START.match(t):
        return False
    return margin is None or margin - MARGIN_TOLERANCE <= ln.x0 <= margin + 3 * MARGIN_TOLERANCE


def _ends_heading(text: str) -> bool:
    return text.rstrip(" ]").endswith(".")


def mark_footnote_blocks(lines: list[Line], margin: float | None, follow_blocks: bool = True) -> list[tuple[int, str]]:
    """Flag amendment-footnote blocks; return (line index, full footnote text) per footnote.

    A block starts at an "N. Sub./Ins. by ..." line and runs - including any old
    text it quotes - until the next structural line: a bold section start,
    chapter, part or schedule heading. Bold heading lines right before a section
    start are un-flagged again, since they are that section's heading.

    Blocks are only followed in layouts with bold section numbers
    (`follow_blocks`); elsewhere nothing reliable ends a block, so only the
    footnote line itself is flagged. In layouts that print footnotes at the foot
    of the page in a smaller font (the FA 2026 print), a block that starts in
    that font also ends at the first line back in body size.
    """
    sizes: Counter = Counter()
    for ln in lines:
        sizes[ln.size] += len(ln.text)
    body = sizes.most_common(1)[0][0] if sizes else 0.0
    notes: list[tuple[int, str]] = []
    in_block = False
    block_small = False
    for i, ln in enumerate(lines):
        if is_footnote_start(ln, margin):
            in_block = follow_blocks
            block_small = ln.size <= body - 1.0
            ln.footnote = True
            notes.append((i, ln.text))
            continue
        if in_block:
            if block_small and ln.size > body - 1.0:
                in_block = False   # page-foot footnote ended; this is statute text again
            elif _is_structural(ln, margin):
                in_block = False
                k, n = i - 1, 0
                while k >= 0 and n < 3 and lines[k].footnote and _is_heading_like(lines[k], margin):
                    lines[k].footnote = False
                    k, n = k - 1, n + 1
            else:
                ln.footnote = True
                if notes:
                    idx, text = notes[-1]
                    notes[-1] = (idx, f"{text} {ln.text}")
    # Footnotes flagged by font size (other layouts) become notes too.
    seen = {i for i, _ in notes}
    for i, ln in enumerate(lines):
        if ln.footnote and i not in seen and (i == 0 or not lines[i - 1].footnote):
            notes.append((i, ln.text))
    return sorted(notes)


def find_body_start(lines: list[Line]) -> int:
    for i, ln in enumerate(lines):
        if BODY_START_PHRASE.search(ln.text):
            for j in range(i, max(-1, i - 4), -1):
                m = SECTION_START.match(lines[j].text)
                if m and m.group("num") == "1":
                    return j
            return i
    return 0


def parse_toc(lines: list[Line], body_start: int) -> dict[str, str]:
    """Map section number -> title from the arrangement of sections, if the PDF has one."""
    toc: dict[str, str] = {}
    last: str | None = None
    for ln in lines[:body_start]:
        text = ln.text.strip()
        if CHAPTER.match(text) or SCHEDULE.match(text):
            last = None
            continue
        m = TOC_ENTRY.match(text)
        if m:
            last = normalize_section(m.group("num"))
            toc[last] = m.group("title").strip()
        elif last and not toc[last].endswith(".") and not text.isupper():
            toc[last] += " " + text
    return {k: v.rstrip(" .") for k, v in toc.items()}


def _schedule_start(lines: list[Line], body_start: int) -> int:
    for i in range(body_start, len(lines)):
        t = lines[i].text.strip()
        if SCHEDULE.match(t) and t.isupper() and len(t) < 80 and not lines[i].footnote:
            return i
    return len(lines)


def _prev_body_line(lines: list[Line], i: int) -> Line | None:
    j = i - 1
    while j >= 0 and lines[j].footnote:
        j -= 1
    return lines[j] if j >= 0 else None


def _candidates(lines: list[Line], start: int, end: int, toc: dict[str, str], bold_mode: bool,
                margin: float | None) -> list[_Candidate]:
    out = []
    for i in range(start, end):
        ln = lines[i]
        if ln.footnote:
            continue
        m = SECTION_START.match(ln.text)
        if not m:
            continue
        num = normalize_section(m.group("num"))
        strong = is_strong_start(ln)
        weight = 1.0
        if toc:
            weight += 1.0 if num in toc else -0.7
        if bold_mode:
            weight += 1.5 if strong else -1.0
            if margin is not None and abs(ln.x0 - margin) > MARGIN_TOLERANCE * 2:
                weight -= 0.5
        prev = _prev_body_line(lines, i)
        if prev is not None:
            ptext = prev.text.rstrip()
            if WRAPPED_TAIL.search(ptext):
                weight -= 0.8
            elif (prev.bold and not prev.italic) or ptext.endswith((".", ":", "—", "]", ";")):
                weight += 0.5
            if toc.get(num) and _norm(toc[num])[:40] in _norm(ptext):
                weight += 1.0
        if weight >= MIN_WEIGHT:
            out.append(_Candidate(i, num, section_key(num), weight, strong))
    return out


def _best_increasing(cands: list[_Candidate]) -> list[_Candidate]:
    """Maximum-weight subsequence with strictly increasing section keys (O(n^2))."""
    n = len(cands)
    if n == 0:
        return []
    best = [c.weight for c in cands]
    prev = [-1] * n
    for i in range(n):
        ki = cands[i].key
        for j in range(i):
            if cands[j].key < ki and best[j] + cands[i].weight > best[i]:
                best[i] = best[j] + cands[i].weight
                prev[i] = j
    i = max(range(n), key=best.__getitem__)
    chain = []
    while i != -1:
        chain.append(cands[i])
        i = prev[i]
    return chain[::-1]


def _heading_start(lines: list[Line], idx: int, title: str, floor: int, margin: float | None,
                   bold_mode: bool) -> int:
    """Return the index where the heading block above a section start begins."""
    ntitle = _norm(title)
    j = idx
    for n in range(3):
        k = j - 1
        if k <= floor or lines[k].footnote:
            break
        ln, t = lines[k], _norm(lines[k].text)
        if ntitle and t and t in ntitle:
            j = k
            continue
        if bold_mode:
            # Last heading line ends with "."; earlier lines of a wrapped heading do not.
            ok = _is_heading_like(ln, margin) and (_ends_heading(ln.text) == (n == 0))
        else:
            ok = ln.bold and not ln.text.isupper()
        if not ok:
            break
        j = k
    return j


def _chapters(lines: list[Line], start: int, end: int, margin: float | None):
    """Chapter headings (index, number, title, index after heading) and part headings (index, label)."""
    chapters, parts = [], []
    i = start
    while i < end:
        t = lines[i].text.strip()
        m = CHAPTER.match(t)
        if m and not lines[i].footnote:
            title_parts = []
            k = i + 1
            while k < end and len(title_parts) < 3:
                tt = lines[k].text.strip()
                centred_bold = lines[k].bold and lines[k].x0 > (margin if margin is not None else 45.0) + 40
                if (tt.isupper() or centred_bold) and not SECTION_START.match(tt) and not CHAPTER.match(tt) \
                        and not _is_part_heading(lines[k], margin):
                    title_parts.append(tt)
                    k += 1
                else:
                    break
            chapters.append((i, chapter_number(m.group("num")), " - ".join(nice_title(t) for t in title_parts), k))
            i = k
            continue
        if not lines[i].footnote and _is_part_heading(lines[i], margin):
            pm = PART.match(t)
            parts.append((i, f"{pm.group('letter')}. {pm.group('title').strip()}"))
        i += 1
    return chapters, parts


def parse_sections(lines: list[Line]) -> tuple[list[Section], dict]:
    body_start = find_body_start(lines)
    margin = _left_margin(lines[body_start:])
    bold_layout = sum(1 for ln in lines[body_start:] if is_strong_start(ln)) >= MIN_BOLD_STARTS
    footnotes = mark_footnote_blocks(lines, margin, follow_blocks=bold_layout)
    toc = parse_toc(lines, body_start)
    sched_start = _schedule_start(lines, body_start)

    # Chapter I and section 1's heading sit just above the body-start line.
    scan_start = next(
        (k for k in range(body_start - 1, max(-1, body_start - 8), -1) if CHAPTER.match(lines[k].text.strip())),
        body_start,
    )

    body = [ln for ln in lines[body_start:sched_start] if not ln.footnote]
    strong_total = sum(1 for ln in body if is_strong_start(ln))
    numbered = sum(1 for ln in body if SECTION_START.match(ln.text))
    # Bold numbers mark section starts when a good share of numbered lines are bold.
    bold_mode = bold_layout and strong_total >= BOLD_SHARE * numbered
    cands = _candidates(lines, body_start, sched_start, toc, bold_mode, margin)
    chain = _best_increasing(cands)
    chapters, parts = _chapters(lines, scan_start, sched_start, margin)
    excluded = {i for c in chapters for i in range(c[0], c[3])} | {i for i, _ in parts}

    starts = []
    for n, c in enumerate(chain):
        floor = chain[n - 1].idx if n else scan_start - 1
        starts.append(_heading_start(lines, c.idx, toc.get(c.num, ""), floor, margin, bold_mode))

    sections: list[Section] = []
    ranges: dict[str, str] = {}
    for n, c in enumerate(chain):
        end = starts[n + 1] if n + 1 < len(chain) else sched_start
        title = toc.get(c.num, "")
        if not title and starts[n] < c.idx:
            title = " ".join(lines[k].text for k in range(starts[n], c.idx) if k not in excluded)
        title = clean_title(title)
        rest = SECTION_START.match(lines[c.idx].text).group("rest")
        if (rm := SECTION_RANGE.match(rest)):
            ranges[c.num] = normalize_section(rm.group("end"))
            title = title or f"Sections {c.num} to {ranges[c.num]}"
        chap = next(((num, t, ci) for (ci, num, t, _) in reversed(chapters) if ci < c.idx), ("", "", -1))
        part = next((label for (pi, label) in reversed(parts) if chap[2] < pi < c.idx), "")
        chapter_title = f"{chap[1]} - {part}" if chap[1] and part else (chap[1] or part)
        body = [
            (lines[k].page, lines[k].text)
            for k in range(c.idx, end)
            if not lines[k].footnote and k not in excluded
        ]
        if not body:
            continue
        pages = [p for p, _ in body]
        notes = [text[:600] for (i, text) in footnotes if c.idx <= i < end]
        head = re.sub(rf"^(?:to|and|&)\s+{SECTION_NUM}\s*\.\s*", "", rest)[:80].lower()
        sections.append(
            Section(
                num=c.num,
                title=title,
                chapter=chap[0],
                chapter_title=chapter_title,
                page_start=min(pages),
                page_end=max(pages),
                lines=body,
                notes=notes[:40],
                omitted=head.lstrip("[ ").startswith("omitted") or head.startswith("[***]"),
            )
        )

    sections.extend(_parse_schedules(lines, sched_start, footnotes))
    report = _report(lines, sections, toc, cands, chain, chapters, parts, footnotes,
                     body_start, sched_start, bold_mode, margin, ranges)
    return sections, report


def _parse_schedules(lines: list[Line], start: int, footnotes: list[tuple[int, str]]) -> list[Section]:
    heads = [
        i for i in range(start, len(lines))
        if not lines[i].footnote and SCHEDULE.match(lines[i].text.strip()) and lines[i].text.strip().isupper()
    ]
    out = []
    for n, i in enumerate(heads):
        end = heads[n + 1] if n + 1 < len(heads) else len(lines)
        body = [(lines[k].page, lines[k].text) for k in range(i, end) if not lines[k].footnote]
        if len(body) < 2:
            continue
        name = SCHEDULE.match(lines[i].text.strip()).group("ord") + " SCHEDULE"
        # Title: first upper-case line after the heading, skipping "[See section 44]"
        title = next((t for _, t in body[1:4] if t.isupper() and not t.startswith("[")), "")
        pages = [p for p, _ in body]
        out.append(
            Section(
                num=name,
                title=nice_title(title),
                chapter="",
                chapter_title="",
                page_start=min(pages),
                page_end=max(pages),
                lines=body,
                notes=[text[:600] for (k, text) in footnotes if i <= k < end][:40],
                kind="schedule",
            )
        )
    return out


def _report(lines, sections, toc, cands, chain, chapters, parts, footnotes,
            body_start, sched_start, bold_mode, margin, ranges) -> dict:
    """Diagnostics for tuning the parser on a new PDF edition."""
    secs = [s for s in sections if s.kind == "section"]
    found = {s.num for s in secs}
    chosen = {c.idx for c in chain}

    ints = {section_key(n)[0] for n in found}
    for start, end in ranges.items():  # "81. to 85C." covers 81..85
        ints |= set(range(section_key(start)[0], section_key(end)[0] + 1))
    ints = sorted(ints)
    gaps = [str(k) for a, b in zip(ints, ints[1:]) for k in range(a + 1, b)]

    covered = {p for s in sections for p, _ in s.lines} | {ln.page for ln in lines if ln.footnote}
    first_page = lines[body_start].page if lines else 1
    last_page = max((ln.page for ln in lines), default=0)
    pages_with_text = {ln.page for ln in lines}
    uncovered = [p for p in range(first_page, last_page + 1) if p in pages_with_text and p not in covered]

    words = {s.num: len(s.text.split()) for s in secs}
    return {
        "mode": "toc-guided" if toc else ("bold-evidence" if bold_mode else "heuristic"),
        "lines": len(lines),
        "pages": last_page,
        "left_margin_x": margin,
        "body_start_page": first_page if lines else None,
        "schedule_start_page": lines[sched_start].page if sched_start < len(lines) else None,
        "toc_entries": len(toc),
        "sections_found": len(found),
        "omitted_sections": sum(s.omitted for s in secs),
        "omitted_ranges": [f"{a}-{b}" for a, b in ranges.items()],
        "untitled_sections": [s.num for s in secs if not s.title][:50],
        "schedules_found": [s.num for s in sections if s.kind == "schedule"],
        "chapters_found": [c[1] for c in chapters],
        "parts_found": len(parts),
        "footnotes": len(footnotes),
        "footnote_lines": sum(ln.footnote for ln in lines),
        "integer_gaps_in_section_numbers": gaps[:200],
        "toc_sections_missing_in_body": sorted(set(toc) - found, key=section_key)[:200],
        "body_sections_not_in_toc": sorted(found - set(toc), key=section_key)[:200] if toc else [],
        "strong_starts_rejected": [
            {"num": c.num, "page": lines[c.idx].page, "text": lines[c.idx].text[:80]}
            for c in cands if c.strong and c.idx not in chosen
        ][:100],
        "near_empty_sections": [n for n, w in words.items() if w < 6 and not next(s for s in secs if s.num == n).omitted][:50],
        "very_long_sections": sorted(
            [{"num": s.num, "words": words[s.num], "pages": s.page_end - s.page_start + 1}
             for s in secs if words[s.num] > 15000 or s.page_end - s.page_start > 30],
            key=lambda d: -d["words"],
        )[:20],
        "pages_without_assigned_text": uncovered[:100],
    }
