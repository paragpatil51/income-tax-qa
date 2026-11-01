"""Stage 3: split each section into citation-addressable chunks.

A chunk is normally one top-level subdivision of a section, e.g. 80C(2) or
24(b), together with its own sub-clauses, provisos and Explanations. The
section's opening words ("Income chargeable ... shall be computed after making
the following deductions, namely:-") are kept as `context`, because a clause
such as 24(a) "a sum equal to thirty per cent of the annual value" means
nothing without them.

Very small neighbouring units are merged and very large ones are split, so
every chunk stays a sensible size for embedding and for the LLM context.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from taxqa.core.models import Chunk, Section
from taxqa.core.section_numbers import SECTION_NUM, normalize_section, section_key

SUB = re.compile(r"^[\[\s]*\((?P<m>\d{1,3}[A-Z]{0,4})\)")
CLAUSE = re.compile(r"^[\[\s]*\((?P<m>[a-z]{1,5})\)")
PROVISO = re.compile(r"^[\[\s]*Provided\b")
EXPLANATION = re.compile(r"^[\[\s]*Explanation\b")
SECTION_PREFIX = re.compile(rf"^[\[\s]*{SECTION_NUM}\s*\.\s*")
ROMAN = re.compile(r"^(?=[ivxl])(x{0,3})(ix|iv|v?i{0,3})$")
# A wrapped cross-reference, not a new subdivision: "... referred to in sub-section" / "(2) of section 23"
REFERENCE_TAIL = re.compile(
    r"\b(sub-section|section|sections|clause|clauses|sub-clause|item|sub-item|paragraph|"
    r"sub-paragraph|rule|sub-rule|Explanation|proviso)\s*$"
)
MARKER_AS_REFERENCE = re.compile(r"^[\[\s]*\([0-9A-Za-z]{1,6}\)\s*(?:of|or|and|to|thereof|above|below|,)(?:\b|\s|$)")
SCHEDULE_RULE = re.compile(r"^\[?\s*(?P<n>\d{1,3}[A-Z]{0,2})\s*\.\s+\S")
SCHEDULE_PART = re.compile(r"^(?:[A-Z]{1,2}\.\s*[—–-]+\s*\S|PART\s+[A-Z0-9]+\b)")
RULE_WORD = {"FIRST": "rule", "SECOND": "rule", "THIRD": "rule", "FOURTH": "rule"}

TINY_WORDS = 25
MIN_WORDS = 60
MERGE_TARGET = 260
MAX_WORDS = 380
WINDOW_OVERLAP = 50


@dataclass
class Para:
    text: str
    pages: list[int] = field(default_factory=list)
    sub: str | None = None      # "(1)" style marker
    clause: str | None = None   # "(a)" / "(i)" style marker

    @property
    def words(self) -> int:
        return len(self.text.split())


def _marker_text(text: str, first: bool) -> str:
    return SECTION_PREFIX.sub("", text, count=1) if first else text


def _starts_paragraph(text: str, prev: str = "") -> bool:
    if PROVISO.match(text) or EXPLANATION.match(text):
        return True
    if not (SUB.match(text) or CLAUSE.match(text)):
        return False
    return not (REFERENCE_TAIL.search(prev) or MARKER_AS_REFERENCE.match(text))


WORDISH = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
HYPHEN_BREAK = re.compile(r"([A-Za-z]+)-$")


def build_vocabulary(sections: list[Section]) -> Counter:
    """Word and hyphenated-compound counts from within lines (never across a line break)."""
    vocab: Counter = Counter()
    for s in sections:
        for _, line in s.lines:
            body = line[:-1] if line.endswith("-") else line
            vocab.update(w.lower() for w in WORDISH.findall(body))
    return vocab


def join_lines(prev: str, text: str, vocab: Counter | None = None) -> str:
    """Join a wrapped line. A line-end hyphen is a typesetting break ("Commis-" + "sioner")
    when the Act elsewhere spells the joined word more often than the hyphenated form;
    otherwise it is a real compound ("one-" + "fifth") and is kept."""
    if not prev.endswith("-"):
        return f"{prev} {text}"
    left, right = HYPHEN_BREAK.search(prev), re.match(r"[a-z]+", text)
    if vocab and left and right:
        whole = (left.group(1) + right.group(0)).lower()
        if vocab[whole] > vocab[f"{left.group(1)}-{right.group(0)}".lower()]:
            return prev[:-1] + text
    return prev + text


def build_paragraphs(section: Section, vocab: Counter | None = None) -> list[Para]:
    paras: list[Para] = []
    prev = ""
    for n, (page, raw) in enumerate(section.lines):
        text = raw.strip()
        if not text:
            continue
        probe = _marker_text(text, first=(n == 0))
        starts = n == 0 or _starts_paragraph(probe, prev)
        prev = text
        if starts:
            p = Para(text=text, pages=[page])
            if (m := SUB.match(probe)):
                p.sub = m.group("m")
            elif (m := CLAUSE.match(probe)):
                p.clause = m.group("m")
            paras.append(p)
        else:
            cur = paras[-1]
            cur.text = join_lines(cur.text, text, vocab)
            if page not in cur.pages:
                cur.pages.append(page)
    return paras


def _merge_limit(words: int, prev_words: int) -> int:
    """A tiny unit (a one-line clause) may join a neighbour up to the hard chunk limit."""
    return MAX_WORDS if min(words, prev_words) < TINY_WORDS else MERGE_TARGET


def _sub_value(m: str) -> tuple[int, str]:
    mm = re.match(r"(\d+)([A-Z]*)", m)
    return int(mm.group(1)), mm.group(2)


SENTENCE_START = re.compile(r"^[\[\s]*\(\d{1,3}[A-Z]{0,4}\)\s*[\[\"'“]?[A-Z\"'“]")


def _best_subsections(paras: list[Para]) -> list[int]:
    """Top-level sub-sections as the maximum-weight subsequence of strictly increasing numbers.

    Taking every increasing "(n)" is not enough: a numbered list inside a sub-section
    ("(3) the Life Insurance Corporation, or (4) the National Housing Bank, ...") would be read
    as sub-sections (3)-(8) and swallow the real ones that follow. Real sub-sections start like
    a sentence ("(3) The provisions ...", "(1) "advance tax" means") after a finished one, so
    they carry more weight than list items and win the ordering contest.
    """
    cands = []
    for i, p in enumerate(paras):
        if not p.sub:
            continue
        w = 1.0
        if SENTENCE_START.match(_marker_text(p.text, i == 0)):
            w += 1.0
        if i == 0 or paras[i - 1].text.rstrip(" ]").endswith((".", ":", "—", ":—")):
            w += 0.5
        cands.append((i, _sub_value(p.sub), w))
    best = [w for _, _, w in cands]
    prev = [-1] * len(cands)
    for a in range(len(cands)):
        for b in range(a):
            if cands[b][1] < cands[a][1] and best[b] + cands[a][2] > best[a]:
                best[a], prev[a] = best[b] + cands[a][2], b
    if not cands:
        return []
    k = max(range(len(cands)), key=best.__getitem__)
    chain = []
    while k != -1:
        chain.append(cands[k][0])
        k = prev[k]
    return chain[::-1]


def top_level_indices(paras: list[Para]) -> tuple[list[int], str]:
    """Indices of paragraphs that open a top-level subdivision, and its style."""
    first = next((p for p in paras if p.sub or p.clause), None)
    if first is None:
        return [], ""
    tops: list[int] = []
    if first.sub:
        return _best_subsections(paras), "sub"

    roman_list = first.clause == "i"
    last_alpha = ""
    for i, p in enumerate(paras):
        m = p.clause
        if not m:
            continue
        if roman_list:
            if ROMAN.match(m):
                tops.append(i)
            continue
        is_roman = bool(ROMAN.match(m))
        # "(i)" right after "(h)" is clause (i), otherwise it is a sub-clause.
        next_letter = chr(ord(last_alpha[0]) + 1) if last_alpha else "a"
        if is_roman and not (len(m) == 1 and m == next_letter):
            continue
        if not last_alpha or m[0] >= last_alpha[0]:
            tops.append(i)
            last_alpha = m
    return tops, "clause"


def _label(p: Para) -> str:
    return p.sub or p.clause or ""


REF_LIST = re.compile(rf"\s*({SECTION_NUM})\b((?:\s*(?:,|and|or|to)\s*{SECTION_NUM}\b)*)")
# "section 2 of the Finance Act, 2024": the other statute must follow the number directly,
# so "sections 80C, 80D and 80E, but not section 2 of the Finance Act" keeps 80C/80D/80E.
OTHER_ACT = re.compile(
    rf"\s*{SECTION_NUM}(?:\s*\([^)]{{0,8}}\))*\s*,?\s+of\s+the\s+(?!Income-?tax\s+Act)[A-Z][\w\s,()-]*?Act\b"
)


def _find_refs(text: str, own: str) -> list[str]:
    """Sections cross-referenced as "section 23", "section 80-IA" or "sections 23, 24 and 25"."""
    refs: list[str] = []
    for m in re.finditer(r"\bsections?\s+", text, flags=re.I):
        window = text[m.end(): m.end() + 80]
        if OTHER_ACT.match(window):
            continue  # e.g. "section 2 of the Finance Act, 2024"
        window = re.sub(r"\s*\([^)]{0,8}\)", "", window)  # drop "(1)", "(b)" pinpoints
        lm = REF_LIST.match(window)
        if not lm:
            continue
        for raw in [lm.group(1)] + re.findall(SECTION_NUM, lm.group(2)):
            num = normalize_section(raw)
            if num != own and num not in refs and section_key(num)[0] <= 300:
                refs.append(num)
    return refs


def _window_split(text: str, size: int = MAX_WORDS - 80, overlap: int = WINDOW_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return out


def _split_large(paras: list[Para]) -> list[list[Para]]:
    """Group paragraphs into pieces under MAX_WORDS; window-split giant paragraphs."""
    groups: list[list[Para]] = []
    cur: list[Para] = []
    count = 0
    for p in paras:
        if p.words > MAX_WORDS:
            if cur:
                groups.append(cur)
                cur, count = [], 0
            for piece in _window_split(p.text):
                groups.append([Para(text=piece, pages=p.pages, sub=p.sub, clause=p.clause)])
            continue
        if cur and count + p.words > MAX_WORDS:
            groups.append(cur)
            cur, count = [], 0
        cur.append(p)
        count += p.words
    if cur:
        groups.append(cur)
    return groups


def chunk_section(section: Section, vocab: Counter | None = None) -> list[Chunk]:
    if section.kind == "schedule":
        return _chunk_schedule(section, vocab)

    paras = build_paragraphs(section, vocab)
    if not paras:
        return []
    tops, _style = top_level_indices(paras)

    if tops:
        preamble = paras[: tops[0]]
        units = [(paras[s], paras[s: (tops[k + 1] if k + 1 < len(tops) else len(paras))]) for k, s in enumerate(tops)]
    else:
        preamble = []
        units = [(None, paras)]

    context = " ".join(p.text for p in preamble)
    context_words = context.split()
    if len(context_words) > 120:
        context = " ".join(context_words[:120]) + " ..."

    # Merge consecutive small units so tiny clauses ("(a) ...;") are not isolated,
    # in either direction: 24(a) (nine words) joins 24(b) as "24(a)-(b)".
    merged: list[tuple[list[str], list[Para]]] = []
    for opener, body in units:
        labels = [_label(opener)] if opener else []
        words = sum(p.words for p in body)
        prev_words = sum(p.words for p in merged[-1][1]) if merged else 0
        if merged and (words < MIN_WORDS or prev_words < MIN_WORDS) and prev_words + words <= _merge_limit(words, prev_words):
            merged[-1][0].extend(labels)
            merged[-1][1].extend(body)
        else:
            merged.append((labels, list(body)))

    chunks: list[Chunk] = []
    if tops and preamble and sum(p.words for p in preamble) > 120:
        # Long preamble carries substantive law of its own; index it separately too.
        merged.insert(0, ([], preamble))
        context = ""

    for labels, body in merged:
        if labels:
            path = f"{section.num}({labels[0]})" if len(labels) == 1 else f"{section.num}({labels[0]})-({labels[-1]})"
        else:
            path = section.num
        for piece in _split_large(body):
            text = "\n".join(p.text for p in piece)
            pages = sorted({pg for p in piece for pg in p.pages})
            chunks.append(
                Chunk(
                    id=f"s{section.num}-{len(chunks) + 1}",
                    section=section.num,
                    section_title=section.title,
                    chapter=section.chapter,
                    chapter_title=section.chapter_title,
                    path=path,
                    page_start=pages[0],
                    page_end=pages[-1],
                    text=text,
                    context=context if labels else "",
                    omitted=section.omitted,
                    refs=_find_refs(text + " " + context, section.num),
                    notes=section.notes,
                )
            )
    return chunks


def _schedule_units(section: Section) -> list[tuple[str, str, list[tuple[int, str]]]]:
    """Split a schedule into (rule number, part heading, lines) units.

    Rules start with "1. ...", "2. ..."; a short line ending in "." just before a
    rule is its heading and moves with it; part headings ("A.-Life insurance
    business", "PART B") become the context of the rules under them.
    """
    units: list[tuple[str, str, list[tuple[int, str]]]] = [("", "", [])]
    part = ""
    last_rule = 0
    for page, text in section.lines[1:]:  # line 0 is "THE FIRST SCHEDULE"
        t = text.strip()
        if SCHEDULE_PART.match(t) and len(t) < 120:
            part = t
            units.append(("", part, [(page, text)]))
            continue
        m = SCHEDULE_RULE.match(t)
        n = int(re.match(r"\d+", m.group("n")).group()) if m else 0
        if m and (n == last_rule + 1 or n == 1 or (m.group("n")[-1:].isalpha() and n == last_rule)):
            last_rule = n
            moved: list[tuple[int, str]] = []
            cur = units[-1][2]
            if cur and len(cur[-1][1]) < 100 and cur[-1][1].rstrip().endswith(".") and not cur[-1][1].lstrip().startswith(("(", "[")) \
                    and not SCHEDULE_RULE.match(cur[-1][1].strip()):
                moved = [cur.pop()]
            units.append((m.group("n"), part, moved + [(page, text)]))
        else:
            units[-1][2].append((page, text))
    return [u for u in units if u[2]]


def _chunk_schedule(section: Section, vocab: Counter | None = None) -> list[Chunk]:
    ordinal = section.num.split()[0]
    name = f"{ordinal.title()} Schedule"
    word = RULE_WORD.get(ordinal, "item")

    # Merge small units (lists of one-line items) up to MERGE_TARGET words.
    groups: list[tuple[list[str], str, list[tuple[int, str]]]] = []
    for rule, part, lines in _schedule_units(section):
        words = sum(len(t.split()) for _, t in lines)
        prev_words = sum(len(t.split()) for _, t in groups[-1][2]) if groups else 0
        # Same part, or the previous unit is only a heading ("A.-Life insurance business").
        compatible = groups and (groups[-1][1] == part or not groups[-1][0])
        if compatible and (words < MIN_WORDS or prev_words < MIN_WORDS) \
                and prev_words + words <= _merge_limit(words, prev_words):
            groups[-1][0].extend([rule] if rule else [])
            groups[-1][2].extend(lines)
            if part and not groups[-1][1]:
                groups[-1] = (groups[-1][0], part, groups[-1][2])
        else:
            groups.append(([rule] if rule else [], part, list(lines)))

    chunks: list[Chunk] = []
    for rules, part, lines in groups:
        if not rules:
            path = name
        elif len(rules) == 1:
            path = f"{name}, {word} {rules[0]}"
        else:
            path = f"{name}, {word}s {rules[0]}-{rules[-1]}"
        text = lines[0][1]
        for _, t in lines[1:]:
            text = join_lines(text, t, vocab) if text.endswith("-") else f"{text}\n{t}"
        pages = [p for p, _ in lines]
        for piece in _window_split(text):
            chunks.append(
                Chunk(
                    id=f"{ordinal.lower()}-schedule-{len(chunks) + 1}",
                    section=section.num,
                    section_title=section.title,
                    chapter="",
                    chapter_title=part,
                    path=path,
                    page_start=min(pages),
                    page_end=max(pages),
                    text=piece,
                    context=section.title,
                    kind="schedule",
                    refs=_find_refs(piece, ""),
                    notes=section.notes,
                )
            )
    return chunks


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    vocab = build_vocabulary(sections)
    out: list[Chunk] = []
    for s in sections:
        out.extend(chunk_section(s, vocab))
    return out
