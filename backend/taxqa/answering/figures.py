"""Deterministic check of figures (percentages, amounts, periods) in answer text.

The Act mostly writes numbers in words ("thirty per cent", "one lakh fifty
thousand rupees", "two hundred and forty days") while the LLM writes "30%",
"Rs. 1.5 lakh" or "240 days". Both sides are reduced to numeric values, and
any figure in a statement that does not occur in the cited provisions is
reported as unverified. Wrong numbers are the most damaging kind of legal
hallucination, and an LLM judge alone is not reliable at catching them.

Deliberately not checked (too noisy, or not a legal figure): section and
sub-division numbers, Explanation/Rule/Form/Chapter numbers, calendar years
and assessment years, dates, ordinals, and the values 0 and 1.
"""

from __future__ import annotations

import re

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"thousand": 10**3, "lakh": 10**5, "lakhs": 10**5, "lac": 10**5, "lacs": 10**5, "crore": 10**7, "crores": 10**7}

_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?|[a-z]+")
_MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
_EXCLUDE = [
    re.compile(r"\b(?:sections?|secs?\.?|u/s\.?)\s*\d{1,3}(?:-?[a-z]{1,5})?(?:\s*\(\s*[0-9a-z]{1,6}\s*\))*"
               r"(?:\s*(?:,|and|or|to|&)\s*\d{1,3}(?:-?[a-z]{1,5})?(?:\s*\(\s*[0-9a-z]{1,6}\s*\))*)*"),
    re.compile(r"\(\s*[0-9a-z]{1,6}\s*\)"),                       # (1), (b), (10d)
    re.compile(r"\b(?:explanation|rule|form|chapter|schedule|item|entry|serial)\s*(?:no\.?\s*)?[0-9ivxlc]+[a-z]*\b"),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?" + _MONTHS + r"\b"),
    re.compile(_MONTHS + r"\s+\d{1,2}\b"),
    re.compile(r"\b(?:19|20)\d{2}(?:\s*-\s*\d{2,4})?\b"),          # years, 2025-26
    re.compile(r"\b\d+(?:st|nd|rd|th)\b"),                        # ordinals
    re.compile(r"\bs\d+\b"),                                      # excerpt ids [S1]
]


def _values(text: str) -> list[tuple[float, str]]:
    """All numeric values in text, with the surface string they came from."""
    tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text.lower().replace("-", " "))]
    out: list[tuple[float, str]] = []
    i = 0
    total = current = 0
    start = None
    active = False

    def flush(end_pos: int) -> None:
        nonlocal total, current, active, start
        if active:
            out.append((float(total + current), text[start:end_pos].strip()))
        total = current = 0
        active = False
        start = None

    last_end = 0
    while i < len(tokens):
        tok, s, e = tokens[i]
        if tok[0].isdigit():
            flush(last_end)
            value = float(tok.replace(",", ""))
            end = e
            if i + 1 < len(tokens) and tokens[i + 1][0] in SCALES:
                value *= SCALES[tokens[i + 1][0]]
                end = tokens[i + 1][2]
                i += 1
            pct = re.match(r"\s*%", text[end:])
            if pct:
                end += pct.end()
            out.append((value, text[s:end]))
            last_end = end
        elif tok in UNITS or tok in TENS:
            if not active:
                active, start = True, s
            current += UNITS.get(tok, TENS.get(tok, 0))
            last_end = e
        elif tok == "hundred" and active:
            current = (current or 1) * 100
            last_end = e
        elif tok in SCALES and active:
            total += (current or 1) * SCALES[tok]
            current = 0
            last_end = e
        elif tok == "and" and active and i + 1 < len(tokens) and (tokens[i + 1][0] in UNITS or tokens[i + 1][0] in TENS):
            pass
        else:
            flush(last_end)
        i += 1
    flush(last_end)
    return out


def text_values(text: str) -> set[float]:
    return {v for v, _ in _values(text or "")}


def statement_figures(text: str) -> list[tuple[float, str]]:
    """Legal figures stated in an answer sentence (see module docstring for exclusions)."""
    cleaned = (text or "").lower()
    for pattern in _EXCLUDE:
        cleaned = pattern.sub(" ", cleaned)
    seen: dict[float, str] = {}
    for value, surface in _values(cleaned):
        if value not in (0.0, 1.0) and value not in seen:
            seen[value] = surface
    return list(seen.items())


def unverified_figures(statement: str, evidence: str) -> list[str]:
    """Figures in `statement` whose value does not occur anywhere in `evidence`."""
    available = text_values(evidence)
    return [surface for value, surface in statement_figures(statement)
            if not any(abs(value - a) < 1e-6 for a in available)]
