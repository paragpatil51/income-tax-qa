"""Parsing of statutory references written by the LLM or found in free text.

Handles the forms a citation to the Act actually takes:
  "Section 24(b)"                    "Sec. 80-IA(4)"            "u/s 115-O"
  "Section 2(14)(iii)"               "clause (b) of section 24"
  "Section 24(b), first proviso"     "second proviso to section 23(1)"
  "Explanation 1 to section 23"      "Section 2(22), Explanation 2"
  "First Schedule, Part A"           "Part A of the First Schedule"

Section numbers are compared in canonical form (upper case, no hyphen or
space), so "80-IA", "80IA" and "80-ia" are the same section whatever spelling
the ingestion step stores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Tolerant section-number fragment: 24, 80C, 115BAC, 80-IA, 80-JJAA, 115-O.
SECTION_NUM = r"\d{1,3}(?:-?[A-Za-z]{1,5})?(?![A-Za-z])"

_SECTION_REF = re.compile(
    rf"\b(?P<kw>sections?|secs?\.?|s\.|u/s\.?)\s*(?P<num>{SECTION_NUM})"
    r"(?P<subs>(?:\s*\(\s*[0-9A-Za-z]{1,6}\s*\))*)",
    re.I,
)
_LIST_TAIL = re.compile(
    rf"(?:\s*(?:,|\band\b|\bor\b|\bto\b|&)\s*(?:{SECTION_NUM})(?:\s*\(\s*[0-9A-Za-z]{{1,6}}\s*\))*)*",
    re.I,
)
# Directly after the reference only: "section 2 of the Finance Act, 2024".
_OTHER_ACT = re.compile(
    r"\s*(?:\(\s*[0-9A-Za-z]{1,6}\s*\)\s*)*of (?:"
    r"the (?!Income-?tax Act)[A-Z][\w,-]*(?:\s+[A-Z][\w,-]*)*\s+Act\b"   # "section 2 of the Finance Act, 2024"
    r"|(?:that|the said) Act\b)"                                           # "section 51 of that Act"
)
_NAMED_SUBDIVISION = re.compile(r"\b(?:sub-section|sub-clause|clause|item)\s*\(\s*([0-9A-Za-z]{1,6})\s*\)", re.I)
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "last": -1,
}
_PROVISO = re.compile(r"\b(?:(" + "|".join(_ORDINALS) + r")\s+)?provisos?\b", re.I)
_EXPLANATION = re.compile(r"\bexplanation\s*(\d{1,2}[A-Za-z]?\b|[IVX]{1,4}\b)?", re.I)
_SCHEDULE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth)"
    r"\s+schedule\b",
    re.I,
)
_PART = re.compile(r"\bpart\s+([A-Z]{1,2}|[IVX]{1,4})\b", re.I)


def canon(num: str) -> str:
    """Canonical section key for comparisons: '80-ia' -> '80IA'."""
    return re.sub(r"[\s-]", "", num or "").upper()


@dataclass
class Provision:
    section: str | None = None          # as written, upper-cased ("80-IA")
    subs: list[str] = field(default_factory=list)   # ["14", "iii"]
    schedule: str | None = None         # "FIRST SCHEDULE"
    proviso: int | None = None          # None = not cited, 0 = a proviso, n = nth, -1 = last
    explanation: str | None = None      # None = not cited, "" = unnumbered, "1", "2A", ...
    part: str | None = None             # Schedule part, e.g. "A"
    extra_sections: list[str] = field(default_factory=list)  # "Sections 23 and 24" -> ["24"]

    @property
    def target(self) -> str | None:
        """Key used to find the cited text among retrieved chunks."""
        if self.section:
            return canon(self.section)
        return canon(self.schedule) if self.schedule else None

    @property
    def label(self) -> str:
        if self.section:
            out = f"Section {self.section}" + "".join(f"({s})" for s in self.subs)
        elif self.schedule:
            out = self.schedule.title() + (f", Part {self.part}" if self.part else "")
        else:
            return ""
        if self.proviso is not None:
            names = {v: k for k, v in _ORDINALS.items()}
            out += f", {names.get(self.proviso, '')} proviso".replace(",  ", ", ")
        if self.explanation is not None:
            out += f", Explanation {self.explanation}".rstrip()
        return out


def parse_provision(text: str) -> Provision:
    text = text or ""
    prov = Provision()
    m = _SECTION_REF.search(text)
    if m:
        prov.section = m.group("num").upper()
        prov.subs = re.findall(r"\(\s*([0-9A-Za-z]{1,6})\s*\)", m.group("subs") or "")
        if m.group("kw").lower().startswith("sections"):
            tail = _LIST_TAIL.match(text, m.end())
            if tail and tail.group(0):
                cleaned = re.sub(r"\(\s*[0-9A-Za-z]{1,6}\s*\)", " ", tail.group(0))
                prov.extra_sections = [n.upper() for n in re.findall(SECTION_NUM, cleaned)]
        # "clause (b) of section 24": named sub-divisions written before the section
        before = text[: m.start()]
        named = _NAMED_SUBDIVISION.findall(before)
        prov.subs = list(dict.fromkeys(named + prov.subs))
    else:
        sm = _SCHEDULE.search(text)
        if sm:
            prov.schedule = f"{sm.group(1).upper()} SCHEDULE"
            pm = _PART.search(text)
            prov.part = pm.group(1).upper() if pm else None

    pv = _PROVISO.search(text)
    if pv:
        prov.proviso = _ORDINALS.get((pv.group(1) or "").lower(), 0)
    ex = _EXPLANATION.search(text)
    if ex:
        prov.explanation = (ex.group(1) or "").upper()
    return prov


def mentioned_sections(text: str) -> list[str]:
    """Sections of this Act named in free text ("section 24", "sections 23, 24 and 25")."""
    out: list[str] = []
    for m in _SECTION_REF.finditer(text or ""):
        if _OTHER_ACT.match(text, m.end()):
            continue  # "section 2 of the Finance Act, 2024"
        nums = [m.group("num")]
        if m.group("kw").lower().startswith("sections"):
            tail = _LIST_TAIL.match(text, m.end())
            if tail and tail.group(0):
                cleaned = re.sub(r"\(\s*[0-9A-Za-z]{1,6}\s*\)", " ", tail.group(0))
                nums += re.findall(SECTION_NUM, cleaned)
        for n in nums:
            if n.upper() not in out:
                out.append(n.upper())
    return out
