"""Core data structures shared by ingestion, retrieval and answering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class Section:
    """One statutory section (or schedule) reconstructed from the PDF."""

    num: str                      # "24", "80C", "115BAC", or "FIRST SCHEDULE"
    title: str
    chapter: str                  # "IV", "VI-A", "" for schedules
    chapter_title: str
    page_start: int               # 1-based PDF page numbers
    page_end: int
    lines: list[tuple[int, str]]  # (page, text) in reading order
    notes: list[str] = field(default_factory=list)  # amendment footnotes on its pages
    kind: str = "section"         # "section" | "schedule"
    omitted: bool = False

    @property
    def text(self) -> str:
        return "\n".join(t for _, t in self.lines)


@dataclass
class Chunk:
    """A retrievable unit: one or more subsections/clauses of a single section."""

    id: str
    section: str
    section_title: str
    chapter: str
    chapter_title: str
    path: str                     # citation label, e.g. "24(b)", "80C(2)", "10(10D)"
    page_start: int
    page_end: int
    text: str                     # statutory text of this unit
    context: str = ""             # section preamble the unit depends on
    kind: str = "section"
    omitted: bool = False
    refs: list[str] = field(default_factory=list)   # sections cross-referenced in text
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.kind == "schedule":
            return self.path or self.section.title()   # e.g. "First Schedule, rule 5"
        return f"Section {self.path}"

    @property
    def header(self) -> str:
        bits = [self.label]
        if self.section_title:
            bits.append(self.section_title)
        if self.chapter:
            bits.append(f"Chapter {self.chapter}" + (f" ({self.chapter_title})" if self.chapter_title else ""))
        return " | ".join(bits)

    @property
    def pages(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    def index_text(self) -> str:
        """Text used for lexical and dense indexing (header + preamble + body)."""
        parts = [self.header]
        if self.context:
            parts.append(self.context)
        parts.append(self.text)
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        # Ignore keys added by newer ingestion versions so older code can still load the index.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
