"""API / pipeline response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class Citation(BaseModel):
    source_id: str                       # "S3" as given to the LLM
    provision: str                       # "Section 24(b)" as written by the LLM
    quote: str = ""
    # Filled in by the verifier
    chunk_id: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_document: str = ""
    found_in_sources: bool = False       # cited section exists among retrieved provisions
    pinpoint_ok: bool = False            # (2), (b), (xviii)... all present in that provision's text
    quote_ok: bool = False               # quote matches the statute text (fuzzy >= threshold)
    quote_score: float = 0.0
    pinpoint: str = ""                   # normalised label, e.g. "Section 24(b), first proviso"
    note: str = ""

    @computed_field
    @property
    def valid(self) -> bool:
        return self.found_in_sources and self.pinpoint_ok and self.quote_ok


class Point(BaseModel):
    number: int = 0                      # 1-based position in the model's list (summary sentences refer to it)
    heading: str = ""
    statement: str
    citations: list[Citation] = Field(default_factory=list)
    support: Literal["supported", "partial", "unsupported", "unchecked"] = "unchecked"
    support_reason: str = ""
    unverified_figures: list[str] = Field(default_factory=list)  # numbers absent from the cited text
    flags: list[str] = Field(default_factory=list)                # verifier findings shown to the user
    removed: bool = False
    removal_reason: str = ""


class SourceView(BaseModel):
    source_id: str
    chunk_id: str
    label: str
    section_title: str
    chapter: str
    pages: str
    page_start: int
    page_end: int
    relevance: float
    via: list[str]
    text: str
    context: str = ""


class Confidence(BaseModel):
    score: int | None                    # 0-100, None when not applicable
    band: Literal["high", "medium", "low", "n/a"]
    components: dict[str, float] = Field(default_factory=dict)
    explanation: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    status: Literal["answered", "partial", "insufficient", "out_of_scope", "error"]
    summary: str = ""
    summary_removed: list[str] = Field(default_factory=list)   # summary sentences withheld by verification
    points: list[Point] = Field(default_factory=list)
    not_covered: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    sources: list[SourceView] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=lambda: Confidence(score=None, band="n/a"))
    warnings: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    source_document: str = ""
    edition: str = ""
    models: dict[str, str] = Field(default_factory=dict)
    timings_ms: dict[str, int] = Field(default_factory=dict)
