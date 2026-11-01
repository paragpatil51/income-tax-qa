"""Regressions found while checking real answers on the FA 2026 text."""

from taxqa.answering.provisions import mentioned_sections, parse_provision
from taxqa.answering.verifier import pinpoint_missing
from taxqa.core.models import Chunk
from taxqa.retrieval.retriever import Hit


def _hit(path: str, text: str) -> Hit:
    return Hit(Chunk(id="c", section="80C", section_title="", chapter="VI-A", chapter_title="", path=path,
                     page_start=400, page_end=400, text=text), 0.9)


def test_subsection_inside_merged_range_counts_as_present():
    # A window of the merged unit 80C(3)-(7) that does not print "(4)" itself.
    hit = _hit("80C(3)-(7)", "(i) in the case of an individual, the individual, the wife or husband and any child")
    assert pinpoint_missing(parse_provision("Section 80C(4)(i)"), hit) == []
    assert pinpoint_missing(parse_provision("Section 80C(9)(i)"), hit) == ["(9)"]


def test_sections_of_other_acts_are_not_income_tax_sections():
    text = ("a banking company to which the Banking Regulation Act, 1949 applies, or any bank referred to in "
            "section 51 of that Act; see also section 2 of the said Act and section 80E")
    assert mentioned_sections(text) == ["80E"]
