from taxqa.answering.confidence import compute_confidence
from taxqa.answering.schemas import Citation, Point
from taxqa.answering.verifier import apply_removals, verify_citations
from taxqa.core.models import Chunk
from taxqa.retrieval.query_planner import explicit_sections
from taxqa.retrieval.retriever import Hit


def _chunk(cid, section, path, text, context=""):
    return Chunk(id=cid, section=section, section_title="", chapter="IV", chapter_title="", path=path,
                 page_start=10, page_end=11, text=text, context=context)


C24 = _chunk(
    "s24-1", "24", "24(a)-(b)",
    "(a) a sum equal to thirty per cent of the annual value;\n(b) where the property has been acquired, constructed, "
    "repaired, renewed or reconstructed with borrowed capital, the amount of any interest payable on such capital:",
    context="24. Income chargeable under the head \"Income from house property\" shall be computed after making the following deductions, namely:-",
)
C23 = _chunk("s23-1", "23", "23(1)", "(1) For the purposes of section 22, the annual value of any property shall be deemed to be-")
SOURCES = {"S1": Hit(C24, 0.92, sub_query=0), "S2": Hit(C23, 0.81, sub_query=0)}


def test_valid_citation_passes_all_checks():
    p = Point(statement="A deduction of thirty per cent of annual value is allowed.",
              citations=[Citation(source_id="S1", provision="Section 24(a)", quote="a sum equal to thirty per cent of the annual value")])
    verify_citations([p], SOURCES, "Income-tax Act, 1961")
    c = p.citations[0]
    assert c.found_in_sources and c.pinpoint_ok and c.quote_ok and c.valid
    assert (c.page_start, c.page_end) == (10, 11)
    assert c.source_document == "Income-tax Act, 1961"
    assert p.unverified_figures == []


def test_fabricated_section_is_rejected_and_point_removed():
    p = Point(statement="Section 80EEA gives an extra deduction for interest.",
              citations=[Citation(source_id="S1", provision="Section 80EEA", quote="additional deduction of interest")])
    warnings = verify_citations([p], SOURCES, "Income-tax Act, 1961")
    assert not p.citations[0].found_in_sources
    assert warnings and "80EEA" in warnings[0]
    apply_removals([p])
    assert p.removed


def test_wrong_pinpoint_and_bad_quote_detected():
    p = Point(statement="Interest is deductible.",
              citations=[Citation(source_id="S1", provision="Section 24(c)", quote="interest is fully deductible without any limit")])
    verify_citations([p], SOURCES, "x")
    c = p.citations[0]
    assert c.found_in_sources and not c.pinpoint_ok and not c.quote_ok
    assert "(c)" in c.note


def test_wrong_source_id_is_reattached_by_section():
    p = Point(statement="Annual value is determined under section 23.",
              citations=[Citation(source_id="S1", provision="Section 23(1)", quote="the annual value of any property shall be deemed to be")])
    verify_citations([p], SOURCES, "x")
    c = p.citations[0]
    assert c.chunk_id == "s23-1" and c.valid
    assert "re-attached" in c.note


def test_confidence_reflects_verification():
    good = Point(statement="x", support="supported",
                 citations=[Citation(source_id="S1", provision="Section 24(a)", quote="a sum equal to thirty per cent of the annual value")])
    verify_citations([good], SOURCES, "x")
    high = compute_confidence([good], {"s24-1": SOURCES["S1"], "s23-1": SOURCES["S2"]}, 1)
    bad = Point(statement="y", support="partial",
                citations=[Citation(source_id="S1", provision="Section 24(c)", quote="nothing like this appears in the text")])
    verify_citations([bad], SOURCES, "x")
    low = compute_confidence([good, bad], {"s24-1": SOURCES["S1"]}, 2)
    assert high.score > low.score
    assert high.band == "high"


def test_explicit_sections():
    assert explicit_sections("What does section 80C provide? Also u/s 24(b) and 115BAC") == ["80C", "24", "115BAC"]
    assert explicit_sections("I filed on the 10th, my 1st return in 2024") == []
