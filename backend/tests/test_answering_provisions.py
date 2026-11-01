"""Citation-label parsing and pinpoint checks for provisos, Explanations, schedules and hyphenated sections."""

from taxqa.answering.provisions import canon, mentioned_sections, parse_provision
from taxqa.answering.schemas import Citation, Point
from taxqa.answering.verifier import verify_citations
from taxqa.core.models import Chunk
from taxqa.retrieval.retriever import Hit


def _chunk(cid, section, path, text, context="", kind="section"):
    return Chunk(id=cid, section=section, section_title="", chapter="", chapter_title="", path=path,
                 page_start=5, page_end=5, text=text, context=context, kind=kind)


def test_parse_basic_and_nested():
    p = parse_provision("Section 80C(2)(xviii)")
    assert (p.section, p.subs) == ("80C", ["2", "xviii"])
    p = parse_provision("section 24 (b)")
    assert (p.section, p.subs) == ("24", ["b"])
    p = parse_provision("Section 2(14)(iii)")
    assert (p.section, p.subs) == ("2", ["14", "iii"])


def test_parse_named_subdivision_before_section():
    p = parse_provision("clause (b) of section 24")
    assert (p.section, p.subs) == ("24", ["b"])
    p = parse_provision("sub-section (2) of section 23")
    assert (p.section, p.subs) == ("23", ["2"])


def test_parse_provisos_and_explanations():
    p = parse_provision("Section 24(b), first proviso")
    assert (p.section, p.subs, p.proviso) == ("24", ["b"], 1)
    p = parse_provision("second proviso to section 23(1)")
    assert (p.section, p.subs, p.proviso) == ("23", ["1"], 2)
    p = parse_provision("Proviso to section 24(b)")
    assert p.proviso == 0
    p = parse_provision("Explanation 1 to section 23")
    assert (p.section, p.explanation) == ("23", "1")
    p = parse_provision("Section 2(22), Explanation")
    assert (p.section, p.subs, p.explanation) == ("2", ["22"], "")


def test_parse_schedule_and_part():
    p = parse_provision("First Schedule, Part A")
    assert (p.section, p.schedule, p.part) == (None, "FIRST SCHEDULE", "A")
    p = parse_provision("Rule 2 in Part B of the Second Schedule")
    assert (p.schedule, p.part) == ("SECOND SCHEDULE", "B")


def test_parse_hyphenated_sections():
    assert parse_provision("Section 80-IA(4)").section == "80-IA"
    assert parse_provision("u/s 115-O").section == "115-O"
    assert parse_provision("sec. 80-JJAA").section == "80-JJAA"
    assert canon("80-ia") == canon("80IA") == "80IA"


def test_parse_multiple_sections_keeps_first():
    p = parse_provision("Sections 23 and 24")
    assert p.section == "23" and p.extra_sections == ["24"]


def test_mentioned_sections():
    text = "Under section 24(b) and sections 23, 25 and 80-IA, but not section 2 of the Finance Act, 2024."
    assert mentioned_sections(text) == ["24", "23", "25", "80-IA"]
    assert mentioned_sections("Interest is deductible, subject to 2 conditions.") == []


C24 = _chunk(
    "s24-1", "24", "24(a)-(b)",
    "(a) a sum equal to thirty per cent of the annual value;\n(b) where the property has been acquired with borrowed "
    "capital, the amount of any interest payable on such capital:\nProvided that in the case of a property referred to "
    "in sub-section (2) of section 23, the amount of deduction shall not exceed thirty thousand rupees.\n"
    "Explanation.-Where the property has been acquired with borrowed capital, the interest shall be deducted.",
)
CIA = _chunk("s80ia-1", "80-IA", "80-IA(4)", "(4) This section applies to any enterprise carrying on the business of developing infrastructure facility.")
SCHED = _chunk("first-schedule-1", "FIRST SCHEDULE", "First Schedule",
               "PART A Life insurance business. Rule 1. The profits and gains of life insurance business shall be taken to be the annual average.",
               kind="schedule")
SOURCES = {"S1": Hit(C24, 0.9), "S2": Hit(CIA, 0.8), "S3": Hit(SCHED, 0.7)}


def _verify(provision, quote, sid="S1"):
    p = Point(statement="statement", citations=[Citation(source_id=sid, provision=provision, quote=quote)])
    verify_citations([p], SOURCES, "Act")
    return p.citations[0]


def test_proviso_citation_verifies():
    c = _verify("Section 24(b), first proviso", "the amount of deduction shall not exceed thirty thousand rupees")
    assert c.valid, c.note


def test_second_proviso_not_present_fails_pinpoint():
    c = _verify("Section 24(b), second proviso", "the amount of deduction shall not exceed thirty thousand rupees")
    assert c.found_in_sources and c.quote_ok and not c.pinpoint_ok


def test_explanation_citation_verifies_and_numbered_one_fails():
    assert _verify("Explanation to section 24", "where the property has been acquired with borrowed capital, the interest shall be deducted").valid
    c = _verify("Section 24, Explanation 2", "where the property has been acquired with borrowed capital, the interest shall be deducted")
    assert not c.pinpoint_ok


def test_hyphenated_section_matches_whatever_spelling():
    c = _verify("Section 80IA(4)", "applies to any enterprise carrying on the business of developing infrastructure facility", sid="S2")
    assert c.valid and c.section == "80-IA"


def test_schedule_part_citation():
    c = _verify("First Schedule, Part A", "the profits and gains of life insurance business shall be taken", sid="S3")
    assert c.valid, c.note
    c = _verify("First Schedule, Part C", "the profits and gains of life insurance business shall be taken", sid="S3")
    assert c.found_in_sources and not c.pinpoint_ok


def test_section_named_but_quote_from_other_excerpt_is_explained():
    c = _verify("Section 23", "a sum equal to thirty per cent of the annual value")
    assert not c.found_in_sources
    assert "quote comes from Section 24" in c.note
