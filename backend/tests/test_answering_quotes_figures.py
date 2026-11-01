"""Quote matching robustness and the deterministic figures check."""

from taxqa.answering.figures import statement_figures, text_values, unverified_figures
from taxqa.answering.schemas import Citation, Point
from taxqa.answering.verifier import quote_score, verify_citations
from taxqa.core.models import Chunk
from taxqa.retrieval.retriever import Hit

TEXT = ("(b) where the property has been acquired, constructed, repaired, renewed or recon- structed with borrowed "
        "capital, the amount of any interest payable on such capital:\nProvided that in the case of a property "
        "referred to in sub-section (2) of section 23, the amount of deduction shall not exceed two lakh rupees.")
CONTEXT = "24. Income chargeable under the head \"Income from house property\" shall be computed after making the following deductions, namely:-"


def test_quote_tolerates_pdf_hyphenation_and_punctuation():
    assert quote_score("acquired, constructed, repaired, renewed or reconstructed with borrowed capital", TEXT) >= 88


def test_quote_may_span_preamble_and_text():
    assert quote_score("shall be computed after making the following deductions namely where the property has been acquired",
                       f"{CONTEXT} {TEXT}") >= 88


def test_ellipsis_pieces_each_checked():
    ok = "where the property has been acquired ... the amount of deduction shall not exceed two lakh rupees"
    bad = "where the property has been acquired ... the deduction is unlimited for every taxpayer in India"
    assert quote_score(ok, TEXT) >= 88
    assert quote_score(bad, TEXT) < 88


def test_short_quotes_must_be_exact_or_are_rejected():
    assert quote_score("two lakh rupees", TEXT) == 100
    assert quote_score("three lakh rupees", TEXT) == 0
    assert quote_score("interest", TEXT) == 0      # too short to prove anything


def test_quote_from_other_chunk_of_same_section_is_found():
    first = Chunk(id="s24-1", section="24", section_title="", chapter="", chapter_title="", path="24(a)",
                  page_start=1, page_end=1, text="(a) a sum equal to thirty per cent of the annual value;")
    second = Chunk(id="s24-2", section="24", section_title="", chapter="", chapter_title="", path="24(b)",
                   page_start=2, page_end=2, text=TEXT)
    sources = {"S1": Hit(first, 0.9), "S2": Hit(second, 0.8)}
    p = Point(statement="Interest on borrowed capital is deductible.",
              citations=[Citation(source_id="S1", provision="Section 24(b)", quote="the amount of any interest payable on such capital")])
    verify_citations([p], sources, "Act")
    c = p.citations[0]
    assert c.chunk_id == "s24-2" and c.valid and c.page_start == 2


def test_word_and_digit_figures_are_equivalent():
    assert {30.0, 150000.0, 240.0} <= text_values(
        "thirty per cent; one lakh fifty thousand rupees; two hundred and forty days")
    assert unverified_figures("A deduction of 30% is allowed, up to Rs. 1,50,000.",
                              "thirty per cent ... one lakh fifty thousand rupees") == []
    assert unverified_figures("Interest up to ₹1.5 lakh is deductible.", "one lakh fifty thousand rupees") == []


def test_wrong_figure_is_reported():
    assert unverified_figures("The deduction is capped at Rs. 3,00,000.", TEXT) == ["3,00,000"]
    assert unverified_figures("A 50% deduction applies.", "thirty per cent of the annual value") == ["50%"]


def test_section_numbers_years_and_dates_are_not_figures():
    s = "Under section 80C(2) and Explanation 2, for AY 2025-26, from 1 April 2026, the 2nd proviso applies."
    assert statement_figures(s) == []


def test_figure_check_runs_inside_verifier():
    chunk = Chunk(id="s24-1", section="24", section_title="", chapter="", chapter_title="", path="24(b)",
                  page_start=1, page_end=1, text=TEXT)
    p = Point(statement="For a self-occupied house the interest deduction is limited to Rs. 3 lakh.",
              citations=[Citation(source_id="S1", provision="Section 24(b), proviso",
                                  quote="the amount of deduction shall not exceed two lakh rupees")])
    verify_citations([p], {"S1": Hit(chunk, 0.9)}, "Act")
    assert p.citations[0].valid
    assert p.unverified_figures == ["3 lakh"]
    assert any("figure" in f for f in p.flags)
