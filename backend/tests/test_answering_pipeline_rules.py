"""Judge handling, removals, summary/caveat filtering and confidence edge cases (no network)."""

from taxqa.answering.confidence import UNJUDGED_CAP, compute_confidence
from taxqa.answering.generator import parse_points, parse_summary, split_sentences
from taxqa.answering.schemas import Citation, Point
from taxqa.answering.verifier import apply_removals, finalize_text, judge_support, verify_citations
from taxqa.core.llm_client import LLMError
from taxqa.core.models import Chunk
from taxqa.retrieval.retriever import Hit

C24 = Chunk(
    id="s24-1", section="24", section_title="", chapter="IV", chapter_title="", path="24(a)-(b)",
    page_start=10, page_end=10,
    text="(a) a sum equal to thirty per cent of the annual value;\n(b) where the property has been acquired, constructed, "
         "repaired, renewed or reconstructed with borrowed capital, the amount of any interest payable on such capital:",
)
SOURCES = {"S1": Hit(C24, 0.9, sub_query=0)}
BY_CHUNK = {"s24-1": SOURCES["S1"]}

DRAFT = {
    "status": "answered",
    "points": [
        {"heading": "Standard deduction", "statement": "You can deduct 30% of the annual value.",
         "citations": [{"source_id": "S1", "provision": "Section 24(a)", "quote": "a sum equal to thirty per cent of the annual value"}]},
        {"heading": "Interest", "statement": "Interest on the housing loan is deductible.",
         "citations": [{"source_id": "[S1]", "provision": "Section 24(b)", "quote": "the amount of any interest payable on such capital"}]},
        {"heading": "Extra relief", "statement": "Section 80EEA gives a further deduction of Rs. 1,50,000.",
         "citations": [{"source_id": "S9", "provision": "Section 80EEA", "quote": "an additional deduction of interest"}]},
    ],
    "summary": [
        {"text": "Yes, you can deduct 30% of the annual value.", "points": [1]},
        {"text": "Interest on the loan is also deductible.", "points": [2]},
        {"text": "Section 80EEA gives a further Rs. 1,50,000.", "points": [3]},
    ],
    "caveats": ["Assumes the property is let out for the whole year.", "Section 80EEA may also apply up to Rs. 1,50,000."],
}


class FakeLLM:
    def __init__(self, verdicts=None, fail=False):
        self.verdicts, self.fail = verdicts or [], fail

    def json(self, system, user, light=False, temperature=0.0):
        if self.fail:
            raise LLMError("quota")
        return {"verdicts": self.verdicts}


def _run(llm):
    points = parse_points(DRAFT)
    verify_citations(points, SOURCES, "Act")
    ran = judge_support(points, BY_CHUNK, llm, question="Can I claim both?")
    apply_removals(points, judge_ran=ran)
    return points, ran


def test_parse_points_normalises_ids_and_numbers():
    points = parse_points(DRAFT)
    assert [p.number for p in points] == [1, 2, 3]
    assert points[1].citations[0].source_id == "S1"


def test_removed_claim_cannot_leak_into_summary_or_caveats():
    points, _ = _run(FakeLLM([{"id": 1, "verdict": "supported"}, {"id": 2, "verdict": "supported"}]))
    assert points[2].removed
    final = finalize_text(DRAFT, points, SOURCES)
    assert "80EEA" not in final.summary and "30%" in final.summary
    assert final.summary_removed == ["Section 80EEA gives a further Rs. 1,50,000."]
    assert final.caveats == ["Assumes the property is let out for the whole year."]
    assert any("Caveat withheld" in w for w in final.warnings)


def test_unlinked_summary_sentences_filtered_when_points_removed():
    draft = dict(DRAFT, summary="Yes, you can deduct 30% of the annual value. You also get a further deduction under the new rules.")
    points, _ = _run(FakeLLM([{"id": 1, "verdict": "supported"}, {"id": 2, "verdict": "supported"}]))
    final = finalize_text(draft, points, SOURCES)
    assert final.summary == "Yes, you can deduct 30% of the annual value."


def test_summary_falls_back_to_verified_points_when_all_sentences_withheld():
    draft = dict(DRAFT, summary=[{"text": "Section 80EEA applies.", "points": [3]}])
    points, _ = _run(FakeLLM([{"id": 1, "verdict": "supported"}, {"id": 2, "verdict": "supported"}]))
    final = finalize_text(draft, points, SOURCES)
    assert "Section 24(a)" in final.summary and "80EEA" not in final.summary


def test_judge_unsupported_removes_point_and_figures_cap_support():
    points, ran = _run(FakeLLM([{"id": 1, "verdict": "supported"}, {"id": 2, "verdict": "unsupported", "reason": "no"}]))
    assert ran and points[1].removed and not points[0].removed
    p = Point(number=1, statement="You can deduct 40% of the annual value.",
              citations=[Citation(source_id="S1", provision="Section 24(a)", quote="a sum equal to thirty per cent of the annual value")])
    verify_citations([p], SOURCES, "Act")
    judge_support([p], BY_CHUNK, FakeLLM([{"id": 1, "verdict": "supported"}]))
    assert p.support == "partial" and p.unverified_figures == ["40%"]


def test_judge_unavailable_requires_matching_quote():
    points = parse_points(DRAFT)
    points[1].citations[0].quote = "interest is always fully deductible for everyone"
    verify_citations(points, SOURCES, "Act")
    ran = judge_support(points, BY_CHUNK, FakeLLM(fail=True))
    apply_removals(points, judge_ran=ran)
    assert not ran
    assert [p.removed for p in points] == [False, True, True]


def test_confidence_edge_cases():
    points, _ = _run(FakeLLM([{"id": 1, "verdict": "supported"}, {"id": 2, "verdict": "supported"}]))
    full = compute_confidence(points, BY_CHUNK, 1, judge_ran=True)
    partial = compute_confidence(points, BY_CHUNK, 1, judge_ran=True, not_covered=1)
    unjudged = compute_confidence(points, BY_CHUNK, 1, judge_ran=False)
    assert partial.score < full.score
    assert unjudged.score <= UNJUDGED_CAP
    assert partial.components["coverage"] == 0.5

    for p in points:
        p.removed = True
    assert compute_confidence(points, BY_CHUNK, 1).band == "n/a"


def test_all_citations_invalid_gives_low_confidence():
    p = Point(number=1, statement="x", support="partial",
              citations=[Citation(source_id="S1", provision="Section 24(z)", quote="nothing like this is in the statute at all")])
    verify_citations([p], SOURCES, "Act")
    conf = compute_confidence([p], BY_CHUNK, 1, judge_ran=True)
    assert conf.components["citations"] == 0.0 and conf.band in ("low", "medium")


def test_sentence_splitting_respects_abbreviations():
    assert split_sentences("Under Sec. 24 you may deduct Rs. 30,000. This is clause (b). Done.") == [
        "Under Sec. 24 you may deduct Rs. 30,000.", "This is clause (b).", "Done."]
    assert [i.points for i in parse_summary({"summary": [{"text": "a", "points": ["1", 2]}]})] == [[1, 2]]
