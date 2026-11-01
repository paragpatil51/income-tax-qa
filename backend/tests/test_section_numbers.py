import re

import pytest

from taxqa.core.section_numbers import SECTION_NUM, normalize_section, section_key, try_normalize_section


@pytest.mark.parametrize(
    "raw, canon",
    [
        ("24", "24"), ("80c", "80C"), ("115BAC", "115BAC"), ("10AA", "10AA"),
        ("80IA", "80-IA"), ("80-ia", "80-IA"), ("80 - IAB", "80-IAB"), ("80‑IB", "80-IB"),
        ("115O", "115-O"), ("245-OA", "245-OA"), ("194IA", "194-IA"),
        ("80-JJA", "80JJA"), ("80-LA", "80LA"), ("80QQB", "80QQB"), ("024", "24"),
    ],
)
def test_normalize(raw, canon):
    assert normalize_section(raw) == canon


def test_invalid():
    assert try_normalize_section("section") is None
    assert try_normalize_section("1234") is None
    with pytest.raises(ValueError):
        normalize_section("80C(2)")


def test_ordering_ignores_hyphen():
    expected = ["80HHF", "80-I", "80-IA", "80-IAB", "80-IB", "80J", "80JJA", "80JJAA", "80LA", "80-O", "80P"]
    shuffled = ["80JJAA", "80-IB", "80P", "80-I", "80LA", "80J", "80-IAB", "80HHF", "80-O", "80JJA", "80-IA"]
    assert sorted(shuffled, key=section_key) == expected
    assert sorted(["11", "10B", "10", "10AA", "10A"], key=section_key) == ["10", "10A", "10AA", "10B", "11"]


def test_fragment_matches_hyphenated_and_plain():
    pat = re.compile(rf"\b({SECTION_NUM})\.")
    assert pat.match("80-IA. (1) Where").group(1) == "80-IA"
    assert pat.match("115BAC. (1)").group(1) == "115BAC"
    assert pat.match("24. Income").group(1) == "24"
    assert re.findall(rf"\b{SECTION_NUM}\b", "the 10th day and 1st April") == []
