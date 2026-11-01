from taxqa.answering.generator import parse_summary


def test_inline_point_links_are_extracted_and_removed():
    draft = {"summary": "You can claim 30% of the annual value (points 1, 2, 3). "
                        "Interest on the loan is deductible (point 4). "
                        "Further deductions may apply under Section 80EE (see points 6 and 7)."}
    items = parse_summary(draft)
    assert [i.points for i in items] == [[1, 2, 3], [4], [6, 7]]
    assert items[0].text == "You can claim 30% of the annual value."
    assert all("point" not in i.text.lower() for i in items)


def test_structured_summary_keeps_its_links():
    items = parse_summary({"summary": [{"text": "Interest is deductible (point 2).", "points": [2]}]})
    assert items[0].points == [2] and items[0].text == "Interest is deductible."
