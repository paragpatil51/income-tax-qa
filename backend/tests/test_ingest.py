from taxqa.ingestion.chunker import Para, _find_refs, build_paragraphs, chunk_sections, top_level_indices
from taxqa.ingestion.pdf_reader import load_clean_lines
from taxqa.ingestion.section_parser import parse_sections, section_key


def _parse(pdf):
    lines = load_clean_lines(pdf)
    sections, report = parse_sections(lines)
    return lines, sections, report


def test_headers_page_numbers_removed_and_footnote_flagged(synthetic_pdf):
    lines, _, _ = _parse(synthetic_pdf)
    texts = [l.text for l in lines]
    assert "THE INCOME-TAX ACT, 1961" not in texts
    assert not any(t.strip() == "3" for t in texts)
    foot = [l for l in lines if l.footnote]
    assert len(foot) == 1 and "Finance Act, 2001" in foot[0].text


def test_sections_found_in_order_and_decoy_rejected(synthetic_pdf):
    _, sections, report = _parse(synthetic_pdf)
    nums = [s.num for s in sections if s.kind == "section"]
    assert nums == ["1", "2", "22", "23", "24", "80C"]
    assert "25" not in nums
    assert report["toc_entries"] == 6
    assert report["toc_sections_missing_in_body"] == []


def test_titles_chapters_pages(synthetic_pdf):
    _, sections, _ = _parse(synthetic_pdf)
    by = {s.num: s for s in sections}
    assert by["24"].title == "Deductions from income from house property"
    assert by["24"].chapter == "IV"
    assert by["80C"].chapter == "VI-A"
    assert by["80C"].title.startswith("Deduction in respect of life insurance premia")
    assert (by["24"].page_start, by["24"].page_end) == (4, 4)
    assert by["23"].notes and "Finance Act, 2001" in by["23"].notes[0]
    # heading of section 24 must not leak into the end of section 23
    assert "Deductions from income from house property" not in by["23"].text


def test_schedule_parsed(synthetic_pdf):
    _, sections, _ = _parse(synthetic_pdf)
    sched = [s for s in sections if s.kind == "schedule"]
    assert len(sched) == 1 and sched[0].num == "FIRST SCHEDULE"
    assert sched[0].page_start == 6


def test_chunks_keep_citation_paths_context_and_refs(synthetic_pdf):
    _, sections, _ = _parse(synthetic_pdf)
    chunks = chunk_sections(sections)
    by = {}
    for c in chunks:
        by.setdefault(c.section, []).append(c)
    c24 = by["24"][0]
    assert c24.path == "24(a)-(b)"
    assert "following deductions" in c24.context
    assert "thirty per cent of the annual value" in c24.text
    assert "23" in c24.refs
    assert c24.page_start == 4
    assert by["80C"][0].path.startswith("80C(1)")
    assert by["22"][0].path == "22"
    assert by["FIRST SCHEDULE"][0].kind == "schedule"


def test_section_key_order():
    nums = ["11", "10B", "10", "10AA", "10A", "80CCD", "80C", "80D", "80CCE"]
    assert sorted(nums, key=section_key) == ["10", "10A", "10AA", "10B", "11", "80C", "80CCD", "80CCE", "80D"]


def test_roman_vs_alpha_clauses():
    paras = [Para("24. Preamble")] + [Para(f"({m}) text", clause=m) for m in ["a", "b", "h", "i", "ii", "j"]]
    tops, style = top_level_indices(paras)
    labels = [paras[i].clause for i in tops]
    assert style == "clause"
    assert labels == ["a", "b", "h", "i", "j"]  # "(i)" after "(h)" is clause (i); "(ii)" is a sub-clause


def test_refs():
    text = "subject to sub-section (2) of section 23 and sections 80C, 80D and 80E, but not section 2 of the Finance Act, 2024"
    assert _find_refs(text, "24") == ["23", "80C", "80D", "80E"]


def test_refs_hyphenated():
    text = "deduction under section 80-IA or section 80IB, and sections 80-IC, 80-ID and 115-O"
    assert _find_refs(text, "80-IA") == ["80-IB", "80-IC", "80-ID", "115-O"]


# --- CBDT layout (no ToC, bold section numbers, inline footnote blocks) ---------------

def _cbdt(pdf):
    lines, sections, report = _parse(pdf)
    return lines, {s.num: s for s in sections}, report


def test_cbdt_sections_found_without_toc(cbdt_pdf):
    _, by, report = _cbdt(cbdt_pdf)
    nums = [n for n, s in by.items() if s.kind == "section"]
    assert nums == ["1", "2", "23", "24", "35B", "80C", "80-IA", "80-IB", "115BAC", "115-O"]
    assert report["mode"] == "bold-evidence"
    assert report["toc_entries"] == 0
    assert report["strong_starts_rejected"] == []


def test_cbdt_footnote_blocks_excluded_and_attached(cbdt_pdf):
    _, by, report = _cbdt(cbdt_pdf)
    s23 = by["23"]
    assert "quoted old clause" not in s23.text          # old text quoted inside a footnote
    assert "Sub. by Act No. 7 of 2025" not in s23.text
    assert any("Sub. by Act No. 7 of 2025" in n for n in s23.notes)
    assert any("Ins. by Act No. 08 of 2023" in n for n in s23.notes)   # upright footnote line
    assert "[(2) The annual value of a house" in s23.text                # amendment marker "12" removed
    assert "12[" not in s23.text
    assert report["footnotes"] >= 3                                       # incl. the "*Amendments ..." note


def test_cbdt_headings_chapters_parts(cbdt_pdf):
    _, by, report = _cbdt(cbdt_pdf)
    assert (by["1"].title, by["1"].chapter) == ("Short title, extent and commencement", "I")
    assert by["24"].title == "Deductions from income from house property"
    assert by["24"].chapter == "IV" and "C. Income from house property" in by["24"].chapter_title
    assert by["80C"].title == "Deduction in respect of life insurance premia, contributions to provident fund, etc"
    assert by["80C"].chapter == "VI-A"
    assert by["115BAC"].title == "Tax on income of individuals, Hindu undivided family and others"
    assert by["115-O"].chapter == "XII-D"
    assert by["115-O"].chapter_title.startswith("Special Provisions Relating to Tax")
    assert by["35B"].omitted and by["35B"].title == "Export markets development allowance"
    assert "Deductions from income from house property" not in by["23"].text


def test_cbdt_text_fidelity(cbdt_pdf):
    _, by, _ = _cbdt(cbdt_pdf)
    assert '(1) "advance tax" means' in by["2"].text                      # space restored after italic marker
    assert '(2) "annual value" means' in by["2"].text                     # same-baseline fragments joined
    assert "Upto Rs. 3,00,000 Nil" in by["115BAC"].text                   # numbered table rows stay statute text
    assert "decoy" in by["24"].text                                       # wrapped numbered line stays in section 24


def test_cbdt_chunks(cbdt_pdf):
    _, by, _ = _cbdt(cbdt_pdf)
    chunks = chunk_sections(list(by.values()))
    paths = {}
    for c in chunks:
        paths.setdefault(c.section, []).append(c)
    c24 = paths["24"][0]
    assert c24.path == "24(a)-(b)" and "(a) a sum equal to thirty per cent" in c24.text
    assert {"80-IB", "80C", "115-O"} <= set(paths["80-IA"][0].refs)
    assert paths["80-IA"][0].id.startswith("s80-IA-")
    assert paths["80-IA"][0].path.startswith("80-IA(1)")
    sched = paths["FIRST SCHEDULE"]
    assert sched[0].path.startswith("First Schedule, rule")
    assert "Profits of life insurance business" in sched[0].text          # rule heading moved with its rule


def test_wrapped_reference_is_not_a_new_subsection(cbdt_pdf):
    _, by, _ = _cbdt(cbdt_pdf)
    subs = [p.sub for p in build_paragraphs(by["23"]) if p.sub]
    assert subs == ["1", "2", "3"]                                        # "(2) of section 22" is a wrapped reference
