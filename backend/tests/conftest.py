"""Builds a small synthetic PDF laid out like the CBDT consolidated Act:
arrangement of sections, chapter headings, bold section headings, running
headers, page numbers, a footnote, a decoy line starting with a number and a
schedule."""

from __future__ import annotations

import pytest

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

R, B = "helv", "hebo"  # Helvetica regular / bold

PAGES = [
    [
        (B, 11, "ARRANGEMENT OF SECTIONS"),
        (R, 10, "CHAPTER I"),
        (R, 10, "PRELIMINARY"),
        (R, 10, "1. Short title, extent and commencement."),
        (R, 10, "2. Definitions."),
        (R, 10, "CHAPTER IV"),
        (R, 10, "COMPUTATION OF TOTAL INCOME"),
        (R, 10, "22. Income from house property."),
        (R, 10, "23. Annual value how determined."),
        (R, 10, "24. Deductions from income from house property."),
        (R, 10, "CHAPTER VI-A"),
        (R, 10, "DEDUCTIONS TO BE MADE IN COMPUTING TOTAL INCOME"),
        (R, 10, "80C. Deduction in respect of life insurance premia, contributions to provident"),
        (R, 10, "fund, etc."),
        (R, 10, "THE FIRST SCHEDULE"),
    ],
    [
        (R, 10, "CHAPTER I"),
        (R, 10, "PRELIMINARY"),
        (B, 10, "Short title, extent and commencement."),
        (R, 10, "1. (1) This Act may be called the Income-tax Act, 1961."),
        (R, 10, "(2) It extends to the whole of India."),
        (B, 10, "Definitions."),
        (R, 10, "2. In this Act, unless the context otherwise requires,-"),
        (R, 10, "(1) \"advance tax\" means the advance tax payable in accordance with the"),
        (R, 10, "provisions of Chapter XVII-C;"),
        (R, 10, "(14) \"capital asset\" means property of any kind held by an assessee, whether"),
        (R, 10, "or not connected with his business or profession;"),
    ],
    [
        (R, 10, "CHAPTER IV"),
        (R, 10, "COMPUTATION OF TOTAL INCOME"),
        (B, 10, "Income from house property."),
        (R, 10, "22. The annual value of property consisting of any buildings or lands appurtenant"),
        (R, 10, "thereto of which the assessee is the owner shall be chargeable to income-tax under"),
        (R, 10, "the head \"Income from house property\"."),
        (B, 10, "Annual value how determined."),
        (R, 10, "23. (1) For the purposes of section 22, the annual value of any property shall be"),
        (R, 10, "deemed to be-"),
        (R, 10, "(a) the sum for which the property might reasonably be expected to let from year"),
        (R, 10, "to year; or"),
        (R, 10, "(b) where the property is let and the actual rent received or receivable by the"),
        (R, 10, "owner is in excess of the sum referred to in clause (a), the amount so received;"),
        (R, 10, "(2) Where the property consists of a house in the occupation of the owner for the"),
        (R, 10, "purposes of his own residence, the annual value shall be taken to be nil. This is"),
        (R, 10, "read with the provisions of section"),
        (R, 10, "25. which is a decoy line produced by line wrapping and must not start a section."),
    ],
    [
        (B, 10, "Deductions from income from house property."),
        (R, 10, "24. Income chargeable under the head \"Income from house property\" shall be"),
        (R, 10, "computed after making the following deductions, namely:-"),
        (R, 10, "(a) a sum equal to thirty per cent of the annual value;"),
        (R, 10, "(b) where the property has been acquired, constructed, repaired, renewed or"),
        (R, 10, "reconstructed with borrowed capital, the amount of any interest payable on such"),
        (R, 10, "capital:"),
        (R, 10, "Provided that in the case of a property referred to in sub-section (2) of"),
        (R, 10, "section 23, the amount of deduction shall not exceed thirty thousand rupees."),
    ],
    [
        (R, 10, "CHAPTER VI-A"),
        (R, 10, "DEDUCTIONS TO BE MADE IN COMPUTING TOTAL INCOME"),
        (B, 10, "Deduction in respect of life insurance premia, contributions to provident fund, etc."),
        (R, 10, "80C. (1) In computing the total income of an assessee, being an individual or a Hindu"),
        (R, 10, "undivided family, there shall be deducted the whole of the amount paid or deposited"),
        (R, 10, "in the previous year, as does not exceed one lakh fifty thousand rupees."),
        (R, 10, "(2) The sums referred to in sub-section (1) shall be any sums paid or deposited as-"),
        (R, 10, "(i) to effect or to keep in force an insurance on the life of persons specified;"),
        (R, 10, "(iv) as a contribution to any provident fund to which the Provident Funds Act,"),
        (R, 10, "1925 applies;"),
    ],
    [
        (R, 10, "THE FIRST SCHEDULE"),
        (R, 10, "INSURANCE BUSINESS"),
        (R, 10, "Rule 1. The profits and gains of any business of insurance shall be computed"),
        (R, 10, "separately from the profits and gains of any other business."),
    ],
]


def build_pdf(path) -> None:
    doc = fitz.open()
    for n, lines in enumerate(PAGES, start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((200, 30), "THE INCOME-TAX ACT, 1961", fontname=R, fontsize=9)
        y = 80
        for font, size, text in lines:
            page.insert_text((60, y), text, fontname=font, fontsize=size)
            y += 16
        if n == 3:
            page.insert_text((60, 790), "1. Substituted by the Finance Act, 2001, w.e.f. 1-4-2002.", fontname=R, fontsize=7)
        page.insert_text((290, 820), str(n), fontname=R, fontsize=9)
    doc.save(path)


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "act.pdf"
    build_pdf(path)
    return path


# ---------------------------------------------------------------------------
# CBDT layout, as observed in the official consolidated PDF (printed from HTML):
# no table of contents; bold heading line, then a bold "N." followed by regular
# text; italic clause markers with no space after them; amendment markers as
# small digits ("12[(2) ..."); footnote blocks right after a section, numbered
# like sections but indented ~3pt and quoting old text; bold-italic omitted
# sections; centred chapter / part headings ("CHAPTER VIA", "Chapter XII-D",
# "C.-Income from house property"); hyphenated numbers (80-IA, 115-O); a
# numbered rate table inside a section; schedules with bold numbered rules.
# Each line is (x, [(font, size, text), ...]); spans are placed left to right.
# ---------------------------------------------------------------------------
TR, TB, TI, TBI = "tiro", "tibo", "tiit", "tibi"  # Times roman / bold / italic / bold-italic
C = None  # centre the line


def _l(x, *spans):
    return (x, list(spans), False)


def _same_line(x, *spans):
    """A separate text fragment on the previous line's baseline (e.g. "(50)" ... "any income")."""
    return (x, list(spans), True)


def _s(text, font=TR, size=12):
    return (font, size, text)


CBDT_PAGES = [
    [
        _l(C, _s("INCOME-TAX ACT, 1961", TB)),
        _l(C, _s("CHAPTER I", TBI)),
        _l(C, _s("PRELIMINARY", TB)),
        _l(45, _s("Short title, extent and commencement.", TB)),
        _l(45, _s("1. (1) This Act may be called the Income-tax Act, 1961.", TB)),
        _l(45, _s("(2) It extends to the whole of India.")),
        _l(48, _s("*Amendments made by Act No. 7 of 2025 have been printed in italics.", TBI)),
        _l(45, _s("Definitions.", TB)),
        _l(45, _s("2.", TB), _s(" In this Act, unless the context otherwise requires,-")),
        _l(67, _s("(1)", TI), _s('"advance tax" means the advance tax payable under Chapter XVII-C;')),
        _l(60, _s("(1A)", TI), _s('"agricultural income" means any rent derived from land used for')),
        _l(90, _s("agricultural purposes;")),
        _l(102, _s("(2)", TI)),
        _same_line(131, _s('"annual value" means its annual value as determined under section 23;')),
    ],
    [
        _l(C, _s("CHAPTER IV", TBI)),
        _l(C, _s("COMPUTATION OF TOTAL INCOME", TB)),
        _l(C, _s("C.-Income from house property", TI)),  # the PDF has an em dash; base-14 fonts lack it
        _l(45, _s("Annual value how determined.", TB)),
        _l(45, _s("23.", TB), _s(" (1) For the purposes of section 22, the annual value of any property shall be deemed to be-")),
        _l(67, _s("(a)", TI), _s("the sum for which the property might reasonably be expected to let;")),
        _l(45, _s("12", TBI, 9), _s("[(2) The annual value of a house occupied by the owner for his own residence shall be nil.]", TBI)),
        _l(45, _s("(3) This is read with the provisions of sub-section")),
        _l(45, _s("(2) of section 22 and is not a new sub-section.")),
        _l(48, _s("12. Sub. by Act No. 7 of 2025, w.e.f. 1-4-2025. Prior to its substitution, sub-section (2) read as under :", TBI)),
        _l(65, _s('"(2) Where the property is in the occupation of the owner, the annual value shall be nil.')),
        _l(67, _s("(a)", TI), _s('quoted old clause that must not appear in the statute text;"')),
        _l(48, _s("13. Ins. by Act No. 08 of 2023, w.e.f. 1-4-2024.")),
        _l(45, _s("Deductions from income from house property.", TB)),
        _l(45, _s("24.", TB), _s(' Income chargeable under the head "Income from house property" shall be computed after')),
        _l(45, _s("making the following deductions, namely:-")),
        _l(67, _s("(a)", TI), _s("a sum equal to thirty per cent of the annual value;")),
        _l(67, _s("(b)", TI), _s("where the property has been acquired or constructed with borrowed capital, the amount")),
        _l(90, _s("of any interest payable on such capital, as referred to in the provisions of section")),
        _l(45, _s("25. which is a decoy produced by line wrapping and must not start a section.")),
        _l(90, _s("Provided that", TB), _s(" the amount of deduction shall not exceed two lakh rupees.")),
    ],
    [
        _l(45, _s("Export markets development allowance.", TB)),
        _l(45, _s("35B.", TBI), _s(" [Omitted by the Direct Tax Laws (Amendment) Act, 1987, w.e.f. 1-4-1989.]", TI)),
        _l(C, _s("CHAPTER VIA", TBI)),
        _l(C, _s("DEDUCTIONS TO BE MADE IN COMPUTING TOTAL INCOME", TB)),
        _l(C, _s("B.-Deductions in respect of certain payments", TI)),
        _l(45, _s("Deduction in respect of life insurance premia, contributions to provident fund,", TB)),
        _l(45, _s("etc.", TB)),
        _l(45, _s("80C.", TB), _s(" (1) In computing the total income of an assessee there shall be deducted the sums referred")),
        _l(45, _s("to in sub-section (2), as does not exceed one lakh fifty thousand rupees.")),
        _l(45, _s("(2) The sums referred to in sub-section (1) shall be sums paid to keep in force an insurance on life;")),
        _l(45, _s("Deductions in respect of profits and gains from industrial undertakings.", TB)),
        _l(45, _s("80-IA.", TB), _s(" (1) Where the gross total income includes profits derived from an eligible business, a")),
        _l(45, _s("deduction shall be allowed, read with section 80-IB and sections 80C and 115-O.")),
        _l(45, _s("(2) The deduction may be claimed for any ten consecutive assessment years.")),
        _l(45, _s("Deductions in case of undertakings in certain special category States.", TB)),
        _l(45, _s("80-IB.", TB), _s(" (1) Where the gross total income of an assessee includes profits, a deduction is allowed.")),
    ],
    [
        _l(C, _s("CHAPTER XII", TBI)),
        _l(C, _s("DETERMINATION OF TAX IN CERTAIN SPECIAL CASES", TB)),
        _l(45, _s("Tax on income of individuals [, Hindu undivided family and others].", TB)),
        _l(45, _s("115BAC.", TB), _s(" (1) The income-tax payable shall be computed at the rates in the following Table:")),
        _l(123, _s("1. Upto Rs. 3,00,000 Nil")),
        _l(123, _s("2. From Rs. 3,00,001 to Rs. 7,00,000 5 per cent")),
        _l(45, _s("(2) The option shall be exercised in the prescribed manner.")),
        _l(C, _s("Chapter XII-D", TBI)),
        _l(C, _s("Special provisions relating to tax on distributed profits", TB)),
        _l(45, _s("Tax on distributed profits of domestic companies.", TB)),
        _l(45, _s("115-O.", TB), _s(" (1) In addition to the income-tax chargeable, any amount declared as dividend is charged.")),
    ],
    [
        _l(C, _s("THE FIRST SCHEDULE", TBI)),
        _l(C, _s("INSURANCE BUSINESS", TB)),
        _l(C, _s("[See section 44]", TI)),
        _l(C, _s("A.-Life insurance business", TI)),
        _l(45, _s("Profits of life insurance business to be computed separately.", TB)),
        _l(45, _s("1.", TB), _s(" In the case of a person who carries on life insurance business, profits are computed separately.")),
        _l(45, _s("Computation of profits of life insurance business.", TB)),
        _l(45, _s("2.", TB), _s(" The profits and gains of life insurance business shall be the annual average of the surplus.")),
    ],
]


def build_cbdt_pdf(path) -> None:
    doc = fitz.open()
    for lines in CBDT_PAGES:
        page = doc.new_page(width=612, height=769)
        y = 23
        for x, spans, same in lines:
            y += 0 if same else 17
            width = sum(fitz.get_text_length(t, fontname=f, fontsize=s) for f, s, t in spans) + 3 * (len(spans) - 1)
            cx = (612 - width) / 2 if x is None else x
            for n, (font, size, text) in enumerate(spans):
                # Markers such as "(a)" are followed by a small visual gap but no space character.
                page.insert_text((cx, y), text, fontname=font, fontsize=size)
                cx += fitz.get_text_length(text, fontname=font, fontsize=size) + (3 if n + 1 < len(spans) else 0)
    doc.save(path)


@pytest.fixture(scope="session")
def cbdt_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("cbdt") / "act_cbdt.pdf"
    build_cbdt_pdf(path)
    return path
