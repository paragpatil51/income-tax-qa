from collections import Counter

import numpy as np

from taxqa.ingestion.chunker import join_lines
from taxqa.ingestion.type3_decoder import Type3Font, _bitmap_metrics, _Glyph, clean_type3_text


def _glyph_stream(bitmap: np.ndarray) -> bytes:
    """A Type3 glyph procedure with an uncompressed inline 1-bit image mask (0 = paint)."""
    h, w = bitmap.shape
    packed = np.packbits(np.where(bitmap, 0, 1).astype(np.uint8), axis=1).tobytes()
    head = f"50 0 0 0 {w} {h} d1\nq\n{w} 0 0 {h} 0 0 cm\nBI\n/W {w}\n/H {h}\n/BPC 1\n/IM true\nID\n".encode()
    return head + packed + b"\nEI\nQ\n"


def test_bitmap_metrics_ink_and_slant():
    upright = np.zeros((40, 20), dtype=bool)
    upright[:, 8:12] = True                      # vertical stem
    leaning = np.zeros((40, 30), dtype=bool)
    for row in range(40):                        # stem leaning right as it rises
        x = 5 + (39 - row) // 3
        leaning[row, x: x + 4] = True
    ink_u, slant_u = _bitmap_metrics(_glyph_stream(upright))
    ink_l, slant_l = _bitmap_metrics(_glyph_stream(leaning))
    assert abs(ink_u - 0.2) < 1e-6 and abs(slant_u) < 0.01
    assert slant_l > 0.25


def test_bitmap_metrics_rejects_compressed_images():
    assert _bitmap_metrics(b"0 0 d0 BI /W 4 /H 4 /F /AHx ID 00 EI") is None


def test_font_decodes_cp1252_names_and_synthetic_space():
    f = Type3Font(codes={0: _Glyph("d"), 2: _Glyph("—"), 3: _Glyph("e")})
    assert f.decode("\x00\x03\x02 ") == "de— "
    assert f.decode("\x09") == "�"


def test_marker_glued_to_text_gets_a_space():
    assert clean_type3_text("16.The income chargeable") == "16. The income chargeable"
    assert clean_type3_text("(i)[***]") == "(i) [***]"
    assert clean_type3_text("income-tax under the head") == "income-tax under the head"


def test_line_break_hyphens_use_document_vocabulary():
    vocab = Counter({"commissioner": 40, "one-fifth": 6, "onefifth": 0, "government": 12})
    assert join_lines("the Principal Commis-", "sioner shall", vocab) == "the Principal Commissioner shall"
    assert join_lines("a sum equal to one-", "fifth of his salary", vocab) == "a sum equal to one-fifth of his salary"
    assert join_lines("Central", "Government", vocab) == "Central Government"
    # unknown word: keep the hyphen rather than guess
    assert join_lines("an entertain-", "ment allowance", Counter()) == "an entertain-ment allowance"
