"""Section-number syntax of the Income-tax Act, shared by ingestion, retrieval and verification.

A section number is 1-3 digits plus an optional upper-case letter suffix:
24, 10AA, 80C, 115BAC, 271AAD. Where the suffix starts with I or O (letters that
could be misread as 1 or 0) the Act prints a hyphen: 80-I, 80-IA, 80-IAB, 115-O,
194-IA, 245-OA. Everything else is unhyphenated (80JJA, 80LA, 80QQB, 115QA).

The canonical form follows that rule, so "80IA", "80-ia" and "80 - IA" all
normalize to "80-IA". Always compare section numbers through
normalize_section() on both sides.
"""

from __future__ import annotations

import re

# Regex fragment (no anchors or groups of its own); callers add \b or context.
# Use without re.IGNORECASE where possible: lower-case letters ("1st", "10th")
# are ordinal suffixes, not section suffixes.
SECTION_NUM = r"\d{1,3}(?:-?[A-Z]{1,5})?"

_PARTS = re.compile(r"(\d{1,3})-?([A-Z]{0,5})")
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "−": "-"})


def normalize_section(num: str) -> str:
    """Canonical section number: '80ia' -> '80-IA', '80-JJA' -> '80JJA', ' 24 ' -> '24'."""
    s = re.sub(r"\s+", "", str(num).translate(_DASHES)).upper()
    m = _PARTS.fullmatch(s)
    if not m:
        raise ValueError(f"not a section number: {num!r}")
    digits, suffix = m.groups()
    digits = str(int(digits))
    return f"{digits}-{suffix}" if suffix[:1] in ("I", "O") else digits + suffix


def try_normalize_section(num: str) -> str | None:
    try:
        return normalize_section(num)
    except ValueError:
        return None


def section_key(num: str) -> tuple[int, str]:
    """Statutory ordering key; the hyphen is ignored.

    80HHF < 80-I < 80-IA < 80-IAB < 80-IB < 80J < 80JJA, and 10 < 10A < 10AA < 10B < 11.
    """
    canon = normalize_section(num).replace("-", "")
    m = _PARTS.fullmatch(canon)
    return int(m.group(1)), m.group(2)
