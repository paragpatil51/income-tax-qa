"""Section-number handling shared by the query planner, index and retriever.

The Act writes some section numbers with a hyphen ("80-IA", "80-IB") and
users write them every way ("80IA", "80 ia", "u/s 80-IA"). All lookups go
through `section_id()` so both sides compare in one canonical form.
"""

from __future__ import annotations

import re

try:  # canonical implementation, if ingestion provides one
    from taxqa.core.section_numbers import normalize_section as _normalize
except ImportError:  # pragma: no cover - fallback until core.section_numbers exists
    _normalize = None

# "24", "80C", "115BAC", "80-IA". Letters must be attached or hyphenated, so that
# under re.I "section 24 and 25" does not read as section "24AND".
SECTION_NUM = r"\d{1,3}(?:\s?-\s?[A-Z]{1,5}|[A-Z]{1,5})?"


def section_id(num: str) -> str:
    """Canonical lookup key for a section number: upper case, no spaces or hyphens."""
    raw = num.strip()
    if _normalize is not None:
        try:
            raw = _normalize(raw) or raw
        except Exception:  # never let a helper bug break retrieval
            pass
    return re.sub(r"[\s-]+", "", raw).upper()
