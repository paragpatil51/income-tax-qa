"""Query understanding: explicit section references and sub-question planning.

The LLM is used here only to *rephrase* the user's question into the
vocabulary of the statute ("rented out" -> "let out", "housing loan" ->
"capital borrowed") and to split multi-part questions. It is told not to name
sections, and any section numbers it adds that the user did not write are
stripped. Which provisions apply is therefore decided by retrieval over the
Act itself, not by the model's memory of the law.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from taxqa.core.llm_client import LLMClient, LLMError
from taxqa.retrieval.section_refs import SECTION_NUM, section_id

MAX_SUB_QUERIES = 4

# "section 80C", "sections 23 and 24", "sec. 10(10D)", "s. 24(b)", "u/s 80-IA"
EXPLICIT_SECTION = re.compile(
    rf"(?:\bsections?|\bsecs?\.?|\bs\.|\bu/s\.?)\s*({SECTION_NUM})\b"
    rf"((?:\s*(?:,|and|or|&)\s*{SECTION_NUM}\b)*)",
    re.I,
)
# Bare references such as "80C", "80-IA" or "115BAC" (upper case only, so "1st"/"10th" do not match;
# not inside parentheses, so the clause in "10(10D)" is not read as section 10D)
BARE_SECTION = re.compile(r"(?<![(\w-])(\d{1,3}-?[A-Z]{1,5})\b(?!\s*\))")
SECTION_MENTION = re.compile(rf"\b(?:sections?|u/s)\s+({SECTION_NUM})(?:\s*\([^)]{{1,8}}\))*", re.I)

PLANNER_SYSTEM = """You prepare search queries for a retrieval system over the Indian Income-tax Act, 1961.
You do NOT answer the question.

Tasks:
1. "in_scope": true if the question is about Indian income tax law (income, deductions, exemptions,
   TDS/TCS, returns, assessment, penalties, residence, capital gains, etc.); false for other laws
   (GST, customs, stamp duty, company law, criminal law) or non-legal questions.
2. Split the question into 1-4 self-contained sub-questions, one per distinct legal issue.
   Keep the facts that matter (e.g. "the property is let out", "loan taken for construction").
3. For each sub-question write a "statutory_query": the same issue in the formal vocabulary a statute
   uses. Examples:
   - "rented out" -> "let out"; "housing loan" -> "capital borrowed for acquisition or construction"
   - "salary" -> "income chargeable under the head Salaries"; "selling shares" -> "transfer of a capital asset"
   - "NRI" -> "person not resident in India"; "my parents" -> "parents of the assessee"
   - "tax-free" -> "shall not be included in computing the total income"
   - "can I claim / deduct" -> "there shall be deducted / deduction shall be allowed"
   - "fine / punishment" -> "penalty / prosecution"; "late filing" -> "return furnished after the due date"

Rules:
- NEVER add section numbers, rates, amounts, limits or dates that the user did not write.
- Never refer to a provision by number ("clause (a)", "sub-section (1)", "section 24"): describe its
  substance instead ("deduction of thirty per cent of the annual value").
- Do not include legal conclusions; only restate what is asked.
- One sub-question is fine when the question has one issue.

Return JSON only: {"in_scope": true, "sub_questions": [{"question": "...", "statutory_query": "..."}]}"""


@dataclass
class SubQuery:
    question: str
    statutory_query: str

    @property
    def rerank_text(self) -> str:
        if self.statutory_query and self.statutory_query.lower() != self.question.lower():
            return f"{self.question} ({self.statutory_query})"
        return self.question


@dataclass
class QueryPlan:
    original: str
    sub_queries: list[SubQuery]
    explicit_sections: list[str] = field(default_factory=list)
    in_scope: bool = True
    planner: str = "rule-based"


def explicit_sections(question: str) -> list[str]:
    """Section numbers the user wrote, in canonical form ("80-IA" -> "80IA")."""
    out: list[str] = []

    def add(num: str) -> None:
        sid = section_id(num)
        if sid and sid not in out:
            out.append(sid)

    for m in EXPLICIT_SECTION.finditer(question):
        add(m.group(1))
        for extra in re.findall(SECTION_NUM, m.group(2) or "", flags=re.I):
            add(extra)
    for m in BARE_SECTION.finditer(question):
        add(m.group(1))
    return out


def _strip_new_sections(text: str, allowed: set[str]) -> str:
    """Remove section references the planner introduced that the user never wrote."""
    def repl(m: re.Match) -> str:
        return m.group(0) if section_id(m.group(1)) in allowed else ""
    return re.sub(r"\s{2,}", " ", SECTION_MENTION.sub(repl, text)).strip()


SUBDIVISION_MENTION = re.compile(r"\b(?:sub-section|sub-clause|clause|proviso|Explanation)\s*\(?\s*[0-9A-Za-z]{0,6}\s*\)?\s+(?:of|to)\b", re.I)


def _introduces_provisions(text: str, question: str, allowed: set[str]) -> bool:
    """True when a rewrite names provisions the user did not: such a rewrite leans on the model's
    memory of the Act, and stripping the numbers leaves fragments ("clause (a) of is allowable")
    that mislead retrieval - so the rewrite is dropped instead."""
    if any(section_id(m.group(1)) not in allowed for m in SECTION_MENTION.finditer(text)):
        return True
    return bool(SUBDIVISION_MENTION.search(text)) and not SUBDIVISION_MENTION.search(question)


def plan_query(question: str, llm: LLMClient | None) -> QueryPlan:
    question = " ".join(question.split())
    sections = explicit_sections(question)
    fallback = QueryPlan(question, [SubQuery(question, question)], sections)
    if llm is None or not llm.configured:
        return fallback
    try:
        data = llm.json(PLANNER_SYSTEM, f"Question: {question}", light=True)
    except LLMError:
        return fallback

    allowed = set(sections)
    subs: list[SubQuery] = []
    seen: set[str] = set()
    for item in (data.get("sub_questions") or [])[:MAX_SUB_QUERIES]:
        if not isinstance(item, dict):
            continue
        q = _strip_new_sections(str(item.get("question", "")).strip(), allowed)
        sq = str(item.get("statutory_query", "")).strip()
        if _introduces_provisions(sq, question, allowed):
            sq = q   # fall back to the plain sub-question rather than a half-stripped rewrite
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        subs.append(SubQuery(q, sq or q))
    if not subs:
        return fallback
    in_scope = data.get("in_scope", True)
    return QueryPlan(
        question,
        subs,
        sections,
        in_scope=in_scope if isinstance(in_scope, bool) else str(in_scope).lower() != "false",
        planner=llm.last_model,
    )
