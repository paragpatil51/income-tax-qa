# Design decisions and trade-offs

Short notes for explaining and defending the architecture. Each entry covers the decision, why it was made, the trade-off, and what would change it.

---

### 1. The core principle: retrieval decides the law, the LLM only explains it
- **Decision.** The LLM never chooses which provisions apply. It only sees provisions retrieved from the ingested Act, and every statement it makes must cite one of them, with a verbatim quote.
- **Why.** The brief says the system "must not rely solely on the LLM's general or pre-trained knowledge to decide which legal provisions apply". Models also misremember section numbers and limits, especially after amendments.
- **How it is enforced.**
  - The query planner is told not to name provisions. A rewrite that names a section or clause the user did not write is discarded, and the user's own wording is always searched too.
  - The generator sees numbered excerpts only.
  - The verifier rejects citations to sections that were not retrieved.
- **If asked "what if retrieval misses the right section?"** The answer degrades to "partial" or "insufficient" instead of being filled in from memory, which is the safer failure for legal content. Retrieval recall is measured separately by `evaluate.py`.

### 2. Structure-aware parsing instead of fixed-size chunking
- **Decision.** Rebuild Chapter → Section → subsection/clause from the PDF, and chunk at the top-level subdivision (e.g. `80C(2)`, `24(a)-(b)`).
- **Why.** The required output is a section-level citation with a PDF page, so the chunk boundaries must be legal boundaries. Fixed 500-token windows would cut clauses in half and lose the section number.
- **Key detail.** The section's opening words are attached to each chunk as `context`. Clause 24(a) is a fragment ("a sum equal to thirty per cent of the annual value") that only makes sense with its lead-in.
- **Trade-off.** Layout heuristics depend on the PDF format. They are mitigated by order-constrained section detection (bold section numbers, or the table of contents where a PDF has one), the ingestion report (missing numbers, untitled or near-empty sections), `inspect_pdf.py`, and synthetic-PDF unit tests.

### 2a. Decoding the official PDF instead of OCR or a third-party copy
- **Problem.** The official FA 2026 PDF draws its text with Type3 bitmap fonts that have no Unicode map, so every page extracts as gibberish.
- **Decision.** Decode it exactly:
  - Glyph names (`/c100`) carry the Windows-1252 code of the source character.
  - Bold and italic are measured from the glyph bitmaps (ink share, slant).
  - Invisible filler glyphs are removed: those with no ink on the rendered page, and those squeezed under the next letter.
  - Line-break hyphens are resolved with the Act's own vocabulary.
- **Why.** The brief requires the official PDF, page numbers and exact statutory language. OCR (about 1,100 pages) is slower, inexact on section numbers such as "(1A)", and loses bold and italic, which the parser needs. A third-party clean copy would not be the official 2026 text.
- **Evidence.** 0 replacement characters, and a 99.4% word-level match with the FA 2025 text of the same Act. The ingestion report shows all 52 chapters and 916 sections.

### 3. Section detection: maximum-weight increasing subsequence
- **Problem.** "30." at the start of a line may be section 30, or a wrapped "…under section / 30. …".
- **Solution.** Score each candidate by evidence: a bold section number, a bold heading above it, a finished sentence before it, not preceded by "section", and a listing in the table of contents where the PDF has one. Then keep the best-scoring subsequence whose numbers strictly increase in statutory order (10 < 10A < 10AA < 10B, 80HHF < 80-I < 80-IA). Out-of-order matches cannot survive.
- **Why not regex alone?** Precision. **Why not an LLM?** It is non-deterministic, slow and costly over thousands of pages, and its output cannot be audited.

### 4. Hybrid retrieval (BM25 + dense) with Reciprocal Rank Fusion
- **BM25** catches exact statutory terms and section numbers ("annual value", "80C", "Hindu undivided family"), which embeddings blur.
- **Dense embeddings** catch paraphrase: "rented out" ≈ "let out", "housing loan" ≈ "capital borrowed".
- **RRF** fuses rankings without having to calibrate two different score scales against each other.
- **Query planning** adds a "statutory phrasing" of each sub-question, which closes most of the vocabulary gap between lay questions and legal text.

### 5. Cross-encoder reranker
- **Decision.** Rescore the fused candidates with `bge-reranker-base`, which reads the question and passage together.
- **Why.** It is far more precise than embedding similarity, and it outputs a 0–1 relevance that is reused for **abstention** (no answer if the best passage is below `MIN_RELEVANCE`) and for the **retrieval component** of the confidence score.
- **Trade-off.** CPU latency. It is bounded by capping the number of passages scored per question.

### 6. Exact NumPy search instead of a vector database
- The Act yields a few thousand chunks. Exact cosine similarity over an in-memory matrix takes milliseconds, has no recall loss from approximation, and needs no server.
- A vector DB (Qdrant/pgvector) becomes worthwhile when adding Rules, circulars and case law, or when several users need concurrent writes. Only `hybrid_index.py` would change.

### 7. Local ONNX embeddings, hosted LLM
- **Constraint.** The development laptop has 8 GB RAM, integrated graphics and no CUDA.
- **Embeddings and reranker** are small, run well on CPU as ONNX exports executed directly with onnxruntime (no PyTorch, no wrapper library), and give reproducible, offline indexing with no API rate limits.
- **A local LLM** that fits in 8 GB (3–8B, quantised) would be slow and too weak at careful legal reasoning, so generation uses Gemini's free tier.
- **Provider risk** is handled by a provider-agnostic client with an OpenAI-compatible fallback (Groq/OpenRouter) and a response cache.

### 8. Structured citations plus a verifier
- **Decision.** The model must output JSON in which each point lists `{source_id, provision, quote}`. A verifier then checks:
  1. the section was retrieved,
  2. the sub-division marker exists in that text,
  3. the quote matches the statute (fuzzy ≥ 88/100; short quotes must match exactly), and any figure in the statement appears in the cited text,
  4. a separate entailment check confirms the statement follows from the cited text.
- **Why.** The brief's central requirement is verifiable citations, and a prompt instruction is not a guarantee. These checks turn "please cite" into something that is measured and enforced.
- **Failure handling.** Statements without a traceable citation, or judged unsupported, are removed and shown with the reason, which is transparent rather than silent. Page numbers and document names come from the index, never from the LLM.

### 9. Confidence score
- **Formula.** 35% entailment support + 25% citation validity + 25% retrieval relevance of cited provisions + 15% sub-question coverage.
- **Rationale.**
  - Support has the highest weight because an unsupported statement is the costliest error.
  - Citation validity and retrieval relevance measure how well the evidence is tied to the answer.
  - Coverage penalises answers that silently skip part of the question.
- **Why it isn't arbitrary.** Every input is measured, and the UI shows the breakdown. The LLM's self-assessment is never used.
- **Honest limitation.** The weights are heuristics. With labelled data they should be fitted (e.g. logistic regression against correctness labels) and checked with a reliability diagram.

### 10. Abstention and out-of-scope handling
- **Before generation.** If the best reranked passage is below `MIN_RELEVANCE`, the system declines and shows the closest provisions "for reference only".
- **Out of scope.** The planner flags questions about other laws (GST, stamp duty), and the system declines unless the Act clearly matches.
- **After generation.** If every statement fails verification, the draft is withheld.
- **Evaluation.** Four questions outside the Act measure correct refusals; answerable questions measure false abstentions. Both feed the choice of threshold.

### 11. Edition choice
- The brief asks for the Income-tax Act, 1961 "for year 2026". The CBDT publishes *Income-tax Act, 1961 as amended by Finance Act, 2026*, so that edition is used.
- **Key context to raise in the interview:** the Income-tax Act, 2025 came into force on 1 April 2026 and repealed the 1961 Act, subject to savings and transitional provisions.
- Answers therefore apply to earlier years and pending matters, and this is shown in the UI and README.
- The natural extension is to ingest the 2025 Act plus the official section-mapping table and show the corresponding new section.

### 12. What I would do with more time
1. Calibrate `MIN_RELEVANCE` and the confidence weights on 100+ expert-labelled questions.
2. Ingest the Income-tax Act, 2025, with 1961 ↔ 2025 section mapping.
3. Add Income-tax Rules and CBDT circulars as separately labelled sources, with source-type filters.
4. Show page-image previews with the quoted text highlighted, using the PDF coordinates already extracted.
5. Resolve the date of amendments marked as future-dated, so answers say "as applicable for tax year X".
6. Run a regression evaluation in CI on every change to the parser or prompts.
