"""Hybrid index: BM25 (exact legal wording, section numbers) + dense vectors (meaning).

The corpus is a few thousand chunks, so dense search is an exact cosine
similarity over an in-memory matrix: no approximate index or database server
is needed, results are exact, and a query takes milliseconds.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from taxqa.core.config import Settings
from taxqa.core.models import Chunk
from taxqa.retrieval.embeddings import Embedder
from taxqa.retrieval.section_refs import section_id

STOPWORDS = set(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which with "
    "shall such any all may under being been can i my me we our you your do does if what how u s".split()
)
HYPHENATED_SECTION = re.compile(r"\b(\d{1,3})\s?-\s?([a-z]{1,5})\b")       # 80-IA -> 80ia
PINPOINT = re.compile(r"\b(\d{1,3}[a-z]{0,5})\s?\(\s?([0-9a-z]{1,6})\s?\)")  # 10(10D) -> 10_10d


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("u/s", " section ")
    text = HYPHENATED_SECTION.sub(r"\1\2", text)
    tokens = [f"{a}_{b}" for a, b in PINPOINT.findall(text)]
    for tok in re.findall(r"[a-z0-9]+", text):
        if tok in STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and not tok[0].isdigit():
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


def bm25_text(c: Chunk) -> str:
    """Index text with the section title repeated: a title match is strong evidence of topic."""
    return f"{c.section_title}\n{c.index_text()}" if c.section_title else c.index_text()


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _previous_vectors(settings: Settings, embedder: Embedder) -> dict[str, np.ndarray]:
    """Vectors of the existing index keyed by chunk-text hash, if built with the same model."""
    meta_path, vec_path = settings.index_dir / "meta.json", settings.index_dir / "embeddings.npy"
    if not (meta_path.exists() and vec_path.exists()):
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hashes = meta.get("text_hashes")
    if not hashes or (meta.get("backend"), meta.get("model")) != (embedder.backend, embedder.name):
        return {}
    vectors = np.load(vec_path)
    return dict(zip(hashes, vectors)) if len(hashes) == len(vectors) else {}


def build_dense_index(chunks: list[Chunk], settings: Settings, embedder: Embedder,
                      previous: dict[str, np.ndarray] | None = None) -> int:
    """Embed the chunks, reusing vectors of unchanged chunk texts. Returns how many were embedded."""
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    texts = [c.index_text() for c in chunks]
    hashes = [_text_hash(t) for t in texts]
    previous = previous if previous is not None else _previous_vectors(settings, embedder)
    todo = [i for i, h in enumerate(hashes) if h not in previous]
    fresh = embedder.embed_documents([texts[i] for i in todo]) if todo else np.zeros((0, 0), dtype=np.float32)
    new = dict(zip((hashes[i] for i in todo), fresh))
    vectors = np.vstack([new[h] if h in new else previous[h] for h in hashes]).astype(np.float32)
    np.save(settings.index_dir / "embeddings.npy", vectors)
    meta = {
        "backend": embedder.backend,
        "model": embedder.name,
        "dim": int(vectors.shape[1]),
        "count": len(chunks),
        "chunk_ids": [c.id for c in chunks],
        "text_hashes": hashes,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (settings.index_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return len(todo)


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


class HybridIndex:
    def __init__(self, settings: Settings, embedder: Embedder):
        self.s = settings
        self.embedder = embedder
        self.chunks = load_chunks(settings.chunks_path)
        if not self.chunks:
            raise RuntimeError(f"{settings.chunks_path} is empty - rerun scripts/ingest.py")
        self.by_id = {c.id: c for c in self.chunks}
        self.position = {c.id: i for i, c in enumerate(self.chunks)}
        self.by_section: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            self.by_section.setdefault(section_id(c.section), []).append(c)

        self.bm25 = BM25Okapi([tokenize(bm25_text(c)) for c in self.chunks])

        meta_path = settings.index_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["chunk_ids"] != [c.id for c in self.chunks]:
            raise RuntimeError("Dense index is out of date with chunks.jsonl - rerun scripts/ingest.py")
        if meta["backend"] != embedder.backend or meta["model"] != embedder.name:
            raise RuntimeError(
                f"Index was built with {meta['backend']}:{meta['model']} but config uses "
                f"{embedder.backend}:{embedder.name} - rerun scripts/ingest.py"
            )
        self.meta = meta
        self.vectors = np.load(settings.index_dir / "embeddings.npy")
        if self.vectors.shape[0] != len(self.chunks):
            raise RuntimeError("embeddings.npy does not match chunks.jsonl - rerun scripts/ingest.py")

    def bm25_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        return [(self.chunks[i], float(scores[i])) for i in _top(scores, k) if scores[i] > 0]

    def dense_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed_query(query)
        sims = self.vectors @ q
        return [(self.chunks[i], float(sims[i])) for i in _top(sims, k)]

    def section_chunks(self, num: str) -> list[Chunk]:
        return self.by_section.get(section_id(num), [])

    def has_section(self, num: str) -> bool:
        return section_id(num) in self.by_section

    def neighbours(self, chunk: Chunk) -> list[Chunk]:
        """Previous and next chunk of the same section (provisos/Explanations often sit there)."""
        i = self.position[chunk.id]
        out = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(self.chunks) and self.chunks[j].section == chunk.section:
                out.append(self.chunks[j])
        return out
