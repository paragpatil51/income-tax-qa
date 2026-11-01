"""Local ONNX inference for the embedding model and the cross-encoder reranker.

Both models are plain ONNX exports published on Hugging Face. They are
downloaded once (single files over HTTPS, resumable) into backend/.models and
run with onnxruntime on the CPU, so no PyTorch, GPU or wrapper library is
needed - only `onnxruntime` and `tokenizers`.

Model ids in settings map to a repository and ONNX file below; any other id is
treated as a Hugging Face repository that ships `onnx/model.onnx` and
`tokenizer.json`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from taxqa.core.config import ROOT

log = logging.getLogger(__name__)

MODELS_DIR = ROOT / ".models"
HF_URL = "https://huggingface.co/{repo}/resolve/main/{path}"


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    onnx_path: str = "onnx/model.onnx"
    max_length: int = 512


KNOWN = {
    # Official BAAI export, full precision: used once per chunk at indexing time.
    "BAAI/bge-base-en-v1.5": ModelSpec("BAAI/bge-base-en-v1.5"),
    "BAAI/bge-small-en-v1.5": ModelSpec("BAAI/bge-small-en-v1.5"),
    # int8 export of BAAI's reranker (279 MB vs 1.1 GB): it runs on every question,
    # and quantisation makes it ~2-3x faster on CPU for a negligible ranking change.
    "BAAI/bge-reranker-base": ModelSpec("Xenova/bge-reranker-base", "onnx/model_quantized.onnx"),
    "BAAI/bge-reranker-base:fp32": ModelSpec("BAAI/bge-reranker-base"),
}


def spec_for(model_id: str) -> ModelSpec:
    return KNOWN.get(model_id) or ModelSpec(model_id)


def ensure_file(repo: str, path: str) -> Path:
    """Download one file of a Hugging Face repository (resuming a partial download)."""
    dest = MODELS_DIR / repo.replace("/", "__") / path
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = HF_URL.format(repo=repo, path=path)
    for attempt in range(5):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=httpx.Timeout(60.0, read=120.0)) as r:
                if r.status_code == 416:      # already complete
                    break
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0                  # server ignored the range: start over
                total = have + int(r.headers.get("content-length", 0))
                log.info("Downloading %s/%s (%.0f MB)", repo, path, total / 1e6)
                next_report = have + 100_000_000
                with part.open("ab" if have else "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if have >= next_report:
                            log.info("  %s: %.0f / %.0f MB", path, have / 1e6, total / 1e6)
                            next_report += 100_000_000
            break
        except (httpx.HTTPError, OSError) as e:
            log.warning("Download of %s interrupted (%s); retrying", path, e)
            if attempt == 4:
                raise RuntimeError(f"Could not download {url}: {e}") from e
    part.replace(dest)
    return dest


class _OnnxModel:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.spec = spec_for(model_id)
        self._session = None
        self._tokenizer = None
        self._inputs: set[str] = set()

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = ensure_file(self.spec.repo, self.spec.onnx_path)
        tok = Tokenizer.from_file(str(ensure_file(self.spec.repo, "tokenizer.json")))
        tok.enable_truncation(max_length=self.spec.max_length)
        pad = next((t for t in ("[PAD]", "<pad>") if tok.token_to_id(t) is not None), None)
        tok.enable_padding(pad_id=tok.token_to_id(pad) if pad else 0, pad_token=pad or "[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # A GPU is used when onnxruntime-gpu is installed (e.g. the Colab indexing notebook); laptops use the CPU.
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self._session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self._inputs = {i.name for i in self._session.get_inputs()}
        log.info("%s running on %s", self.model_id, self._session.get_providers()[0])
        self._tokenizer = tok

    def _run(self, encodings) -> np.ndarray:
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        return self._session.run(None, feeds)[0]


class OnnxEmbedder(_OnnxModel):
    """BGE bi-encoder: CLS-token embedding, L2-normalised by the caller."""

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        self.load()
        if self._session.get_providers()[0] == "CUDAExecutionProvider":
            batch_size = max(batch_size, 64)   # a GPU is far faster with larger batches
        # Sort by length so each padded batch wastes little compute; restore order at the end.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), 0), dtype=np.float32)
        vectors: list[np.ndarray] = [None] * len(texts)   # type: ignore[list-item]
        for start in range(0, len(order), batch_size):
            idx = order[start: start + batch_size]
            hidden = self._run(self._tokenizer.encode_batch([texts[i] for i in idx]))
            for i, vec in zip(idx, hidden[:, 0, :]):
                vectors[i] = vec
            if len(texts) > 500 and (start // batch_size) % 25 == 0:
                log.info("  embedded %d / %d", min(start + batch_size, len(texts)), len(texts))
        return np.vstack(vectors).astype(np.float32) if vectors else out


class OnnxCrossEncoder(_OnnxModel):
    """BGE reranker: one relevance logit per (query, passage) pair."""

    def rerank(self, query: str, docs: list[str], batch_size: int = 16) -> list[float]:
        self.load()
        scores: list[float] = []
        for start in range(0, len(docs), batch_size):
            pairs = [(query, d) for d in docs[start: start + batch_size]]
            logits = self._run(self._tokenizer.encode_batch(pairs))
            scores.extend(float(x) for x in logits.reshape(len(pairs), -1)[:, 0])
        return scores
