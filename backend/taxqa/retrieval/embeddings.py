"""Dense embeddings: local ONNX model (default, no GPU needed) or Gemini API."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import httpx
import numpy as np

from taxqa.core.config import Settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QUERY_CACHE_SIZE = 256
GEMINI_BATCH_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"


def _normalize(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class Embedder:
    def __init__(self, settings: Settings):
        self.s = settings
        self.backend = settings.embedding_backend
        self._model = None
        self._http = None
        # One lock for model load and inference: requests arrive on several threads, and
        # running ONNX sessions concurrently on a 4-core laptop only adds contention.
        self._lock = threading.Lock()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.s.gemini_embedding_model if self.backend == "gemini" else self.s.local_embedding_model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self.backend == "gemini":
            return self._gemini(texts, "RETRIEVAL_DOCUMENT")
        with self._lock:
            return _normalize(self._local().embed(texts))

    def embed_query(self, text: str) -> np.ndarray:
        if (hit := self._query_cache.get(text)) is not None:
            return hit
        if self.backend == "gemini":
            vec = self._gemini([text], "RETRIEVAL_QUERY")[0]
        else:
            prefix = BGE_QUERY_PREFIX if "bge" in self.name.lower() else ""
            with self._lock:
                vec = _normalize(self._local().embed([prefix + text]))[0]
        with self._cache_lock:
            self._query_cache[text] = vec
            while len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return vec

    def warmup(self) -> None:
        if self.backend != "gemini":
            self.embed_query("warm up")

    def _local(self):
        if self._model is None:
            from taxqa.retrieval.onnx_models import OnnxEmbedder

            self._model = OnnxEmbedder(self.s.local_embedding_model)
        return self._model

    def _gemini(self, texts: list[str], task: str) -> np.ndarray:
        if not self.s.gemini_api_key:
            raise RuntimeError("EMBEDDING_BACKEND=gemini requires GEMINI_API_KEY")
        self._http = self._http or httpx.Client(timeout=120)
        model = self.s.gemini_embedding_model
        out: list[list[float]] = []
        for start in range(0, len(texts), 100):
            batch = texts[start: start + 100]
            body = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task,
                        "outputDimensionality": 768,
                    }
                    for t in batch
                ]
            }
            for attempt in range(6):
                r = self._http.post(
                    GEMINI_BATCH_URL.format(model=model),
                    headers={"x-goog-api-key": self.s.gemini_api_key},
                    json=body,
                )
                if r.status_code == 200:
                    out.extend(e["values"] for e in r.json()["embeddings"])
                    break
                if r.status_code in (429, 500, 503) and attempt < 5:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise RuntimeError(f"Gemini embedding failed: HTTP {r.status_code} {r.text[:300]}")
        return _normalize(np.array(out))
