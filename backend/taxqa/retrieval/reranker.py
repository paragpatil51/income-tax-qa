"""Cross-encoder reranker (CPU, ONNX via onnxruntime).

Returns calibrated relevance probabilities in [0, 1]. If the model cannot be
loaded, `scores()` returns None and the retriever falls back to fused ranks.

Cost model: a cross-encoder reads every (question, passage) pair, so latency
grows with pairs x passage length. The retriever therefore caps the number of
pairs per question (RERANK_BUDGET) and the passage length (RERANK_MAX_CHARS).
"""

from __future__ import annotations

import logging
import math
import threading

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()
        self.available = True

    def _load(self):
        if self._model is None and self.available:
            try:
                from taxqa.retrieval.onnx_models import OnnxCrossEncoder

                model = OnnxCrossEncoder(self.model_name)
                model.load()
                self._model = model
            except Exception as e:  # download / unsupported model -> degrade to fused ranking
                log.warning("Reranker %s unavailable, falling back to fused ranks: %s", self.model_name, e)
                self.available = False
        return self._model

    def scores(self, query: str, docs: list[str]) -> list[float] | None:
        if not docs:
            return []
        with self._lock:
            model = self._load()
            if model is None:
                return None
            try:
                raw = list(model.rerank(query, docs, batch_size=self.batch_size))
            except Exception as e:  # transient runtime failure: skip reranking for this query only
                log.warning("Reranking failed: %s", e)
                return None
        return [1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, float(x))))) for x in raw]

    def warmup(self) -> None:
        self.scores("warm up", ["warm up"])
