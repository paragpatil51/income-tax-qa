"""API contract tests with a fake pipeline (no index, models or network needed)."""

import dataclasses
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from taxqa.answering.schemas import Answer
from taxqa.api import main, routes
from taxqa.core.config import settings


@pytest.fixture
def client(monkeypatch, tmp_path):
    test_settings = dataclasses.replace(settings, warmup=False, processed_dir=tmp_path, index_dir=tmp_path)
    monkeypatch.setattr(main, "settings", test_settings)
    monkeypatch.setattr(routes, "settings", test_settings)
    monkeypatch.setattr(routes, "_pipeline", None)
    monkeypatch.setattr(routes, "_sections_cache", None)
    return TestClient(main.app, raise_server_exceptions=False)


def fake_pipeline(answer: Answer | None = None):
    return SimpleNamespace(
        index=SimpleNamespace(
            chunks=[SimpleNamespace(section=s, kind="section") for s in ("24", "24", "80C")],
            by_section={"24": [], "80C": []},
            meta={"built_at": "today"},
        ),
        embedder=SimpleNamespace(backend="local", name="bge"),
        reranker=SimpleNamespace(available=True),
        llm=SimpleNamespace(configured=True),
        ask=lambda q: answer or Answer(question=q, status="answered", summary="ok"),
    )


def test_health_and_ask_when_index_missing(client, monkeypatch):
    def boom(_):
        raise FileNotFoundError(2, "No such file", "chunks.jsonl")

    monkeypatch.setattr(routes, "QAPipeline", boom)
    h = client.get("/api/health")
    assert h.status_code == 200 and h.json()["ready"] is False and "ingest.py" in h.json()["detail"]
    r = client.post("/api/ask", json={"question": "Is agricultural income taxable?"})
    assert r.status_code == 503 and r.json()["error"]["type"] == "index_not_ready"


def test_health_never_raises_on_unexpected_error(client, monkeypatch):
    def boom(_):
        raise ValueError("corrupt index")

    monkeypatch.setattr(routes, "QAPipeline", boom)
    h = client.get("/api/health")
    assert h.status_code == 200 and h.json()["ready"] is False and "corrupt index" in h.json()["detail"]


def test_ask_validation_and_success(client, monkeypatch):
    monkeypatch.setattr(routes, "QAPipeline", lambda _: fake_pipeline())
    bad = client.post("/api/ask", json={"question": "   hi   "})
    assert bad.status_code == 422
    assert bad.json()["error"]["type"] == "validation_error" and "5 characters" in bad.json()["error"]["message"]
    assert client.post("/api/ask", json={"question": "x" * 2001}).status_code == 422

    ok = client.post("/api/ask", json={"question": "  Is   agricultural income taxable?  "})
    assert ok.status_code == 200
    body = ok.json()
    assert body["question"] == "Is agricultural income taxable?"
    for key in ("status", "summary", "points", "confidence", "sources", "warnings", "sub_questions", "models", "timings_ms"):
        assert key in body
    h = client.get("/api/health").json()
    assert h["ready"] is True and h["sections"] == 2


def test_section_lookup_normalises_numbers(client):
    sections = [{"num": "80-IA", "title": "Infrastructure", "lines": [[5, "80-IA. (1) Where ..."]]}]
    (routes.settings.processed_dir / "sections.json").write_text(json.dumps(sections), encoding="utf-8")
    assert client.get("/api/section/80IA").json()["num"] == "80-IA"
    assert client.get("/api/section/80-ia").status_code == 200
    missing = client.get("/api/section/999")
    assert missing.status_code == 404


def test_section_lookup_before_ingest_is_503(client):
    r = client.get("/api/section/24")
    assert r.status_code == 503 and r.json()["error"]["type"] == "index_not_ready"
