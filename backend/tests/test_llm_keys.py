"""Failover across several Gemini keys (mocked endpoints, no network, never real keys)."""

import dataclasses
import json

import httpx
import pytest

from taxqa.core import llm_client
from taxqa.core.config import settings
from taxqa.core.llm_client import LLMClient

DAILY_429 = {"error": {"code": 429, "message": "Quota exceeded per day", "details": [
    {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
OK = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)


def make(tmp_path, handler, keys=("k1", "k2", "k3"), models="m1,m2"):
    s = dataclasses.replace(settings, gemini_api_key=keys[0], gemini_api_keys=tuple(keys), gemini_model=models,
                            fallback_api_key="", cache_path=tmp_path / "c.sqlite")
    return LLMClient(s, http=httpx.Client(transport=httpx.MockTransport(handler)))


def model_and_key(req):
    return req.url.path.split("/models/")[1].split(":")[0], req.headers["x-goog-api-key"]


def test_daily_quota_on_one_key_hands_over_to_the_next_key_same_model(tmp_path):
    calls = []

    def handler(req):
        calls.append(model_and_key(req))
        return httpx.Response(429, json=DAILY_429) if req.headers["x-goog-api-key"] == "k1" else httpx.Response(200, json=OK)

    llm = make(tmp_path, handler)
    assert llm.json("s", "q1") == {"ok": True}
    assert calls == [("m1", "k1"), ("m1", "k2")]
    assert llm.last_model == "gemini:m1 (key 2)"
    assert llm.json("s", "q2") == {"ok": True}
    assert calls[-1] == ("m1", "k2")               # k1 is skipped for this model after its daily quota


def test_overloaded_model_skips_its_other_keys(tmp_path):
    calls = []

    def handler(req):
        calls.append(model_and_key(req))
        model, _ = model_and_key(req)
        return httpx.Response(503, json={"error": {"code": 503, "message": "overloaded"}}) if model == "m1" else httpx.Response(200, json=OK)

    llm = make(tmp_path, handler)
    assert llm.json("s", "q") == {"ok": True}
    assert calls == [("m1", "k1"), ("m2", "k1")]   # overload is the model's: k2 and k3 not tried on m1


def test_rejected_key_is_skipped_afterwards(tmp_path):
    calls = []

    def handler(req):
        calls.append(model_and_key(req))
        if req.headers["x-goog-api-key"] == "k1":
            return httpx.Response(403, json={"error": {"code": 403, "message": "API key not valid"}})
        return httpx.Response(200, json=OK)

    llm = make(tmp_path, handler)
    assert llm.json("s", "q1") == {"ok": True} and calls == [("m1", "k1"), ("m1", "k2")]
    assert llm.json("s", "q2") == {"ok": True} and calls[-1] == ("m1", "k2")


def test_keys_never_appear_in_labels_or_usage(tmp_path):
    secret = "test-only-secret-must-never-be-logged-0123"
    llm = make(tmp_path, lambda req: httpx.Response(200, json=OK), keys=(secret, "k2"))
    llm.json("s", "q")
    usage = llm.cache.usage_today()
    assert secret not in json.dumps(usage) and secret not in llm.last_model
    assert list(usage) == ["gemini:m1 (key 1)"]


def test_single_override_uses_only_that_key(tmp_path):
    # Overriding gemini_api_key alone (as other tests do) must never reach the configured key list.
    seen = []
    s = dataclasses.replace(settings, gemini_api_key="only", gemini_api_keys=("real1", "real2"), gemini_model="m1",
                            fallback_api_key="", cache_path=tmp_path / "c.sqlite")
    llm = LLMClient(s, http=httpx.Client(transport=httpx.MockTransport(lambda req: (seen.append(req.headers["x-goog-api-key"]), httpx.Response(200, json=OK))[1])))
    llm.json("s", "q")
    assert seen == ["only"]
