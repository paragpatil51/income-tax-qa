"""LLM client behaviour against mocked Gemini / OpenAI-compatible endpoints (no network)."""

import dataclasses
import json

import httpx
import pytest

from taxqa.core import llm_client
from taxqa.core.config import settings
from taxqa.core.llm_client import LLMClient, LLMError, error_info, parse_json


def gemini_ok(text: str, thought: str | None = None, finish: str = "STOP") -> dict:
    parts = ([{"text": thought, "thought": True}] if thought else []) + [{"text": text}]
    return {"candidates": [{"content": {"parts": parts}, "finishReason": finish}]}


def openai_ok(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def quota_429(quota_id: str, delay: str) -> dict:
    return {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
    ]}}


def make_client(tmp_path, handler, fallback: bool = False, **overrides) -> LLMClient:
    s = dataclasses.replace(
        settings,
        gemini_api_key="test-key",
        fallback_api_key="fb-key" if fallback else "",
        cache_path=tmp_path / "cache.sqlite",
        llm_max_wait=20.0,
        **overrides,
    )
    return LLMClient(s, http=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: slept.append(s))
    return slept


def test_thought_parts_ignored_and_result_cached(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        assert req.headers["x-goog-api-key"] == "test-key"
        body = json.loads(req.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(200, json=gemini_ok('{"answer": 1}', thought="thinking about {not json}"))

    c = make_client(tmp_path, handler)
    assert c.json("sys", "user") == {"answer": 1}
    assert c.last_model.startswith("gemini:")
    assert c.json("sys", "user") == {"answer": 1}
    assert len(calls) == 1 and c.last_model.endswith("(cached)")


def test_per_minute_429_retried_after_server_delay(tmp_path, no_sleep):
    responses = [httpx.Response(429, json=quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "7s")),
                 httpx.Response(200, json=gemini_ok('{"ok": true}'))]
    c = make_client(tmp_path, lambda req: responses.pop(0))
    assert c.json("s", "u") == {"ok": True}
    assert no_sleep == [7.0]


def test_daily_quota_goes_straight_to_fallback(tmp_path, no_sleep):
    def handler(req):
        if "googleapis" in str(req.url):
            return httpx.Response(429, json=quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "40s"))
        assert req.headers["authorization"] == "Bearer fb-key"
        return httpx.Response(200, json=openai_ok('{"via": "fallback"}'))

    c = make_client(tmp_path, handler, fallback=True)
    assert c.json("s", "u") == {"via": "fallback"}
    assert no_sleep == []
    assert c.last_model.startswith("fallback:")


def test_long_rate_limit_skips_to_fallback_instead_of_waiting(tmp_path, no_sleep):
    def handler(req):
        if "googleapis" in str(req.url):
            return httpx.Response(429, json=quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "55s"))
        return httpx.Response(200, json=openai_ok('{"x": 1}'))

    c = make_client(tmp_path, handler, fallback=True)
    assert c.json("s", "u") == {"x": 1}
    assert no_sleep == []


def test_blocked_response_raises_without_fallback(tmp_path):
    c = make_client(tmp_path, lambda req: httpx.Response(200, json=gemini_ok("", finish="RECITATION")))
    with pytest.raises(LLMError, match="RECITATION"):
        c.json("s", "u")


def test_prompt_block_and_unknown_model(tmp_path):
    c = make_client(tmp_path, lambda req: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(LLMError, match="blocked"):
        c.json("s", "u")
    c = make_client(tmp_path, lambda req: httpx.Response(404, json={"error": {"message": "models/x is not found"}}))
    with pytest.raises(LLMError, match="not available"):
        c.json("s", "u2")


def test_invalid_json_triggers_one_repair_call(tmp_path):
    replies = [gemini_ok("Sure! here it is: {broken"), gemini_ok('```json\n{"fixed": true}\n```')]
    c = make_client(tmp_path, lambda req: httpx.Response(200, json=replies.pop(0)))
    assert c.json("s", "u") == {"fixed": True}


def test_fallback_without_json_mode_support(tmp_path):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append("response_format" in body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return httpx.Response(200, json=openai_ok('<think>hmm</think>{"a": 2}'))

    s = dataclasses.replace(settings, gemini_api_key="", fallback_api_key="k", cache_path=tmp_path / "c.sqlite")
    c = LLMClient(s, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.json("s", "u") == {"a": 2}
    assert seen == [True, False]


def test_error_info_parsing():
    r = httpx.Response(429, text="Rate limit reached on requests per day (RPD). Please try again in 1m26.4s")
    info = error_info(r)
    assert info.daily_quota and info.retry_after == pytest.approx(86.4)
    r = httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"message": "slow down"}})
    info = error_info(r)
    assert not info.daily_quota and info.retry_after == 3.0
    assert error_info(httpx.Response(429, text="try again in 739ms")).retry_after == pytest.approx(0.739)


def test_parse_json_rejects_non_objects():
    with pytest.raises(json.JSONDecodeError):
        parse_json("no json here")
    assert parse_json('prefix {"a": {"b": 1}} suffix') == {"a": {"b": 1}}


def test_exhausted_model_is_skipped_but_its_cache_still_used(tmp_path):
    import httpx
    from dataclasses import replace
    from taxqa.core.config import settings
    from taxqa.core.llm_client import LLMClient

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = request.url.path.split("/models/")[1].split(":")[0]
        calls.append(model)
        if model == "m1":
            return httpx.Response(429, json={"error": {"message": "Quota exceeded per day", "details": [
                {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]})

    s = replace(settings, gemini_api_key="k", gemini_model="m1,m2", fallback_api_key="",
                cache_path=tmp_path / "c.sqlite")
    llm = LLMClient(s, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.json("sys", "q1") == {"ok": True} and calls == ["m1", "m2"]
    assert llm.json("sys", "q2") == {"ok": True} and calls == ["m1", "m2", "m2"]   # m1 skipped after daily quota
    assert llm.json("sys", "q1") == {"ok": True} and len(calls) == 3                # cached: no call at all
