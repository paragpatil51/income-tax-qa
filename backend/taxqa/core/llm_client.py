"""Provider-agnostic JSON-mode LLM client.

* Primary: Google Gemini (REST, JSON response mode).
* Fallback: any OpenAI-compatible endpoint (Groq, OpenRouter, OpenAI).
* Responses are cached in SQLite keyed by (cache version, provider, model,
  prompt), which keeps demos reproducible and protects free-tier quotas.

Rate limits: a per-minute 429 is retried after the server's `retryDelay` when
that is short; a long delay or an exhausted daily quota moves straight on to
the next provider instead of blocking the user.

Several Gemini keys (GEMINI_API_KEY plus GEMINI_API_KEYS) act as failover:
each model is tried with each key in turn, so a key whose quota is used up or
that is rejected hands over to the next; an overloaded model (503) skips its
other keys, since overload is the model's, not the key's. Keys never appear
in logs or usage records, only their position ("key 2").
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from taxqa.core.config import Settings

log = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE = {408, 429, 500, 502, 503, 504}
# Gemini finish reasons that mean "no usable answer"; retrying the same prompt will not help.
BLOCKED_FINISH = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "LANGUAGE"}
# Bump when response parsing or the JSON contract changes, so stale cached answers are not reused.
CACHE_VERSION = "2"


class LLMError(RuntimeError):
    pass


@dataclass
class _ErrorInfo:
    message: str
    retry_after: float | None = None
    daily_quota: bool = False


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created REAL)")
            c.execute(
                "CREATE TABLE IF NOT EXISTS usage (day TEXT, model TEXT, calls INTEGER, errors INTEGER, "
                "prompt_tokens INTEGER, output_tokens INTEGER, PRIMARY KEY (day, model))"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def record(self, model: str, ok: bool, prompt_tokens: int = 0, output_tokens: int = 0) -> None:
        """Count real API calls per model per day (UTC, like provider quotas); cache hits are free."""
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with self.lock, self._conn() as c:
            c.execute("INSERT OR IGNORE INTO usage VALUES (?, ?, 0, 0, 0, 0)", (day, model))
            c.execute(
                "UPDATE usage SET calls = calls + 1, errors = errors + ?, prompt_tokens = prompt_tokens + ?, "
                "output_tokens = output_tokens + ? WHERE day = ? AND model = ?",
                (0 if ok else 1, prompt_tokens, output_tokens, day, model),
            )

    def usage_today(self) -> dict[str, dict]:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with self.lock, self._conn() as c:
            rows = c.execute("SELECT model, calls, errors, prompt_tokens, output_tokens FROM usage WHERE day = ?",
                             (day,)).fetchall()
        return {m: {"calls": n, "errors": e, "prompt_tokens": p, "output_tokens": o} for m, n, e, p, o in rows}

    def get(self, key: str) -> dict | None:
        with self.lock, self._conn() as c:
            row = c.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        with self.lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(value), time.time()))


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models on OpenAI-compatible hosts
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    data = json.loads(text[start: end + 1])
    if not isinstance(data, dict):
        raise json.JSONDecodeError("top-level JSON is not an object", text, 0)
    return data


def _parse_seconds(value: str) -> float | None:
    """'43s', '1.5s', '1m26.4s', '739ms', '20' -> seconds."""
    if (ms := re.fullmatch(r"\s*(\d+(?:\.\d+)?)ms\s*", value or "")):
        return float(ms.group(1)) / 1000
    m = re.fullmatch(r"\s*(?:(\d+(?:\.\d+)?)m)?\s*(\d+(?:\.\d+)?)?s?\s*", value or "")
    if not m or not (m.group(1) or m.group(2)):
        return None
    return float(m.group(1) or 0) * 60 + float(m.group(2) or 0)


THINKING_BUDGET = 1024   # tokens; unbounded thinking made answers take 1.5-3 minutes on the free tier
QUOTA_COOLDOWN_S = 3600  # skip a model this long after its daily quota is exhausted (free tier: 20 requests/day/model)


def _thinking_config(model: str) -> dict:
    """Bound the model's internal reasoning. Gemini 2.x takes a token budget, Gemini 3.x a level."""
    if re.search(r"gemini-2\.", model):
        return {"thinkingBudget": 0 if "lite" in model else THINKING_BUDGET}
    return {"thinkingLevel": "low"}


def error_info(resp: httpx.Response) -> _ErrorInfo:
    """Extract message, retry delay and daily-quota flag from Gemini / OpenAI-style error bodies."""
    text = resp.text[:2000]
    info = _ErrorInfo(message=f"HTTP {resp.status_code}: {text[:400]}")
    try:
        err = resp.json().get("error", {})
    except (ValueError, AttributeError):
        err = {}
    if isinstance(err, dict):
        if err.get("message"):
            info.message = f"HTTP {resp.status_code}: {str(err['message'])[:400]}"
        for d in err.get("details") or []:
            if not isinstance(d, dict):
                continue
            if d.get("retryDelay"):
                info.retry_after = _parse_seconds(str(d["retryDelay"]))
            for v in d.get("violations") or []:
                if "PerDay" in str(v.get("quotaId", "")):
                    info.daily_quota = True
    if info.retry_after is None and resp.headers.get("retry-after"):
        info.retry_after = _parse_seconds(resp.headers["retry-after"])
    if info.retry_after is None and (m := re.search(r"try again in ((?:\d+m)?\d+(?:\.\d+)?m?s)", text)):
        info.retry_after = _parse_seconds(m.group(1))
    if resp.status_code == 429 and re.search(r"per ?day|PerDay|\bRPD\b|daily", text, re.I):
        info.daily_quota = True
    return info


class LLMClient:
    def __init__(self, settings: Settings, http: httpx.Client | None = None):
        self.s = settings
        self.http = http or httpx.Client(timeout=httpx.Timeout(settings.llm_timeout, connect=10.0))
        self.cache = _Cache(settings.cache_path)
        self._local = threading.local()
        self._exhausted: dict[tuple[str, str, int], float] = {}   # (provider, model, key no.) -> skip until (epoch s)

    @property
    def configured(self) -> bool:
        return bool(self._gemini_keys or self.s.fallback_api_key)

    @property
    def _gemini_keys(self) -> tuple[str, ...]:
        keys = tuple(getattr(self.s, "gemini_api_keys", ()) or ())
        if keys and keys[0] == self.s.gemini_api_key:
            return keys
        # gemini_api_key set on its own (or overridden, as tests do): that single key, nothing else
        return (self.s.gemini_api_key,) if self.s.gemini_api_key else ()

    def _key(self, provider: str, idx: int) -> str:
        return self._gemini_keys[idx] if provider == "gemini" else self.s.fallback_api_key

    def _label(self, provider: str, model: str, idx: int) -> str:
        many = provider == "gemini" and len(self._gemini_keys) > 1
        return f"{provider}:{model}" + (f" (key {idx + 1})" if many else "")

    @property
    def last_model(self) -> str:
        """Model that served the most recent call *in this thread* (requests run in a thread pool)."""
        return getattr(self._local, "model", "")

    def _providers(self, light: bool) -> list[tuple[str, str, int]]:
        """Ordered fallback chain of (provider, model, key no.). GEMINI_MODEL / GEMINI_LIGHT_MODEL may
        list several models ("gemini-flash-latest,gemini-2.5-flash"); each is tried with every key."""
        out = []
        keys = self._gemini_keys
        if keys:
            models = self.s.gemini_light_model if light else self.s.gemini_model
            out.extend(("gemini", m.strip(), i) for m in models.split(",") if m.strip() for i in range(len(keys)))
        if self.s.fallback_api_key:
            out.append(("fallback", self.s.fallback_model, 0))
        return out

    def json(self, system: str, user: str, *, light: bool = False, temperature: float = 0.0) -> dict:
        providers = self._providers(light)
        if not providers:
            raise LLMError("No LLM configured: set GEMINI_API_KEY (or FALLBACK_API_KEY) in backend/.env")
        errors = []
        # The answer does not depend on the key, so the cache is keyed by model only.
        def cache_key(provider: str, model: str) -> str:
            return hashlib.sha256(
                json.dumps([CACHE_VERSION, provider, model, system, user, temperature]).encode()
            ).hexdigest()

        for provider, model, _ in providers:   # a cached answer from any model in the chain is free
            if (hit := self.cache.get(cache_key(provider, model))) is not None:
                self._local.model = f"{provider}:{model} (cached)"
                return hit
        # (model, key) pairs whose daily quota ran out or whose key was rejected are skipped for a while
        # (they would only add a failing round trip), unless every pair is exhausted - then all are tried.
        now = time.time()
        live = [p for p in providers if self._exhausted.get(p, 0) <= now] or providers
        overloaded: set[tuple[str, str]] = set()
        for n, (provider, model, idx) in enumerate(live):
            if (provider, model) in overloaded:
                continue
            label = self._label(provider, model, idx)
            try:
                data = self._json_with_repair(provider, model, idx, system, user, temperature, n < len(live) - 1)
            except LLMError as e:
                log.warning("LLM %s failed: %s", label, e)
                errors.append(f"{label}: {e}")
                if "daily quota" in str(e) or "key rejected" in str(e):
                    self._exhausted[(provider, model, idx)] = time.time() + QUOTA_COOLDOWN_S
                elif "overloaded" in str(e):
                    overloaded.add((provider, model))
                continue
            self.cache.set(cache_key(provider, model), data)
            self._local.model = label
            return data
        raise LLMError(" | ".join(errors))

    def _json_with_repair(self, provider: str, model: str, idx: int, system: str, user: str,
                          temperature: float, can_fallback: bool) -> dict:
        text = self._call(provider, model, system, user, temperature, can_fallback, idx)
        try:
            return parse_json(text)
        except json.JSONDecodeError as e:
            repair = (
                f"{user}\n\nYour previous reply was not valid JSON ({e.msg}). "
                "Reply again with ONLY the JSON object, no prose and no code fences."
            )
            try:
                return parse_json(self._call(provider, model, system, repair, temperature, can_fallback, idx))
            except json.JSONDecodeError as e2:
                raise LLMError(f"invalid JSON from model: {e2.msg}") from e2

    def _post(self, provider: str, model: str, system: str, user: str, temperature: float,
              json_mode: bool, idx: int = 0) -> httpx.Response:
        if provider == "gemini":
            config = {"temperature": temperature, "responseMimeType": "application/json"}
            if json_mode:   # json_mode is switched off below if a model rejects the thinking setting
                config["thinkingConfig"] = _thinking_config(model)
            return self.http.post(
                GEMINI_URL.format(model=model),
                headers={"x-goog-api-key": self._key(provider, idx)},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": config,
                },
            )
        body = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return self.http.post(
            f"{self.s.fallback_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.s.fallback_api_key}"},
            json=body,
        )

    def _call(self, provider: str, model: str, system: str, user: str, temperature: float,
              can_fallback: bool = False, idx: int = 0) -> str:
        json_mode = True
        label = self._label(provider, model, idx)
        for attempt in range(3):
            try:
                resp = self._post(provider, model, system, user, temperature, json_mode, idx)
            except httpx.TimeoutException as e:
                if can_fallback or attempt == 2:
                    raise LLMError(f"timed out after {self.s.llm_timeout:.0f}s") from e
                continue
            except httpx.HTTPError as e:
                if attempt == 2:
                    raise LLMError(f"network error: {e}") from e
                time.sleep(2 * (attempt + 1))
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    meta = data.get("usageMetadata") or data.get("usage") or {}
                    self.cache.record(label, True,
                                      meta.get("promptTokenCount") or meta.get("prompt_tokens") or 0,
                                      (meta.get("candidatesTokenCount") or meta.get("completion_tokens") or 0)
                                      + (meta.get("thoughtsTokenCount") or 0))
                    return self._extract_text(provider, data)
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    raise LLMError(f"unexpected response shape: {e}") from e
            self.cache.record(label, False)

            info = error_info(resp)
            if resp.status_code == 400 and provider == "fallback" and json_mode and "response_format" in resp.text:
                json_mode = False  # provider/model without JSON mode: rely on the prompt instead
                continue
            if resp.status_code == 400 and provider == "gemini" and json_mode and re.search(r"thinking", resp.text, re.I):
                json_mode = False  # model without this thinking control: send the request without it
                continue
            if resp.status_code == 404:
                raise LLMError(f"model '{model}' not available ({info.message}); check GEMINI_MODEL / FALLBACK_MODEL")
            if resp.status_code in (401, 403):
                raise LLMError(f"API key rejected ({info.message})")
            if resp.status_code == 503 and can_fallback:
                # "Model overloaded" lasts minutes, not seconds: use the next model right away.
                raise LLMError(f"model overloaded; trying next model ({info.message})")
            if resp.status_code in RETRYABLE and not info.daily_quota and attempt < 2:
                delay = info.retry_after if info.retry_after is not None else 3.0 * (attempt + 1)
                if can_fallback and delay > self.s.llm_max_wait:
                    raise LLMError(f"rate limited for {delay:.0f}s; trying next provider ({info.message})")
                time.sleep(min(delay, 60.0))
                continue
            if info.daily_quota:
                raise LLMError(f"daily quota exhausted ({info.message})")
            raise LLMError(info.message)
        raise LLMError("exhausted retries")

    @staticmethod
    def _extract_text(provider: str, data: dict) -> str:
        if provider == "gemini":
            block = (data.get("promptFeedback") or {}).get("blockReason")
            if block:
                raise LLMError(f"prompt blocked by provider ({block})")
            candidates = data.get("candidates") or []
            if not candidates:
                raise LLMError("no candidates in response")
            cand = candidates[0]
            finish = cand.get("finishReason", "")
            parts = (cand.get("content") or {}).get("parts") or []
            # Thinking models return their reasoning as parts flagged "thought": true.
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
            if finish in BLOCKED_FINISH:
                raise LLMError(f"response blocked by provider (finishReason={finish})")
            if not text.strip():
                raise LLMError(f"empty response (finishReason={finish or 'unknown'})")
            return text
        content = data["choices"][0]["message"].get("content")
        if not content:
            raise LLMError("empty response from fallback provider")
        return content
