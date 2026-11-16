"""Minimal Gemini REST client with structured (JSON-schema) output.

Uses plain `requests` so there is no SDK to install. Swapping to another
provider later means writing another class with the same generate_json().
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    data: dict
    input_tokens: int
    output_tokens: int
    latency_ms: int


class GeminiClient:
    def __init__(self, api_key: str, model: str, min_interval: float = 4.5,
                 session: requests.Session | None = None, max_retries: int = 4, sleep=time.sleep):
        self.api_key = api_key
        self.model = model
        self.min_interval = min_interval  # free tier has a low requests-per-minute cap
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self._sleep = sleep
        self._last_call = 0.0

    def _headers(self) -> dict:
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        resp = self.session.get(f"{API_BASE}/models", headers=self._headers(),
                                params={"pageSize": 200}, timeout=30)
        resp.raise_for_status()
        return [m["name"].removeprefix("models/") for m in resp.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])]

    def generate_json(self, prompt: str, schema: dict) -> LLMResult:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        url = f"{API_BASE}/models/{self.model}:generateContent"
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()
            t0 = time.monotonic()
            resp = self.session.post(url, headers=self._headers(), json=body, timeout=120)
            latency_ms = int((time.monotonic() - t0) * 1000)
            if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                delay = min(60, 5 * 2 ** attempt)
                log.warning("Gemini HTTP %s, retrying in %ss", resp.status_code, delay)
                self._sleep(delay)
                continue
            if resp.status_code == 404:
                raise LLMError(f"model '{self.model}' not found. Run: python -m jobsintel.enrich --list-models")
            if resp.status_code >= 400:
                raise LLMError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")
            payload = resp.json()
            usage = payload.get("usageMetadata", {})
            try:
                cand = payload["candidates"][0]
                text = "".join(p.get("text", "") for p in cand["content"]["parts"])
                data = json.loads(text)
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                reason = (payload.get("candidates") or [{}])[0].get("finishReason") or payload.get("promptFeedback")
                raise LLMError(f"unusable Gemini response ({reason}): {exc}") from exc
            return LLMResult(
                data=data,
                input_tokens=usage.get("promptTokenCount", 0),
                # thinking models bill thought tokens as output
                output_tokens=usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
                latency_ms=latency_ms,
            )
        raise LLMError("unreachable")
