"""Minimal chat client for an OpenAI-compatible endpoint (Groq by default).

The API key is read from the environment (GROQ_API_KEY). Errors are returned as a short code, never
with the key or raw request content. Tests inject a fake client instead of calling the network.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class LLMReply:
    ok: bool
    text: str = ""
    error: str | None = None


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str = config.LLM_MODEL, base_url: str = config.LLM_BASE_URL,
                 timeout: float = config.LLM_TIMEOUT_S):
        self._key = api_key if api_key is not None else config.llm_api_key()
        self.model, self.base_url, self.timeout = model, base_url, timeout
        self.calls = 0
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self._key)

    def chat(self, system: str, user: str, max_tokens: int = 600, json_mode: bool = False) -> LLMReply:
        if not self._key:
            return LLMReply(False, error="LLM_NOT_CONFIGURED")
        try:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._key, base_url=self.base_url, timeout=self.timeout, max_retries=0)
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            if "gpt-oss" in self.model:  # reasoning model: keep reasoning short, leave room for the answer
                kwargs["extra_body"] = {"reasoning_effort": "low"}
                max_tokens += 1024
            self.calls += 1
            r = self._client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
            return LLMReply(True, (r.choices[0].message.content or "").strip())
        except Exception as e:  # do not leak secrets or payloads
            name = type(e).__name__
            code = "LLM_AUTH_ERROR" if "Authentication" in name else "LLM_RATE_LIMIT" if "RateLimit" in name else \
                "LLM_TIMEOUT" if "Timeout" in name else "LLM_MODEL_NOT_FOUND" if "NotFound" in name else "LLM_ERROR"
            return LLMReply(False, error=code)
