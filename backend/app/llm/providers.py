"""Configurable LLM providers with fallback.

  local  : Qwen via Ollama - live conversation state, correction parsing, cheap orchestration.
  groq   : GPT-OSS-120B via Groq - structured extraction, candidate verification, review.
  vlm    : Qwen-VL via Ollama - difficult visual interpretation (never counting/measuring).

Policy: LLM output is *advisory text/structure*. Callers must validate it against
evidence (e.g. extracted title must appear in OCR text). LLMs never produce
prices, measurements or totals.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Optional

from ..config import get_settings
from ..http import request_json

log = logging.getLogger(__name__)


def _parse_json(text: str) -> Optional[Any]:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None


class LLMRouter:
    def __init__(self):
        self.s = get_settings()
        self.stats = {"groq_calls": 0, "groq_tokens": 0, "local_calls": 0, "fallbacks": 0, "failures": 0}

    @property
    def groq_ok(self) -> bool:
        return bool(self.s.groq_api_key)

    @property
    def local_ok(self) -> bool:
        return self.s.lca_local_llm_provider == "ollama"

    async def _groq(self, system: str, user: str, json_mode: bool) -> str:
        body = {"model": self.s.lca_groq_model, "temperature": 0.0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        d = await request_json("POST", "https://api.groq.com/openai/v1/chat/completions",
                               headers={"Authorization": f"Bearer {self.s.groq_api_key}"}, json=body, timeout=30, retries=2)
        self.stats["groq_calls"] += 1
        self.stats["groq_tokens"] += int(d.get("usage", {}).get("total_tokens", 0))
        return d["choices"][0]["message"]["content"]

    async def _local(self, system: str, user: str, json_mode: bool) -> str:
        body = {"model": self.s.lca_local_llm_model, "stream": False, "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["format"] = "json"
        d = await request_json("POST", f"{self.s.ollama_base_url}/api/chat", json=body, timeout=60, retries=0)
        self.stats["local_calls"] += 1
        return d["message"]["content"]

    async def complete(self, system: str, user: str, *, prefer: str = "local", json_mode: bool = False) -> Optional[str]:
        order = ["groq", "local"] if prefer == "groq" else ["local", "groq"]
        for i, name in enumerate(order):
            if name == "groq" and not self.groq_ok: continue
            if name == "local" and not self.local_ok: continue
            try:
                return await (self._groq if name == "groq" else self._local)(system, user, json_mode)
            except Exception as e:
                self.stats["fallbacks"] += 1
                log.warning("LLM provider %s failed (%s); trying fallback", name, e)
        self.stats["failures"] += 1
        return None

    async def complete_json(self, system: str, user: str, *, prefer: str = "local") -> Optional[Any]:
        txt = await self.complete(system, user, prefer=prefer, json_mode=True)
        return _parse_json(txt or "")

    async def vision(self, image_jpeg: bytes, prompt: str) -> Optional[Any]:
        """Qwen-VL via Ollama. Returns parsed JSON or None if unavailable."""
        if not self.local_ok:
            return None
        try:
            body = {"model": self.s.lca_local_vlm_model, "stream": False, "format": "json", "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image_jpeg).decode()]}]}
            d = await request_json("POST", f"{self.s.ollama_base_url}/api/chat", json=body, timeout=90, retries=0)
            self.stats["local_calls"] += 1
            return _parse_json(d["message"]["content"])
        except Exception as e:
            log.info("VLM unavailable: %s", e)
            return None


_router: Optional[LLMRouter] = None


def router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
