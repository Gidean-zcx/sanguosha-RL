from __future__ import annotations
from typing import Any, Dict, Optional, List
import os
import json
import numpy as np

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # lazy optional


def _build_prompt(observation: Dict[str, Any], legal_indices: List[int]) -> str:
    return (
        "You are an agent playing Sanguosha. You must output exactly one integer that is a legal action index.\n"
        "Rules:\n"
        "- Action space is fixed-size; only actions with mask=1 are legal.\n"
        "- Output only the integer (no text).\n"
        "Observation (json):\n" + json.dumps(observation, ensure_ascii=False) + "\n"
        "Legal indices: " + ",".join(map(str, legal_indices)) + "\n"
        "Your answer:"
    )


class LLMAdapter:
    def __init__(self, provider: str = "auto", model: Optional[str] = None, seed: int = 0, endpoint: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "auto")).lower()
        self.model = model or os.getenv("LLM_MODEL", "")
        self.endpoint = endpoint or os.getenv("LLM_ENDPOINT", "")
        self.rng = np.random.default_rng(seed)

    def _choose_random(self, legal_mask) -> int:
        legal_indices = np.nonzero(legal_mask)[0]
        if len(legal_indices) == 0:
            return 0
        return int(self.rng.choice(legal_indices))

    def _parse_int(self, text: str, legal_indices: List[int]) -> Optional[int]:
        text = text.strip()
        # try plain int
        try:
            val = int(text)
            return val if val in legal_indices else None
        except Exception:
            pass
        # try to extract first integer in text
        num = ""
        for ch in text:
            if ch.isdigit():
                num += ch
            elif num:
                break
        if num:
            try:
                val = int(num)
                return val if val in legal_indices else None
            except Exception:
                return None
        return None

    def _chat_openai(self, prompt: str, legal_indices: List[int]) -> Optional[int]:
        if httpx is None:
            return None
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not api_key:
            return None
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 8,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return self._parse_int(text, legal_indices)
        except Exception:
            return None

    def _chat_openrouter(self, prompt: str, legal_indices: List[int]) -> Optional[int]:
        if httpx is None:
            return None
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = self.model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        if not api_key:
            return None
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 8,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return self._parse_int(text, legal_indices)
        except Exception:
            return None

    def _chat_endpoint(self, prompt: str, legal_indices: List[int]) -> Optional[int]:
        if httpx is None or not self.endpoint:
            return None
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.post(
                    self.endpoint,
                    json={"prompt": prompt, "model": self.model},
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("text") or data.get("output") or ""
                return self._parse_int(text, legal_indices)
        except Exception:
            return None

    def act(self, observation: Dict[str, Any], legal_mask) -> int:
        legal_indices = list(np.nonzero(legal_mask)[0])
        if not legal_indices:
            return 0
        # Build prompt
        prompt = _build_prompt(observation, legal_indices)
        # Route by provider
        providers = []
        if self.provider in ("auto", "openai"):
            providers.append(self._chat_openai)
        if self.provider in ("auto", "openrouter"):
            providers.append(self._chat_openrouter)
        if self.provider in ("auto", "endpoint", "custom"):
            providers.append(self._chat_endpoint)
        # Try in order
        for fn in providers:
            try:
                val = fn(prompt, legal_indices)
                if isinstance(val, int) and val in legal_indices:
                    return val
            except Exception:
                continue
        # Fallback: random legal
        return self._choose_random(legal_mask)