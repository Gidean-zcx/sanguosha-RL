from __future__ import annotations
from typing import Any, Dict, Optional
import os
import numpy as np


class LLMAdapter:
    def __init__(self, provider: str = "auto", model: Optional[str] = None, seed: int = 0):
        self.provider = provider
        self.model = model or os.getenv("LLM_MODEL", "")
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Dict[str, Any], legal_mask) -> int:
        # Fallback heuristic: random legal
        legal_indices = np.nonzero(legal_mask)[0]
        if len(legal_indices) == 0:
            return 0
        return int(self.rng.choice(legal_indices))