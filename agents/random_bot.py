from __future__ import annotations
from typing import Any
import numpy as np


class RandomLegalBot:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, legal_mask) -> int:
        legal_indices = np.nonzero(legal_mask)[0]
        if len(legal_indices) == 0:
            return 0
        return int(self.rng.choice(legal_indices))