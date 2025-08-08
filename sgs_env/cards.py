from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

# Suits: 0 spade, 1 heart, 2 club, 3 diamond
SUITS = [0, 1, 2, 3]
RANKS = list(range(1, 14))

# Supported names (include >=20 equipments across weapons/armors/horses)
NAMES = [
    # basics and tricks
    "sha", "shan", "tao", "juedou", "guohe", "shunshou", "nanman", "wanjian", "wuxie",
    "le", "bingliang", "shandian",
    # weapons/armors/horses (subset with simple effects wired)
    "crossbow", "qinggang", "qinglong", "fangtian", "guanshi", "qilin", "zhangba", "hanbing", "guding",
    "bagua", "renwang", "tengjia", "baiyin", "minus_horse", "plus_horse",
]

CARD_ID_TO_NAME: Dict[int, str] = {i + 1: n for i, n in enumerate(NAMES)}
NAME_TO_CARD_ID: Dict[str, int] = {n: i for i, n in CARD_ID_TO_NAME.items()}


def build_standard_deck(rng: np.random.Generator, size: int = 108) -> List[Tuple[int, int, int]]:
    # Construct a deck by sampling names with weights to approximate distribution
    weights = {
        "sha": 20,
        "shan": 18,
        "tao": 8,
        "juedou": 3,
        "guohe": 4,
        "shunshou": 4,
        "nanman": 2,
        "wanjian": 2,
        "wuxie": 3,
        "le": 2,
        "bingliang": 2,
        "shandian": 1,
        "crossbow": 1,
        "qinggang": 1,
        "qinglong": 1,
        "fangtian": 1,
        "guanshi": 1,
        "qilin": 1,
        "zhangba": 1,
        "hanbing": 1,
        "guding": 1,
        "bagua": 1,
        "renwang": 1,
        "tengjia": 1,
        "baiyin": 1,
        "minus_horse": 2,
        "plus_horse": 2,
    }
    name_list = list(weights.keys())
    probs = np.array([weights[n] for n in name_list], dtype=np.float64)
    probs /= probs.sum()
    deck: List[Tuple[int, int, int]] = []
    for _ in range(size):
        name = rng.choice(name_list, p=probs)
        cid = NAME_TO_CARD_ID[name]
        suit = int(rng.choice(SUITS))
        rank = int(rng.choice(RANKS))
        deck.append((cid, suit, rank))
    rng.shuffle(deck)
    return deck