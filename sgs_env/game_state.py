from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from .constants import Phase, Role


@dataclass
class PlayerState:
    seat: int
    role: Role
    hp: int = 4
    max_hp: int = 4
    alive: bool = True
    hand: List[Tuple[int, int]] = field(default_factory=list)  # (card_id, suit)
    equip_weapon_range: int = 1
    equip_minus_horse: bool = False
    equip_plus_horse: bool = False
    judgement_zone: List[int] = field(default_factory=list)  # delayed tool cards


@dataclass
class GameConfig:
    num_players: int = 4
    seed: int = 0
    record_replay: bool = True


@dataclass
class GameRNG:
    seed: int
    rng: np.random.Generator


@dataclass
class GameState:
    config: GameConfig
    rng: GameRNG
    deck: List[Tuple[int, int]] = field(default_factory=list)  # (card_id, suit_int)
    discard_pile: List[Tuple[int, int]] = field(default_factory=list)
    players: Dict[str, PlayerState] = field(default_factory=dict)
    agent_order: List[str] = field(default_factory=list)
    current_agent_idx: int = 0
    current_phase: Phase = Phase.PREPARE
    turn_count: int = 0
    used_sha_in_turn: Dict[str, bool] = field(default_factory=dict)

    def serialize_public(self) -> Dict:
        return {
            "turn": self.turn_count,
            "phase": self.current_phase.value,
            "discard_pile_size": len(self.discard_pile),
            "players": {
                agent: {
                    "seat": p.seat,
                    "alive": p.alive,
                    "hp": p.hp,
                    "max_hp": p.max_hp,
                    "hand_size": len(p.hand),
                    "equip": {
                        "weapon_range": p.equip_weapon_range,
                        "minus_horse": p.equip_minus_horse,
                        "plus_horse": p.equip_plus_horse,
                    },
                    "judgement_zone_size": len(p.judgement_zone),
                }
                for agent, p in self.players.items()
            },
        }

    def to_info_rng_hash(self) -> str:
        # Simple hash from seed and deck order for determinism checks
        deck_arr = np.array(self.deck, dtype=np.int64).flatten()
        val = int(deck_arr[: min(64, deck_arr.size)].sum()) ^ int(self.rng.seed)
        return f"seed:{self.rng.seed}|h:{val}"