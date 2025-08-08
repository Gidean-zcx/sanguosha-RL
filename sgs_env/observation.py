from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from .constants import ACTION_SPACE_SIZE, Action, Phase
from .game_state import GameState, PlayerState


def build_observation(state: GameState, agent: str) -> Dict:
    me: PlayerState = state.players[agent]
    public = state.serialize_public()
    obs = {
        "self": {
            "hp": me.hp,
            "max_hp": me.max_hp,
            "hand": list(me.hand),
            "equip": {
                "weapon_range": me.equip_weapon_range,
                "minus_horse": me.equip_minus_horse,
                "plus_horse": me.equip_plus_horse,
            },
            "judgement_zone": list(me.judgement_zone),
        },
        "public": public,
        "turn_agent": state.agent_order[state.current_agent_idx],
        "phase": state.current_phase.value,
    }
    return obs


def build_legal_action_mask(state: GameState, agent: str) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    me = state.players[agent]
    if not me.alive:
        mask[Action.PASS] = 1
        return mask

    if state.agent_order[state.current_agent_idx] != agent:
        # not my turn: only responses (simplified: allow pass)
        mask[Action.PASS] = 1
        return mask

    phase = state.current_phase
    if phase in (Phase.PREPARE, Phase.JUDGEMENT, Phase.END):
        mask[Action.CONFIRM] = 1
        return mask

    if phase == Phase.DRAW:
        mask[Action.CONFIRM] = 1
        return mask

    if phase == Phase.PLAY:
        # simplified: can PASS or PLAY_SHA once if has card, PLAY_TAO if has card
        mask[Action.PASS] = 1
        has_sha = any_card_named(state, me, "sha")
        has_tao = any_card_named(state, me, "tao")
        if has_sha and not state.used_sha_in_turn.get(agent, False):
            mask[Action.PLAY_SHA] = 1
        if has_tao and me.hp < me.max_hp:
            mask[Action.PLAY_TAO] = 1
        return mask

    if phase == Phase.DISCARD:
        # must discard until hand_size <= hp (hand limit=hp)
        over = len(me.hand) - me.hp
        if over > 0:
            mask[Action.DISCARD_CARD] = 1
        else:
            mask[Action.CONFIRM] = 1
        return mask

    mask[Action.PASS] = 1
    return mask


def any_card_named(state: GameState, me: PlayerState, name: str) -> bool:
    for cid, _suit in me.hand:
        if card_name(cid) == name:
            return True
    return False


# Minimal mapping for demo
_CARD_ID_TO_NAME = {
    1: "sha",
    2: "shan",
    3: "tao",
}


def card_name(card_id: int) -> str:
    return _CARD_ID_TO_NAME.get(card_id, f"unknown:{card_id}")