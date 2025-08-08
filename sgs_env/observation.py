from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from .constants import ACTION_SPACE_SIZE, Action, Phase
from .game_state import GameState, PlayerState
from .actions import (
    INDEX_PASS,
    INDEX_TAO,
    INDEX_CONFIRM,
    INDEX_RESP_SHAN,
    INDEX_SHA_BASE,
    NUM_SEAT_SLOTS,
    INDEX_DISCARD_BASE,
    NUM_DISCARD_SLOTS,
    action_index_for_sha_to_seat,
)


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
        mask[INDEX_PASS] = 1
        return mask

    if state.agent_order[state.current_agent_idx] != agent:
        # not my turn or response turn for someone else
        if state.response_pending and state.response_pending.get("defender") == agent:
            # can play SHAN if has it or PASS
            has_shan = any_card_named(state, me, "shan")
            mask[INDEX_PASS] = 1
            if has_shan:
                mask[INDEX_RESP_SHAN] = 1
            return mask
        mask[INDEX_PASS] = 1
        return mask

    phase = state.current_phase
    if phase in (Phase.PREPARE, Phase.JUDGEMENT, Phase.END):
        mask[INDEX_CONFIRM] = 1
        return mask

    if phase == Phase.DRAW:
        mask[INDEX_CONFIRM] = 1
        return mask

    if phase == Phase.PLAY:
        # PASS always allowed
        mask[INDEX_PASS] = 1
        # SHA to targets within range
        if not state.used_sha_in_turn.get(agent, False) and any_card_named(state, me, "sha"):
            for seat in range(NUM_SEAT_SLOTS):
                target = state.agent_by_seat(seat)
                if target and target != agent and state.players[target].alive:
                    if in_sha_range(state, agent, target):
                        mask[action_index_for_sha_to_seat(seat)] = 1
        # TAO if wounded
        if any_card_named(state, me, "tao") and me.hp < me.max_hp:
            mask[INDEX_TAO] = 1
        return mask

    if phase == Phase.DISCARD:
        # discard by slot indices
        over = len(me.hand) - me.hp
        if over > 0:
            for i in range(min(NUM_DISCARD_SLOTS, len(me.hand))):
                mask[INDEX_DISCARD_BASE + i] = 1
        else:
            mask[INDEX_CONFIRM] = 1
        return mask

    mask[INDEX_PASS] = 1
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


def in_sha_range(state: GameState, attacker: str, defender: str) -> bool:
    # simplified: weapon range +1 base, no horses yet
    rng = state.players[attacker].equip_weapon_range
    # circular distance by seat
    a = state.players[attacker].seat
    d = state.players[defender].seat
    n = len(state.agent_order)
    clockwise = (d - a) % n
    counter = (a - d) % n
    dist = min(clockwise, counter)
    # horses: attacker has -1 horse reduces distance by 1; defender's +1 horse increases by 1
    if state.players[attacker].equip_minus_horse:
        dist = max(1, dist - 1)
    if state.players[defender].equip_plus_horse:
        dist = dist + 1
    return dist <= rng