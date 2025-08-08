from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from .constants import ACTION_SPACE_SIZE, Phase
from .game_state import GameState, PlayerState
from .actions import (
    INDEX_PASS,
    INDEX_TAO,
    INDEX_CONFIRM,
    INDEX_RESP_SHAN,
    INDEX_RESP_SHA,
    INDEX_RESP_WUXIE,
    INDEX_SHA_BASE,
    NUM_SEAT_SLOTS,
    INDEX_DISCARD_BASE,
    NUM_DISCARD_SLOTS,
    action_index_for_sha_to_seat,
    action_index_for_targeted,
    seat_from_targeted,
    INDEX_JUEDOU_BASE,
    INDEX_GUOHE_BASE,
    INDEX_SHUNSHOU_BASE,
    INDEX_NANMAN,
    INDEX_WANJIAN,
    INDEX_LE_BASE,
    INDEX_BINGLIANG_BASE,
    INDEX_SHANDIAN,
    INDEX_EQUIP_WEAPON,
    INDEX_EQUIP_ARMOR,
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

    # Response windows override normal turns
    if state.response_pending:
        typ = state.response_pending.get("type")
        target = state.response_pending.get("target")
        if typ == "shan" and target == agent:
            mask[INDEX_PASS] = 1
            if any_card_named(state, me, "shan"):
                mask[INDEX_RESP_SHAN] = 1
            return mask
        if typ == "sha" and target == agent:
            mask[INDEX_PASS] = 1
            if any_card_named(state, me, "sha"):
                mask[INDEX_RESP_SHA] = 1
            return mask
        if typ == "wuxie" and state.response_pending.get("current") == agent:
            mask[INDEX_PASS] = 1
            if any_card_named(state, me, "wuxie"):
                mask[INDEX_RESP_WUXIE] = 1
            return mask

    # Dying window
    if state.dying_pending and state.dying_pending.get("agent") == agent:
        mask[INDEX_PASS] = 1
        if any_card_named(state, me, "tao"):
            mask[INDEX_TAO] = 1
        return mask

    # Normal turn check
    if state.agent_order[state.current_agent_idx] != agent:
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
        mask[INDEX_PASS] = 1
        # unlimited sha via hero/equip
        unlimited_sha = state.players[agent].hero == "zhangfei" or state.players[agent].equip_weapon_name == "crossbow"
        can_sha = any_card_named(state, me, "sha") or (state.players[agent].hero == "guanyu" and any_red_card(me)) or (state.players[agent].hero == "zhaoyun" and any_card_named(state, me, "shan"))
        if (unlimited_sha or not state.used_sha_in_turn.get(agent, False)) and can_sha:
            for seat in range(NUM_SEAT_SLOTS):
                target = state.agent_by_seat(seat)
                if target and target != agent and state.players[target].alive:
                    # kongcheng target immunity
                    if state.players[target].hero == "zhugeliang" and len(state.players[target].hand) == 0:
                        continue
                    if in_sha_range(state, agent, target):
                        mask[action_index_for_sha_to_seat(seat)] = 1
        # TAO
        if any_card_named(state, me, "tao") and me.hp < me.max_hp:
            mask[INDEX_TAO] = 1
        # JUEDOU
        if any_card_named(state, me, "juedou"):
            for seat in range(NUM_SEAT_SLOTS):
                target = state.agent_by_seat(seat)
                if target and target != agent and state.players[target].alive:
                    mask[action_index_for_targeted(INDEX_JUEDOU_BASE, seat)] = 1
        # GUOHE (range: any)
        if any_card_named(state, me, "guohe"):
            for seat in range(NUM_SEAT_SLOTS):
                target = state.agent_by_seat(seat)
                if target and target != agent and state.players[target].alive:
                    mask[action_index_for_targeted(INDEX_GUOHE_BASE, seat)] = 1
        # SHUNSHOU (distance 1)
        if any_card_named(state, me, "shunshou"):
            for seat in range(NUM_SEAT_SLOTS):
                target = state.agent_by_seat(seat)
                if target and target != agent and state.players[target].alive:
                    if in_distance_one(state, agent, target):
                        mask[action_index_for_targeted(INDEX_SHUNSHOU_BASE, seat)] = 1
        # NANMAN & WANJIAN
        if any_card_named(state, me, "nanman"):
            mask[INDEX_NANMAN] = 1
        if any_card_named(state, me, "wanjian"):
            mask[INDEX_WANJIAN] = 1
        # Delayed: LE/BINGLIANG to targets; SHANDIAN to self
        if any_card_named(state, me, "le"):
            for seat in range(NUM_SEAT_SLOTS):
                target = state.agent_by_seat(seat)
                if target and target != agent and state.players[target].alive:
                    mask[action_index_for_targeted(INDEX_LE_BASE, seat)] = 1
        if any_card_named(state, me, "bingliang"):
            for seat in range(NUM_SEAT_SLOTS):
                target = state.agent_by_seat(seat)
                if target and target != agent and state.players[target].alive:
                    mask[action_index_for_targeted(INDEX_BINGLIANG_BASE, seat)] = 1
        if any_card_named(state, me, "shandian"):
            mask[INDEX_SHANDIAN] = 1
        # Equip actions
        if any_card_named(state, me, "crossbow"):
            mask[INDEX_EQUIP_WEAPON] = 1
        if any_card_named(state, me, "bagua") or any_card_named(state, me, "renwang") or any_card_named(state, me, "minus_horse") or any_card_named(state, me, "plus_horse"):
            mask[INDEX_EQUIP_ARMOR] = 1
        return mask

    if phase == Phase.DISCARD:
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


def any_red_card(me: PlayerState) -> bool:
    for _cid, suit in me.hand:
        if suit in (1, 3):  # hearts/diamonds
            return True
    return False


_CARD_ID_TO_NAME = {
    1: "sha",
    2: "shan",
    3: "tao",
    4: "juedou",
    5: "guohe",
    6: "shunshou",
    7: "nanman",
    8: "wanjian",
    9: "wuxie",
    10: "le",
    11: "bingliang",
    12: "shandian",
    13: "crossbow",
    14: "bagua",
    15: "renwang",
    16: "minus_horse",
    17: "plus_horse",
}


def card_name(card_id: int) -> str:
    return _CARD_ID_TO_NAME.get(card_id, f"unknown:{card_id}")


def in_sha_range(state: GameState, attacker: str, defender: str) -> bool:
    rng = state.players[attacker].equip_weapon_range
    a = state.players[attacker].seat
    d = state.players[defender].seat
    n = len(state.agent_order)
    clockwise = (d - a) % n
    counter = (a - d) % n
    dist = min(clockwise, counter)
    if state.players[attacker].equip_minus_horse:
        dist = max(1, dist - 1)
    if state.players[defender].equip_plus_horse:
        dist = dist + 1
    return dist <= rng


def in_distance_one(state: GameState, attacker: str, defender: str) -> bool:
    # distance one check for shunshou
    a = state.players[attacker].seat
    d = state.players[defender].seat
    n = len(state.agent_order)
    clockwise = (d - a) % n
    counter = (a - d) % n
    dist = min(clockwise, counter)
    if state.players[attacker].equip_minus_horse:
        dist = max(1, dist - 1)
    if state.players[defender].equip_plus_horse:
        dist = dist + 1
    return dist <= 1