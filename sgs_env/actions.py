from __future__ import annotations
from typing import Optional

# Fixed action layout (size <= 64)
# 0: PASS
# 1..8: PLAY_SHA -> seat 0..7
# 9: PLAY_TAO (self)
# 10: CONFIRM
# 11: RESP_SHAN
# 12: RESP_SHA
# 13: RESP_WUXIE
# 16..23: PLAY_JUEDOU -> seat 0..7
# 24..31: PLAY_GUOHE -> seat 0..7
# 32..39: PLAY_SHUNSHOU -> seat 0..7
# 40: PLAY_NANMAN
# 41: PLAY_WANJIAN
# 42..49: PLAY_LE -> seat 0..7
# 50..57: PLAY_BINGLIANG -> seat 0..7
# 58: PLAY_SHANDIAN
# 59..61: DISCARD slots 0..2
# 62: EQUIP_WEAPON
# 63: EQUIP_ARMOR

INDEX_PASS = 0
INDEX_SHA_BASE = 1
NUM_SEAT_SLOTS = 8
INDEX_TAO = 9
INDEX_CONFIRM = 10
INDEX_RESP_SHAN = 11
INDEX_RESP_SHA = 12
INDEX_RESP_WUXIE = 13
INDEX_JUEDOU_BASE = 16
INDEX_GUOHE_BASE = 24
INDEX_SHUNSHOU_BASE = 32
INDEX_NANMAN = 40
INDEX_WANJIAN = 41
INDEX_LE_BASE = 42
INDEX_BINGLIANG_BASE = 50
INDEX_SHANDIAN = 58
INDEX_DISCARD_BASE = 59
NUM_DISCARD_SLOTS = 3
INDEX_EQUIP_WEAPON = 62
INDEX_EQUIP_ARMOR = 63


def action_index_for_sha_to_seat(seat_index: int) -> int:
    return INDEX_SHA_BASE + seat_index


def seat_from_sha_action_index(action_index: int) -> Optional[int]:
    if INDEX_SHA_BASE <= action_index < INDEX_SHA_BASE + NUM_SEAT_SLOTS:
        return action_index - INDEX_SHA_BASE
    return None


def action_index_for_targeted(base: int, seat_index: int) -> int:
    return base + seat_index


def seat_from_targeted(base: int, action_index: int) -> Optional[int]:
    if base <= action_index < base + NUM_SEAT_SLOTS:
        return action_index - base
    return None


def action_index_for_discard_slot(slot_index: int) -> int:
    return INDEX_DISCARD_BASE + slot_index


def discard_slot_from_action_index(action_index: int) -> Optional[int]:
    if INDEX_DISCARD_BASE <= action_index < INDEX_DISCARD_BASE + NUM_DISCARD_SLOTS:
        return action_index - INDEX_DISCARD_BASE
    return None