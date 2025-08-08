from __future__ import annotations
from typing import Optional

# Layout (fixed):
# 0: PASS
# 1..8: PLAY_SHA to seat 0..7 (mask invalid or self)
# 9: PLAY_TAO (self only)
# 10: CONFIRM
# 11: RESP_SHAN
# 32..63: DISCARD hand slot 0..31

INDEX_PASS = 0
INDEX_SHA_BASE = 1
NUM_SEAT_SLOTS = 8
INDEX_TAO = 9
INDEX_CONFIRM = 10
INDEX_RESP_SHAN = 11
INDEX_DISCARD_BASE = 32
NUM_DISCARD_SLOTS = 32


def action_index_for_sha_to_seat(seat_index: int) -> int:
    return INDEX_SHA_BASE + seat_index


def seat_from_sha_action_index(action_index: int) -> Optional[int]:
    if INDEX_SHA_BASE <= action_index < INDEX_SHA_BASE + NUM_SEAT_SLOTS:
        return action_index - INDEX_SHA_BASE
    return None


def action_index_for_discard_slot(slot_index: int) -> int:
    return INDEX_DISCARD_BASE + slot_index


def discard_slot_from_action_index(action_index: int) -> Optional[int]:
    if INDEX_DISCARD_BASE <= action_index < INDEX_DISCARD_BASE + NUM_DISCARD_SLOTS:
        return action_index - INDEX_DISCARD_BASE
    return None