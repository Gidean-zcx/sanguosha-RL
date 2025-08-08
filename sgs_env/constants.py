from __future__ import annotations
from enum import Enum, IntEnum, auto


class Phase(Enum):
    PREPARE = "prepare"
    JUDGEMENT = "judgement"
    DRAW = "draw"
    PLAY = "play"
    DISCARD = "discard"
    END = "end"


class Role(Enum):
    LORD = "lord"
    LOYALIST = "loyalist"
    REBEL = "rebel"
    RENEGADE = "renegade"


class Suit(Enum):
    SPADE = "spade"
    HEART = "heart"
    CLUB = "club"
    DIAMOND = "diamond"


class Action(IntEnum):
    # meta
    PASS = 0
    # play-phase basic actions
    PLAY_SHA = 1
    PLAY_TAO = 2
    PLAY_WUXIE = 3
    PLAY_JUEDOU = 4
    PLAY_GUOHE = 5
    PLAY_SHUNSHOU = 6
    PLAY_NANMAN = 7
    PLAY_WANJIAN = 8
    PLAY_LE = 9
    PLAY_BINGLIANG = 10
    PLAY_SHANDIAN = 11
    # responses
    RESP_SHAN = 20
    RESP_WUXIE = 21
    # equipment
    EQUIP_WEAPON = 30
    EQUIP_HORSE_MINUS = 31
    EQUIP_HORSE_PLUS = 32
    # phase control
    DISCARD_CARD = 40
    CONFIRM = 41


# Fixed action space size for mask approach
ACTION_SPACE_SIZE: int = 64

# Simplified card ids (only minimal needed now)
BASIC_CARDS = {
    "sha",
    "shan",
    "tao",
}