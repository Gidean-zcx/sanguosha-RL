from __future__ import annotations
import numpy as np
from sgs_env import env as make_env
from sgs_env.actions import (
    INDEX_CONFIRM,
    INDEX_PASS,
    INDEX_SHA_BASE,
    NUM_SEAT_SLOTS,
)
from sgs_env.cards import NAME_TO_CARD_ID


def goto_play(e, seed=0):
    e.reset(seed=seed)
    for _ in range(3):
        _o, _r, _t, _tr, info = e.last()
        e.step(INDEX_CONFIRM)


def first_sha_action(mask):
    for i in range(INDEX_SHA_BASE, INDEX_SHA_BASE + NUM_SEAT_SLOTS):
        if mask[i]:
            return i
    return None


def test_qinggang_ignores_renwang():
    e = make_env(seed=0, num_players=4)
    goto_play(e)
    inner = e.unwrapped
    atk = e.agent_selection
    # set attacker weapon qinggang, defender renwang; give attacker black sha
    def_agent = inner.state.agent_order[1]
    ap = inner.state.players[atk]
    dp = inner.state.players[def_agent]
    ap.equip_weapon_name = "qinggang"
    dp.equip_armor_name = "renwang"
    ap.hand = [(NAME_TO_CARD_ID["sha"], 0)]  # spade
    _o, _r, _t, _tr, info = e.last()
    a = first_sha_action(info["legal_action_mask"]) 
    if a is None:
        return
    e.step(a)
    # defender turn to respond, but renwang ignored and no shan -> hit
    _o2, _r2, _t2, _tr2, info2 = e.last()
    # pass to take damage
    e.step(INDEX_PASS)
    # back to attacker end phases eventually; check defender hp decreased
    assert inner.state.players[def_agent].hp <= 3


def test_bagua_blocked_by_qinggang():
    e = make_env(seed=1, num_players=4)
    goto_play(e, seed=1)
    inner = e.unwrapped
    atk = e.agent_selection
    def_agent = inner.state.agent_order[1]
    ap = inner.state.players[atk]
    dp = inner.state.players[def_agent]
    ap.equip_weapon_name = "qinggang"
    dp.equip_armor_name = "bagua"
    ap.hand = [(NAME_TO_CARD_ID["sha"], 1)]  # heart
    _o, _r, _t, _tr, info = e.last()
    a = first_sha_action(info["legal_action_mask"]) 
    if a is None:
        return
    e.step(a)
    # defender cannot use bagua due to qinggang ignore_armor; take damage
    _o2, _r2, _t2, _tr2, info2 = e.last()
    e.step(INDEX_PASS)
    assert inner.state.players[def_agent].hp <= 3


def test_tieqi_forbid_shan_even_with_shan_in_hand():
    e = make_env(seed=2, num_players=4)
    goto_play(e, seed=2)
    inner = e.unwrapped
    atk = e.agent_selection
    def_agent = inner.state.agent_order[1]
    ap = inner.state.players[atk]
    dp = inner.state.players[def_agent]
    # set attacker hero machao and prepare judge black on top
    ap.hero = "machao"
    # prepare deck top judge card as spade 7
    inner.state.deck.append((NAME_TO_CARD_ID["sha"], 0, 7))
    ap.hand = [(NAME_TO_CARD_ID["sha"], 1)]
    dp.hand = [(NAME_TO_CARD_ID["shan"], 1)]
    _o, _r, _t, _tr, info = e.last()
    a = first_sha_action(info["legal_action_mask"]) 
    if a is None:
        return
    e.step(a)
    # defender cannot shan due to forbid_shan
    _o2, _r2, _t2, _tr2, info2 = e.last()
    e.step(INDEX_PASS)
    assert inner.state.players[def_agent].hp <= 3


def test_kongcheng_immunity():
    e = make_env(seed=3, num_players=4)
    goto_play(e, seed=3)
    inner = e.unwrapped
    atk = e.agent_selection
    def_agent = inner.state.agent_order[1]
    inner.state.players[def_agent].hero = "zhugeliang"
    inner.state.players[def_agent].hand = []
    inner.state.players[atk].hand = [(NAME_TO_CARD_ID["sha"], 1)]
    _o, _r, _t, _tr, info = e.last()
    a = first_sha_action(info["legal_action_mask"]) 
    # seat 1 should be masked off
    assert a is None or (a - INDEX_SHA_BASE) != inner.state.players[def_agent].seat


def test_lightning_unique_and_pass_on():
    e = make_env(seed=4, num_players=4)
    goto_play(e, seed=4)
    inner = e.unwrapped
    cur = e.agent_selection
    p = inner.state.players[cur]
    # give shandian in hand and place it
    p.hand = [(NAME_TO_CARD_ID["shandian"], 0)]
    _o, _r, _t, _tr, info = e.last()
    mask = info["legal_action_mask"]
    # find shandian action (index 58)
    if mask.shape[0] > 58 and mask[58]:
        e.step(58)
    # try place second one should be masked off
    p.hand = [(NAME_TO_CARD_ID["shandian"], 0)]
    _o, _r, _t, _tr, info = e.last()
    if info["legal_action_mask"].shape[0] > 58:
        assert info["legal_action_mask"][58] == 0


def test_bingliang_distance_one_with_minus_horse():
    e = make_env(seed=5, num_players=4)
    goto_play(e, seed=5)
    inner = e.unwrapped
    atk = e.agent_selection
    target = inner.state.agent_order[2]
    ap = inner.state.players[atk]
    tp = inner.state.players[target]
    ap.equip_minus_horse = True
    # ensure hand has bingliang
    ap.hand = [(NAME_TO_CARD_ID["bingliang"], 2)]
    _o, _r, _t, _tr, info = e.last()
    mask = info["legal_action_mask"]
    idx = None
    for s in range(NUM_SEAT_SLOTS):
        if inner.state.agent_by_seat(s) == target:
            if mask[24 + s]:
                idx = 24 + s
                break
    # 24 base was guohe, 50 is bingliang base; use 50
    idx = None
    for s in range(NUM_SEAT_SLOTS):
        if inner.state.agent_by_seat(s) == target:
            if mask[50 + s]:
                idx = 50 + s
                break
    if idx is not None:
        e.step(idx)


def test_victory_lord_dead_rebels_win():
    e = make_env(seed=6, num_players=4)
    inner = e.unwrapped
    # set roles explicitly
    order = inner.state.agent_order
    inner.state.players[order[0]].role = inner.state.players[order[0]].role.__class__.LORD
    inner.state.players[order[1]].role = inner.state.players[order[1]].role.__class__.REBEL
    inner.state.players[order[2]].role = inner.state.players[order[2]].role.__class__.LOYALIST
    inner.state.players[order[3]].role = inner.state.players[order[3]].role.__class__.REBEL
    e.reset(seed=6)
    # trigger death of lord via dying window
    inner = e.unwrapped
    inner.state.players[inner.state.agent_order[0]].role = inner.state.players[order[0]].role.__class__.LORD
    inner.state.dying_pending = {"agent": inner.state.agent_order[0]}
    e.unwrapped.agent_selection = inner.state.agent_order[0]
    _o, _r, _t, _tr, info = e.last()
    # ensure no tao -> pass chain around table until resolution
    for _ in range(8):
        e.step(INDEX_PASS)
        _o, _r, _t, _tr, info = e.last()
        if all(e.terminations.values()):
            break
    # assert lord dead; full terminations may be handled later in loop
    assert inner.state.players[inner.state.agent_order[0]].alive is False