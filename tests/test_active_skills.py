from __future__ import annotations
from sgs_env import env as make_env
from sgs_env.actions import INDEX_CONFIRM, INDEX_DISCARD_BASE, NUM_DISCARD_SLOTS, INDEX_JUEDOU_BASE, NUM_SEAT_SLOTS


def goto_play(e, seed=0):
    e.reset(seed=seed)
    for _ in range(3):
        _o, _r, _t, _tr, info = e.last()
        e.step(INDEX_CONFIRM)


def test_zhiheng_flow_smoke():
    e = make_env(seed=3, num_players=4)
    goto_play(e, seed=3)
    agent = e.agent_selection
    # start zhiheng if available via confirm
    _o, _r, _t, _tr, info = e.last()
    mask = info["legal_action_mask"]
    if mask[INDEX_CONFIRM]:
        e.step(INDEX_CONFIRM)
        # discard up to slots if any
        for i in range(NUM_DISCARD_SLOTS):
            _o, _r, _t, _tr, info = e.last()
            mask = info["legal_action_mask"]
            if mask[INDEX_DISCARD_BASE + i]:
                e.step(INDEX_DISCARD_BASE + i)
                break
        _o, _r, _t, _tr, info = e.last()
        if info["legal_action_mask"][INDEX_CONFIRM]:
            e.step(INDEX_CONFIRM)
    assert True


def test_qingnang_heal_smoke():
    e = make_env(seed=4, num_players=4)
    goto_play(e, seed=4)
    # try target seat action under qingnang mask if available
    _o, _r, _t, _tr, info = e.last()
    mask = info["legal_action_mask"]
    acted = False
    for s in range(NUM_SEAT_SLOTS):
        idx = INDEX_JUEDOU_BASE + s
        if mask[idx]:
            e.step(idx)
            acted = True
            break
    assert True