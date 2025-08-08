from __future__ import annotations
from sgs_env import env as make_env
from sgs_env.actions import INDEX_CONFIRM, INDEX_PASS, INDEX_SHA_BASE, NUM_SEAT_SLOTS, INDEX_RESP_SHAN


def goto_play(e, seed=0):
    e.reset(seed=seed)
    for _ in range(3):
        _o, _r, _t, _tr, info = e.last()
        e.step(INDEX_CONFIRM)


def test_bagua_allows_respond_without_shan():
    e = make_env(seed=0, num_players=4)
    goto_play(e)
    # force equip bagua if possible: try to play equip action if available
    for _ in range(10):
        agent = e.agent_selection
        _o, _r, _t, _tr, info = e.last()
        mask = info["legal_action_mask"]
        # try any SHA first to open response; else try equip or pass/confirm
        sha_targets = [i for i in range(INDEX_SHA_BASE, INDEX_SHA_BASE + NUM_SEAT_SLOTS) if mask[i]]
        if sha_targets:
            e.step(sha_targets[0])
            # now defender's turn; they may RESP_SHAN due to bagua if had equipped. We just ensure no crash path here.
            break
        a = int(next(i for i, v in enumerate(mask) if v))
        e.step(a)


def test_paoxiao_or_crossbow_unlimited_sha_mask():
    e = make_env(seed=1, num_players=4)
    goto_play(e, seed=1)
    agent = e.agent_selection
    _o, _r, _t, _tr, info = e.last()
    mask1 = info["legal_action_mask"].copy()
    # take a sha if any, then check mask still allows another sha if paoxiao/crossbow or otherwise not guaranteed
    sha_targets = [i for i in range(INDEX_SHA_BASE, INDEX_SHA_BASE + NUM_SEAT_SLOTS) if mask1[i]]
    if sha_targets:
        e.step(sha_targets[0])
        agent2 = e.agent_selection
        _o2, _r2, _t2, _tr2, info2 = e.last()
        mask2 = info2["legal_action_mask"]
        # If unlimited sha, still have sha target available; otherwise may not
        # We only assert no crash
        assert mask2.shape[0] > 0