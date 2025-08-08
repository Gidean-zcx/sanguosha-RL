from __future__ import annotations
from sgs_env import env as make_env
from sgs_env.actions import INDEX_CONFIRM, INDEX_PASS, INDEX_SHA_BASE, NUM_SEAT_SLOTS


def play_until_end(e, max_steps=500):
    steps = 0
    while steps < max_steps and not (all(e.terminations.values()) or all(e.truncations.values())):
        _o, _r, _t, _tr, info = e.last()
        mask = info["legal_action_mask"]
        # priority: sha if can, else confirm if available, else pass
        a = None
        for i in range(INDEX_SHA_BASE, INDEX_SHA_BASE + NUM_SEAT_SLOTS):
            if mask[i]:
                a = i
                break
        if a is None:
            if mask[INDEX_CONFIRM]:
                a = INDEX_CONFIRM
            else:
                a = INDEX_PASS if mask[INDEX_PASS] else int(next(i for i,v in enumerate(mask) if v))
        e.step(a)
        steps += 1
    return steps


def test_full_game_smoke_1():
    e = make_env(seed=10, num_players=4)
    e.reset(seed=10)
    steps = play_until_end(e, max_steps=800)
    assert steps > 0


def test_full_game_smoke_2():
    e = make_env(seed=11, num_players=4)
    e.reset(seed=11)
    steps = play_until_end(e, max_steps=800)
    assert steps > 0