from __future__ import annotations
from sgs_env import env as make_env
from sgs_env.actions import INDEX_CONFIRM, INDEX_PASS


def fast_forward_to_play(e):
    e.reset(seed=0)
    for _ in range(3):
        _o, _r, _t, _tr, info = e.last()
        e.step(INDEX_CONFIRM)


def test_game_smoke_runs_to_some_end():
    e = make_env(seed=3, num_players=4)
    fast_forward_to_play(e)
    steps = 0
    while steps < 200 and not (all(e.terminations.values()) or all(e.truncations.values())):
        _o, _r, _t, _tr, info = e.last()
        mask = info["legal_action_mask"]
        # choose first legal action
        a = int(next(i for i, v in enumerate(mask) if v))
        e.step(a)
        steps += 1
    assert steps > 0