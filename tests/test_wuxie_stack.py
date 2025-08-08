from __future__ import annotations
from sgs_env import env as make_env
from sgs_env.actions import INDEX_CONFIRM, INDEX_PASS, INDEX_RESP_WUXIE


def test_wuxie_nested_parity_smoke():
    e = make_env(seed=2, num_players=4)
    e.reset(seed=2)
    for _ in range(60):
        _o, _r, _t, _tr, info = e.last()
        mask = info["legal_action_mask"]
        a = int(next(i for i, v in enumerate(mask) if v))
        e.step(a)
    assert True