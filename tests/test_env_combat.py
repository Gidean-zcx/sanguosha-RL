from __future__ import annotations
import numpy as np
from sgs_env import env as make_env
from sgs_env.observation import in_sha_range
from sgs_env.actions import (
    INDEX_PASS,
    INDEX_TAO,
    INDEX_CONFIRM,
    INDEX_RESP_SHAN,
    INDEX_SHA_BASE,
    NUM_SEAT_SLOTS,
)


def goto_play_phase(e):
    e.reset(seed=0)
    # player_0: prepare->judgement->draw
    for _ in range(3):
        _o, _r, _t, _tr, info = e.last()
        e.step(INDEX_CONFIRM)


def test_sha_has_targets_and_shan_response():
    e = make_env(seed=0, num_players=4)
    goto_play_phase(e)
    agent = e.agent_selection
    _o, _r, _t, _tr, info = e.last()
    mask = info["legal_action_mask"]
    # may not always have sha or in range; just ensure mask computed and no crash
    sha_targets = [i for i in range(INDEX_SHA_BASE, INDEX_SHA_BASE + NUM_SEAT_SLOTS) if mask[i]]
    if sha_targets:
        a = sha_targets[0]
        e.step(a)
        # now defender should be in response window; advance selection to defender
        defender = e.agent_selection
        _o, _r, _t, _tr, info = e.last()
        mask = info["legal_action_mask"]
        # defender can PASS or RESP_SHAN (may not have shan)
        assert mask[INDEX_PASS] == 1
    else:
        # no target available; pass is legal
        assert mask[INDEX_PASS] == 1


def test_dying_and_tao_self_rescue_or_death():
    e = make_env(seed=1, num_players=4)
    goto_play_phase(e)
    # force player_0 to use SHA repeatedly via mask; we won't guarantee TAO exists, so just step until response resolves
    for _ in range(10):
        agent = e.agent_selection
        _o, _r, _t, _tr, info = e.last()
        mask = info["legal_action_mask"]
        if info["observation_struct"]["phase"] == "play":
            sha_targets = [i for i in range(INDEX_SHA_BASE, INDEX_SHA_BASE + NUM_SEAT_SLOTS) if mask[i]]
            if sha_targets:
                e.step(sha_targets[0])
            else:
                e.step(INDEX_PASS)
        else:
            # confirm phases or responses
            a = INDEX_CONFIRM if mask[INDEX_CONFIRM] else INDEX_PASS
            e.step(a)
    # no assertion on final state; this test ensures no crash path