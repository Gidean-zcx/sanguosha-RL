from __future__ import annotations
import numpy as np
from sgs_env import env as make_env
from sgs_env.constants import Phase
from sgs_env.actions import INDEX_CONFIRM, INDEX_PASS, INDEX_DISCARD_BASE, NUM_DISCARD_SLOTS


def test_env_reset_and_iterates():
    e = make_env(seed=123, num_players=4)
    e.reset(seed=123)
    assert len(e.agents) == 4
    agent = e.agent_selection
    obs, rew, term, trunc, info = e.last()
    assert "legal_action_mask" in info
    assert info["legal_action_mask"].shape[0] >= 64


def step_until_phase(e, phase: str, limit=200):
    # ensure started
    try:
        _ = e.agent_selection
    except Exception:
        e.reset()
    for _ in range(limit):
        agent = e.agent_selection
        _obs, _rew, _term, _trunc, info = e.last()
        if info["observation_struct"]["phase"] == phase:
            return agent
        # take confirm by default
        mask = info["legal_action_mask"]
        action = int(INDEX_CONFIRM) if mask[int(INDEX_CONFIRM)] else int(INDEX_PASS)
        e.step(action)
    raise RuntimeError("phase not reached")


def test_discard_to_hp_enforced():
    e = make_env(seed=42, num_players=4)
    e.reset(seed=42)
    # Fast-forward to player_0's discard phase
    agent = step_until_phase(e, phase="discard")
    assert agent == e.agent_selection
    _obs, _rew, _term, _trunc, info = e.last()
    me = info["observation_struct"]["self"]
    # Force a state where hand > hp by stepping draw+play minimal
    # If already over limit, mask should require discard
    over = len(me["hand"]) - me["hp"]
    mask = info["legal_action_mask"]
    if over > 0:
        # at least one discard slot should be available when over limit
        from sgs_env.actions import INDEX_DISCARD_BASE, NUM_DISCARD_SLOTS
        assert any(mask[INDEX_DISCARD_BASE + i] for i in range(NUM_DISCARD_SLOTS))
        # Do a few discards until allowed to confirm
        for _ in range(over):
            # discard slot 0 if available else first available
            idx = next((INDEX_DISCARD_BASE + i for i in range(NUM_DISCARD_SLOTS) if mask[INDEX_DISCARD_BASE + i]), None)
            assert idx is not None
            e.step(idx)
            _obs, _rew, _term, _trunc, info = e.last()
            mask = info["legal_action_mask"]
        assert mask[int(INDEX_CONFIRM)] == 1
    else:
        # If not over, should be able to confirm
        assert mask[int(INDEX_CONFIRM)] == 1