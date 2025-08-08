from __future__ import annotations
from typing import Dict, List, Tuple
import uuid

from sgs_env import env as make_env
from recorder import RLDSRecorder, StepRecord
from agents import RandomLegalBot


class LocalRoom:
    def __init__(self, seed: int = 0, num_players: int = 4, record: bool = True) -> None:
        self.seed = seed
        self.num_players = num_players
        self.record = record
        self.game_id = str(uuid.uuid4())
        self.recorder = RLDSRecorder()

    def run_episode(self, max_steps: int = 200) -> str:
        e = make_env(seed=self.seed, num_players=self.num_players)
        e.reset(seed=self.seed)
        bots = {agent: RandomLegalBot(seed=self.seed + i) for i, agent in enumerate(e.agents)}
        step_idx = 0
        terminated_all = False
        while step_idx < max_steps and not terminated_all:
            agent = e.agent_selection
            obs, rew, term, trunc, info = e.last()
            mask = info.get("legal_action_mask")
            a = bots[agent].act(mask)
            if self.record:
                rec = StepRecord(
                    episode_id=self.game_id,
                    step=step_idx,
                    agent=agent,
                    observation=info.get("observation_struct"),
                    action=int(a),
                    reward=float(rew),
                    terminated=bool(term),
                    truncated=bool(trunc),
                    info={"rng": info.get("rng"), "events": info.get("events", [])},
                )
                self.recorder.append_step(self.game_id, rec)
            e.step(a)
            step_idx += 1
            terminated_all = all(e.terminations.values()) or all(e.truncations.values())
        return self.game_id