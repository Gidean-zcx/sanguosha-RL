from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import uuid
import numpy as np

from sgs_env import env as make_env
from recorder import RLDSRecorder, StepRecord
from agents import RandomLegalBot
from agents.llm_adapter import LLMAdapter


class LocalRoom:
    def __init__(self, seed: int = 0, num_players: int = 4, record: bool = True) -> None:
        self.seed = seed
        self.num_players = num_players
        self.record = record
        self.game_id = str(uuid.uuid4())
        self.recorder = RLDSRecorder()
        # seats config: seat index -> dict(kind, provider, model)
        self.seats: Dict[int, Dict[str, Optional[str]]] = {}

    def run_episode(self, max_steps: int = 200) -> str:
        e = make_env(seed=self.seed, num_players=self.num_players)
        e.reset(seed=self.seed)
        # default: random bots
        bots = {agent: RandomLegalBot(seed=self.seed + i) for i, agent in enumerate(e.agents)}
        step_idx = 0
        terminated_all = False
        rng = np.random.default_rng(self.seed + 999)
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
                # DPO pair (sample a rejected different legal action if exists)
                legal_idxs = [i for i, v in enumerate(mask) if v]
                if len(legal_idxs) > 1:
                    rejected = int(rng.choice([i for i in legal_idxs if i != a]))
                    self.recorder.append_dpo_pair({
                        "episode_id": self.game_id,
                        "step": step_idx,
                        "agent": agent,
                        "chosen": {"action": int(a)},
                        "rejected": {"action": int(rejected)},
                        "meta": {"rng": info.get("rng")},
                    })
                # GRPO group (top up to 4 legal actions)
                if len(legal_idxs) > 1:
                    choices = legal_idxs[:4]
                    group = {
                        "episode_id": self.game_id,
                        "step": step_idx,
                        "agent": agent,
                        "group": [
                            {"action": int(x), "reward": 0.0, "logprob": None}
                            for x in choices
                        ],
                    }
                    self.recorder.append_grpo_group(group)
            e.step(a)
            step_idx += 1
            terminated_all = all(e.terminations.values()) or all(e.truncations.values())
        return self.game_id


class HeadlessBatch:
    def __init__(self, seed: int = 0, num_players: int = 4, record: bool = True) -> None:
        self.seed = seed
        self.num_players = num_players
        self.record = record

    def run(self, num_episodes: int = 8, parallelism: int = 4, max_steps: int = 200) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        game_ids: List[str] = []
        with ThreadPoolExecutor(max_workers=parallelism) as ex:
            futs = [ex.submit(LocalRoom(self.seed + i, self.num_players, self.record).run_episode, max_steps) for i in range(num_episodes)]
            for f in as_completed(futs):
                game_ids.append(f.result())
        return game_ids


class RoomCoordinator:
    def __init__(self) -> None:
        self.rooms: Dict[str, LocalRoom] = {}

    def create_room(self, seed: int = 0, num_players: int = 4, record: bool = True) -> str:
        room = LocalRoom(seed=seed, num_players=num_players, record=record)
        self.rooms[room.game_id] = room
        return room.game_id

    def get_room(self, game_id: str) -> LocalRoom:
        return self.rooms[game_id]

    def join(self, game_id: str, seat: int, kind: str = "human", provider: Optional[str] = None, model: Optional[str] = None) -> bool:
        room = self.get_room(game_id)
        if seat < 0 or seat >= room.num_players:
            return False
        room.seats[seat] = {"kind": kind, "provider": provider, "model": model}
        return True