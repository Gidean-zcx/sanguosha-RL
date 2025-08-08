from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json
import os
from datetime import datetime


@dataclass
class StepRecord:
    episode_id: str
    step: int
    agent: str
    observation: Any
    action: int
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class RLDSRecorder:
    def __init__(self, root: str = "replays") -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _date_dir(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d")

    def _path_for_game(self, game_id: str) -> str:
        date_dir = self._date_dir()
        dirp = os.path.join(self.root, date_dir)
        os.makedirs(dirp, exist_ok=True)
        return os.path.join(dirp, f"{game_id}.rlds.jsonl")

    def _path_for_dpo(self) -> str:
        date_dir = self._date_dir()
        dirp = os.path.join(self.root, date_dir)
        os.makedirs(dirp, exist_ok=True)
        return os.path.join(dirp, "dpo_pairs.jsonl")

    def _path_for_grpo(self) -> str:
        date_dir = self._date_dir()
        dirp = os.path.join(self.root, date_dir)
        os.makedirs(dirp, exist_ok=True)
        return os.path.join(dirp, "grpo_groups.jsonl")

    def append_step(self, game_id: str, rec: StepRecord) -> None:
        path = self._path_for_game(game_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def append_dpo_pair(self, pair: Dict[str, Any]) -> None:
        path = self._path_for_dpo()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    def append_grpo_group(self, group: Dict[str, Any]) -> None:
        path = self._path_for_grpo()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(group, ensure_ascii=False) + "\n")