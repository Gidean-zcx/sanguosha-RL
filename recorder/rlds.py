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

    def _path_for_game(self, game_id: str) -> str:
        date_dir = datetime.utcnow().strftime("%Y%m%d")
        dirp = os.path.join(self.root, date_dir)
        os.makedirs(dirp, exist_ok=True)
        return os.path.join(dirp, f"{game_id}.rlds.jsonl")

    def append_step(self, game_id: str, rec: StepRecord) -> None:
        path = self._path_for_game(game_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")