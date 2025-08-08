from __future__ import annotations
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from coordinator import LocalRoom, HeadlessBatch
import asyncio

app = FastAPI(title="SGS PettingZoo API")


class CreateRoomReq(BaseModel):
    seed: int = 0
    num_players: int = 4
    record: bool = True


@app.post("/rooms")
def create_room(req: CreateRoomReq):
    room = LocalRoom(seed=req.seed, num_players=req.num_players, record=req.record)
    game_id = room.run_episode(max_steps=100)
    return {"game_id": game_id}


class HeadlessReq(BaseModel):
    seed: int = 0
    num_players: int = 4
    record: bool = True
    num_episodes: int = 8
    parallelism: int = 4
    max_steps: int = 200


@app.post("/headless/run")
def headless_run(req: HeadlessReq):
    hb = HeadlessBatch(seed=req.seed, num_players=req.num_players, record=req.record)
    ids = hb.run(num_episodes=req.num_episodes, parallelism=req.parallelism, max_steps=req.max_steps)
    return {"game_ids": ids}


@app.websocket("/ws/headless")
async def ws_headless(ws: WebSocket):
    await ws.accept()
    # stream a single headless episode events by stepping LocalRoom and pushing last().infos
    room = LocalRoom(seed=0, num_players=4, record=False)
    from sgs_env import env as make_env
    e = make_env(seed=0, num_players=4)
    e.reset(seed=0)
    step = 0
    while step < 50:
        agent = e.agent_selection
        _o, _r, _t, _tr, info = e.last()
        await ws.send_json({"agent": agent, "info": {
            "mask": info.get("legal_action_mask").tolist(),
            "events": info.get("events", []),
            "phase": info.get("observation_struct", {}).get("phase"),
        }})
        # auto random legal action
        import numpy as np
        mask = info.get("legal_action_mask")
        legal = [i for i, v in enumerate(mask) if v]
        a = int(np.random.default_rng(0).choice(legal)) if legal else 0
        e.step(a)
        step += 1
    await ws.close()