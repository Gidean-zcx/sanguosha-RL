from __future__ import annotations
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from coordinator import LocalRoom, HeadlessBatch, RoomCoordinator
import asyncio
from agents import RandomLegalBot
from agents.llm_adapter import LLMAdapter

app = FastAPI(title="SGS PettingZoo API")

rc = RoomCoordinator()


class CreateRoomReq(BaseModel):
    seed: int = 0
    num_players: int = 4
    record: bool = True


@app.post("/rooms")
def create_room(req: CreateRoomReq):
    room_id = rc.create_room(seed=req.seed, num_players=req.num_players, record=req.record)
    return {"game_id": room_id}


class JoinReq(BaseModel):
    game_id: str
    seat: int
    kind: str = "human"  # human/llm/bot


@app.post("/rooms/join")
def join_room(req: JoinReq):
    # MVP: no persistence of joiners, just acknowledge
    return {"ok": True}


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


@app.websocket("/ws/game/{game_id}")
async def ws_game(ws: WebSocket, game_id: str):
    await ws.accept()
    room = rc.get_room(game_id)
    from sgs_env import env as make_env
    e = make_env(seed=room.seed, num_players=room.num_players)
    e.reset(seed=room.seed)
    bots = {agent: RandomLegalBot(seed=room.seed + i) for i, agent in enumerate(e.agents)}
    step = 0
    while step < 200 and not (all(e.terminations.values()) or all(e.truncations.values())):
        agent = e.agent_selection
        _o, _r, _t, _tr, info = e.last()
        await ws.send_json({
            "agent": agent,
            "info": {
                "mask": info.get("legal_action_mask").tolist(),
                "events": info.get("events", []),
                "phase": info.get("observation_struct", {}).get("phase"),
            },
        })
        # wait for client action with timeout; fallback to bot
        try:
            msg = await asyncio.wait_for(ws.receive_json(), timeout=0.2)
            a = int(msg.get("action", 0))
        except Exception:
            mask = info.get("legal_action_mask")
            a = bots[agent].act(mask)
        e.step(a)
        step += 1
    await ws.close()