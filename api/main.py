from __future__ import annotations
from fastapi import FastAPI, WebSocket, Query
from pydantic import BaseModel
from coordinator import LocalRoom, HeadlessBatch, RoomCoordinator
import asyncio
from agents import RandomLegalBot
from agents.llm_adapter import LLMAdapter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
from sgs_env.cards import CARD_ID_TO_NAME
from sgs_env.actions import (
    INDEX_SHA_BASE,
    INDEX_JUEDOU_BASE,
    INDEX_GUOHE_BASE,
    INDEX_SHUNSHOU_BASE,
    INDEX_LE_BASE,
    INDEX_BINGLIANG_BASE,
    NUM_SEAT_SLOTS,
    action_index_for_sha_to_seat,
    action_index_for_targeted,
)

app = FastAPI(title="SGS PettingZoo API")

# mount static web ui
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


rc = RoomCoordinator()


class CreateRoomReq(BaseModel):
    seed: int = 0
    num_players: int = 4
    record: bool = True


@app.get("/")
def root_page():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"ok": True}


@app.get("/meta/cards")
def cards_meta():
    return JSONResponse({"card_id_to_name": CARD_ID_TO_NAME})


@app.get("/rooms")
def list_rooms():
    return {"rooms": list(rc.rooms.keys())}


@app.get("/rooms/{game_id}")
def room_meta(game_id: str):
    room = rc.get_room(game_id)
    return {"game_id": game_id, "seed": room.seed, "num_players": room.num_players, "record": room.record, "seats": room.seats}


@app.get("/replays")
def list_replay_dates():
    root = "replays"
    if not os.path.isdir(root):
        return {"dates": []}
    dates = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    dates.sort()
    return {"dates": dates}


@app.get("/replays/{date}/files")
def list_replay_files(date: str):
    root = os.path.join("replays", date)
    if not os.path.isdir(root):
        return {"files": []}
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    files.sort()
    return {"files": files}


@app.get("/replays/{date}/download")
def download_replay(date: str, file: str = Query(..., description="file name under date dir")):
    path = os.path.join("replays", date, file)
    if not os.path.isfile(path):
        return JSONResponse({"error": "not_found"}, status_code=404)
    return FileResponse(path)


@app.post("/rooms")
def create_room(req: CreateRoomReq):
    room_id = rc.create_room(seed=req.seed, num_players=req.num_players, record=req.record)
    return {"game_id": room_id}


class JoinReq(BaseModel):
    game_id: str
    seat: int
    kind: str = "human"  # human/llm/bot
    provider: str | None = None
    model: str | None = None


@app.post("/rooms/join")
def join_room(req: JoinReq):
    ok = rc.join(req.game_id, req.seat, req.kind, req.provider, req.model)
    return {"ok": bool(ok)}


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
            "observation": info.get("observation_struct", {}),
            "rewards": e.rewards,
            "terminations": e.terminations,
            "truncations": e.truncations,
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
async def ws_game(ws: WebSocket, game_id: str, seat: int = Query(-1, description="control seat; -1 watch")):
    await ws.accept()
    room = rc.get_room(game_id)
    from sgs_env import env as make_env
    e = make_env(seed=room.seed, num_players=room.num_players)
    e.reset(seed=room.seed)
    # Build controllers per seat: seat param controls that seat; others follow room.seats config (default bot)
    agents = list(e.agents)
    seat_of_agent = {agent: e.unwrapped.state.players[agent].seat for agent in agents}
    controllers: dict[str, tuple[str, object]] = {}
    for agent in agents:
        s = seat_of_agent[agent]
        if seat >= 0 and s == seat:
            controllers[agent] = ("human", None)
            continue
        cfg = room.seats.get(s, {"kind": "bot"})
        kind = (cfg.get("kind") or "bot").lower()
        if kind == "llm":
            prov = cfg.get("provider") or "auto"
            model = cfg.get("model") or None
            controllers[agent] = ("llm", LLMAdapter(provider=prov, model=model, seed=room.seed + s))
        elif kind == "human":
            # if no controlling connection for this seat, will fallback to bot upon timeout
            controllers[agent] = ("human", None)
        else:
            controllers[agent] = ("bot", RandomLegalBot(seed=room.seed + s))

    # recent history for LLM context
    history: list[dict] = []

    step = 0
    while step < 300 and not (all(e.terminations.values()) or all(e.truncations.values())):
        agent = e.agent_selection
        _o, _r, _t, _tr, info = e.last()
        obs_struct = info.get("observation_struct")
        # build target hints
        targets = []
        st = e.unwrapped.state
        for s in range(NUM_SEAT_SLOTS):
            tgt = st.agent_by_seat(s)
            if tgt and st.players[tgt].alive and tgt != agent:
                item = {
                    "seat": s,
                    "sha": action_index_for_sha_to_seat(s),
                    "juedou": action_index_for_targeted(INDEX_JUEDOU_BASE, s),
                    "guohe": action_index_for_targeted(INDEX_GUOHE_BASE, s),
                    "shunshou": action_index_for_targeted(INDEX_SHUNSHOU_BASE, s),
                    "le": action_index_for_targeted(INDEX_LE_BASE, s),
                    "bingliang": action_index_for_targeted(INDEX_BINGLIANG_BASE, s),
                }
                targets.append(item)
        seating = [{"agent": a, "seat": seat_of_agent[a], "alive": e.unwrapped.state.players[a].alive} for a in agents]
        msg = {
            "agent": agent,
            "info": {
                "mask": info.get("legal_action_mask").tolist(),
                "events": info.get("events", []),
                "phase": obs_struct.get("phase") if isinstance(obs_struct, dict) else None,
                "observation": obs_struct,
                "rewards": e.rewards,
                "terminations": e.terminations,
                "truncations": e.truncations,
                "targets": targets,
                "seating": seating,
            },
        }
        await ws.send_json(msg)
        mask = info.get("legal_action_mask")
        ctrl_kind, ctrl_obj = controllers.get(agent, ("bot", RandomLegalBot(seed=0)))
        a: int | None = None
        preferred = None
        # priority: human for matching seat, else LLM, else bot
        if ctrl_kind == "human":
            try:
                cli = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
                a = int(cli.get("action", 0))
                preferred = cli.get("card")  # optional selected concrete card (cid,suit)
            except Exception:
                a = None
        if a is None and ctrl_kind == "llm":
            try:
                a = int(ctrl_obj.act(obs_struct, mask, history))
            except Exception:
                a = None
        if a is None:
            a = int(RandomLegalBot(seed=room.seed + step).act(mask))
        # set preferred card for this agent if provided
        st.preferred_card[agent] = tuple(preferred) if (isinstance(preferred, list) and len(preferred) == 2) else None
        # record into history (limited fields)
        history.append({
            "agent": agent,
            "phase": msg["info"]["phase"],
            "events": msg["info"]["events"],
            "action": int(a),
        })
        if len(history) > 32:
            history = history[-32:]
        e.step(a)
        step += 1
    await ws.close()