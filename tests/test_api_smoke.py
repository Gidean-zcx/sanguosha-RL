from __future__ import annotations
import json
from fastapi.testclient import TestClient
from api.main import app


def test_headless_run():
    c = TestClient(app)
    r = c.post("/headless/run", json={"seed": 0, "num_players": 4, "record": False, "num_episodes": 2, "parallelism": 2, "max_steps": 10})
    assert r.status_code == 200
    data = r.json()
    assert "game_ids" in data


def test_create_and_join_and_meta():
    c = TestClient(app)
    r = c.post("/rooms", json={"seed": 0, "num_players": 4, "record": True})
    assert r.status_code == 200
    gid = r.json()["game_id"]
    # list rooms
    lr = c.get("/rooms")
    assert lr.status_code == 200 and gid in lr.json().get("rooms", [])
    # room meta
    rm = c.get(f"/rooms/{gid}")
    assert rm.status_code == 200 and rm.json().get("game_id") == gid
    r2 = c.post("/rooms/join", json={"game_id": gid, "seat": 0, "kind": "human"})
    assert r2.status_code == 200 and r2.json()["ok"]
    r3 = c.get("/meta/cards")
    assert r3.status_code == 200
    meta = r3.json()
    assert "card_id_to_name" in meta and isinstance(meta["card_id_to_name"], (list, dict))


def test_ws_game_watch_and_control():
    c = TestClient(app)
    gid = c.post("/rooms", json={"seed": 1, "num_players": 4, "record": False}).json()["game_id"]
    # watch only
    with c.websocket_connect(f"/ws/game/{gid}") as ws:
        msg = ws.receive_json()
        assert "agent" in msg and "info" in msg
        assert "seating" in msg["info"] and "targets" in msg["info"]
    # control seat 0
    with c.websocket_connect(f"/ws/game/{gid}?seat=0") as ws:
        msg = ws.receive_json()
        mask = msg["info"]["mask"]
        # send a legal or 0
        a = next((i for i,v in enumerate(mask) if v), 0)
        ws.send_json({"action": a})