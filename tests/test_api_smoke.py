from __future__ import annotations
import json
from fastapi.testclient import TestClient
from api.main import app


def test_create_room_and_headless_smoke():
    c = TestClient(app)
    r = c.post("/rooms", json={"seed": 0, "num_players": 4, "record": False})
    assert r.status_code == 200
    game_id = r.json()["game_id"]
    assert isinstance(game_id, str)

    r = c.post("/headless/run", json={"seed": 0, "num_players": 4, "record": False, "num_episodes": 2, "parallelism": 2, "max_steps": 10})
    assert r.status_code == 200
    data = r.json()
    assert "game_ids" in data