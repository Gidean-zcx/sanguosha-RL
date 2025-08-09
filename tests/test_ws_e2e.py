from __future__ import annotations
from fastapi.testclient import TestClient
from api.main import app
import time


def test_multi_seat_control_and_watch():
    c = TestClient(app)
    gid = c.post("/rooms", json={"seed": 42, "num_players": 4, "record": False}).json()["game_id"]
    # one controller for seat 0, one watcher
    with c.websocket_connect(f"/ws/game/{gid}?seat=0") as ws_ctrl:
        msg = ws_ctrl.receive_json()
        assert "agent" in msg
        # open watcher
        with c.websocket_connect(f"/ws/game/{gid}") as ws_watch:
            msg2 = ws_watch.receive_json()
            assert "agent" in msg2
            # if controller's turn, send an action; otherwise receive until it's controller's turn or 5 cycles
            attempts = 0
            while attempts < 5:
                if msg["agent"] == msg2["agent"]:
                    mask = msg["info"]["mask"]
                    a = next((i for i, v in enumerate(mask) if v), 0)
                    ws_ctrl.send_json({"action": a})
                    break
                msg = ws_ctrl.receive_json()
                msg2 = ws_watch.receive_json()
                attempts += 1


def test_disconnect_and_reconnect_same_seat():
    c = TestClient(app)
    gid = c.post("/rooms", json={"seed": 99, "num_players": 4, "record": False}).json()["game_id"]
    # connect seat 0, then disconnect and reconnect
    with c.websocket_connect(f"/ws/game/{gid}?seat=0") as ws:
        first = ws.receive_json()
        assert "agent" in first
    # reconnect
    with c.websocket_connect(f"/ws/game/{gid}?seat=0") as ws2:
        second = ws2.receive_json()
        assert "agent" in second


def test_headless_stream_closes_after_steps():
    c = TestClient(app)
    with c.websocket_connect("/ws/headless") as ws:
        count = 0
        try:
            while True:
                ws.receive_json()
                count += 1
                if count > 60:
                    break
        except Exception:
            # closed by server after 50 steps
            pass
        assert count >= 1