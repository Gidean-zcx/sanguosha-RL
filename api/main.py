from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from coordinator import LocalRoom

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