import typer
from typing import Optional
import numpy as np
from rich import print

from sgs_env import env as make_env
from agents import RandomLegalBot
from coordinator import HeadlessBatch
import uvicorn

app = typer.Typer(help="Sanguosha (PettingZoo) CLI")


def _choose_from_mask(mask) -> int:
    legal = [i for i, v in enumerate(mask) if v]
    if not legal:
        return 0
    while True:
        try:
            raw = typer.prompt(f"Choose action from legal {legal}", default=str(legal[0]))
            a = int(raw)
            if a in legal:
                return a
        except Exception:
            pass
        print("[red]Invalid action. Try again.[/red]")


@app.command()
def play(seed: int = 0, num_players: int = 4, max_steps: int = 200):
    """本地交互游玩：你控制座位0，其余为随机Bot。"""
    e = make_env(seed=seed, num_players=num_players)
    e.reset(seed=seed)
    bots = {agent: RandomLegalBot(seed=seed + i) for i, agent in enumerate(e.agents)}
    human_agent = e.agents[0]
    step = 0
    while step < max_steps and not (all(e.terminations.values()) or all(e.truncations.values())):
        agent = e.agent_selection
        obs, rew, term, trunc, info = e.last()
        mask = info.get("legal_action_mask")
        print(f"\n[bold]Agent[/bold]: {agent}  [bold]Phase[/bold]: {info.get('observation_struct', {}).get('phase')}")
        if agent == human_agent:
            a = _choose_from_mask(mask)
        else:
            a = bots[agent].act(mask)
        e.step(a)
        step += 1
    print("\n[green]Game finished.[/green]")


@app.command()
def selfplay(seed: int = 0, num_players: int = 4, episodes: int = 8, parallelism: int = 4, max_steps: int = 200, record: bool = True):
    """批量自博弈，生成RLDS日志。"""
    hb = HeadlessBatch(seed=seed, num_players=num_players, record=record)
    ids = hb.run(num_episodes=episodes, parallelism=parallelism, max_steps=max_steps)
    print({"game_ids": ids})


@app.command()
def api(host: str = "0.0.0.0", port: int = 8000):
    """启动 FastAPI 服务。"""
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()