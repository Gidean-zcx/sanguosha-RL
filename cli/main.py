import typer
from typing import Optional
import numpy as np
from rich import print

from sgs_env import env as make_env
from agents import RandomLegalBot
from coordinator import HeadlessBatch
import uvicorn
import os
import json

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


@app.command("convert-rlds")
def convert_rlds(
    replays_root: str = typer.Argument("replays", help="RLDS 根目录（包含按日期划分的子目录）"),
    date: Optional[str] = typer.Option(None, help="指定日期目录(YYYYMMDD)，不指定则取最新"),
    out_dpo: str = typer.Option("dpo_pairs.out.jsonl", help="输出 DPO JSONL 文件路径"),
    out_grpo: str = typer.Option("grpo_groups.out.jsonl", help="输出 GRPO JSONL 文件路径"),
):
    """将 RLDS 目录中的 DPO/GRPO 数据汇总导出为 TRL 友好的 JSONL。"""
    if not os.path.isdir(replays_root):
        print(f"[red]目录不存在: {replays_root}[/red]")
        raise typer.Exit(1)
    picked_date = date
    if picked_date is None:
        # 选择最新日期目录
        sub = [d for d in os.listdir(replays_root) if os.path.isdir(os.path.join(replays_root, d))]
        if not sub:
            print("[red]未找到任何日期目录[/red]")
            raise typer.Exit(1)
        picked_date = sorted(sub)[-1]
    day_dir = os.path.join(replays_root, picked_date)
    dpo_path = os.path.join(day_dir, "dpo_pairs.jsonl")
    grpo_path = os.path.join(day_dir, "grpo_groups.jsonl")
    if not os.path.isfile(dpo_path) and not os.path.isfile(grpo_path):
        print(f"[yellow]指定日期无 DPO/GRPO 文件: {picked_date}[/yellow]")
        raise typer.Exit(1)
    # 简单透传/合并：逐行读取，写出到目标文件
    def concat_jsonl(src: str, dst: str):
        if not os.path.isfile(src):
            return 0
        n = 0
        with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                fout.write(line + "\n")
                n += 1
        return n
    n_dpo = concat_jsonl(dpo_path, out_dpo)
    n_grpo = concat_jsonl(grpo_path, out_grpo)
    print({"date": picked_date, "dpo": n_dpo, "grpo": n_grpo, "out_dpo": out_dpo, "out_grpo": out_grpo})


if __name__ == "__main__":
    app()