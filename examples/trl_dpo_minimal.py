#!/usr/bin/env python3
# Minimal DPO training example (optional deps). Reads dpo_pairs.jsonl and prepares a dataset.

from __future__ import annotations
import os, json
from typing import List, Dict


def load_dpo_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main():
    date = sorted(os.listdir("replays"))[-1]
    path = os.path.join("replays", date, "dpo_pairs.jsonl")
    if not os.path.isfile(path):
        print("No dpo_pairs.jsonl found. Run headless first.")
        return
    data = load_dpo_jsonl(path)
    print({"loaded": len(data), "path": path})
    try:
        from datasets import Dataset  # type: ignore
        from trl import DPOConfig, DPOTrainer  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception:
        print("datasets/trl/transformers not installed. Skipping actual training.")
        return
    # Convert to a toy dataset: map action indices to strings
    def example_mapper(ex):
        chosen = str(ex["chosen"]["action"]) if isinstance(ex.get("chosen"), dict) else str(ex.get("chosen"))
        rejected = str(ex["rejected"]["action"]) if isinstance(ex.get("rejected"), dict) else str(ex.get("rejected"))
        prompt = f"legal_actions=[...] observation=..."  # you can serialize full observation if recorded
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    mapped = [example_mapper(x) for x in data]
    ds = Dataset.from_list(mapped)
    model_name = os.environ.get("TRL_MODEL", "distilbert-base-uncased")
    print({"dataset_rows": len(ds), "model": model_name})
    # NOTE: Replace with a causal LM checkpoint suited for DPO
    # config = DPOConfig(output_dir="./trl_out", per_device_train_batch_size=2)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tok = AutoTokenizer.from_pretrained(model_name)
    # trainer = DPOTrainer(model=model, args=config, beta=0.1, train_dataset=ds, tokenizer=tok)
    # trainer.train()
    print("Example complete (no actual training run).")


if __name__ == "__main__":
    main()