#!/usr/bin/env python3
"""
Phase 2  ·  Fine-tune >from< the Phase-1 checkpoint on a JSONL file that
already contains *explanation* and *taunt* fields.

Each line must look like:

{
  "fen":"…",
  "from":"e2","to":"e4","piece":"pawn",
  "explanation":"targets f7 and gains space",
  "taunt":"your king feels drafty already"
}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

import config as cfg
from train_phase1_moves import build_pair, Collator   # reuse utilities


def run(args):
    data_path = Path(args.jsonl)
    assert data_path.exists(), data_path

    tok = AutoTokenizer.from_pretrained(cfg.CKPT_MOVES)
    tok.add_special_tokens({"additional_special_tokens": ["<|json|>", "</|json|>"]})
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.CKPT_MOVES, quantization_config=cfg.BNB_CFG, device_map="auto"
    )
    model.resize_token_embeddings(len(tok))

    pairs = []
    with data_path.open() as f:
        for line in f:
            pairs.append(build_pair(json.loads(line), tok))   # now expl & taunt filled!
    ds = Dataset.from_list(pairs)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(cfg.CKPT_FINAL),
            num_train_epochs=1,
            per_device_train_batch_size=8,
            learning_rate=3e-4,
            gradient_accumulation_steps=2,
            fp16=cfg.DTYPE == torch.float16,
            bf16=cfg.DTYPE == torch.bfloat16,
            logging_steps=50,
            save_steps=500,
            report_to="none",
        ),
        train_dataset=ds,
        data_collator=Collator(tok),
    )

    trainer.train()
    model.save_pretrained(cfg.CKPT_FINAL)
    tok.save_pretrained(cfg.CKPT_FINAL)
    print(f"✓ Phase 2 saved to {cfg.CKPT_FINAL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="JSONL with expl + taunt")
    run(ap.parse_args())
