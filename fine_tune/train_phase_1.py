#!/usr/bin/env python3
"""
Phase 1  ·  Fine-tune on a JSONL file that holds FEN + best move only.
Each line must look like:

{"fen":"rnbqkbnr/pppp…","from":"e2","to":"e4","piece":"pawn"}

`explanation` & `taunt` are filled with "" so the JSON structure
matches exactly what the ipynb expected.
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
from peft import LoraConfig, get_peft_model

import config as cfg


# ── helpers ────────────────────────────────────────────────────────────────
def build_pair(row: dict, tok) -> dict:
    comp = {
        "from": row["from"].lower(),
        "to": row["to"].lower(),
        "piece": row["piece"].lower(),
        "explanation": "",
        "taunt": "",
    }
    comp_str = json.dumps(comp, separators=(",", ":"))
    prompt = (
        cfg.SYSTEM
        + "<|user|>FEN: " + row["fen"] + "\n\n"
        + "<|assistant|><|json|>"
    )
    completion = comp_str + tok.eos_token
    return {"prompt": prompt, "completion": completion}


def mask_prompt(tok_out, prompt_len: int):
    """Mask the prompt tokens so the loss is computed only on the completion."""
    labels = tok_out["input_ids"].clone()
    labels[:, :prompt_len] = -100
    tok_out["labels"] = labels
    return tok_out


class Collator:
    def __init__(self, tok, max_len=512):
        self.tok, self.max_len = tok, max_len

    def __call__(self, batch):
        prompts  = [b["prompt"] + b["completion"] for b in batch]
        enc = self.tok(
            prompts, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        # loss masking
        for i, b in enumerate(batch):
            mask_prompt(enc, len(self.tok(b["prompt"])["input_ids"]))
        return enc


# ── main ───────────────────────────────────────────────────────────────────
def run(args):
    data_path = Path(args.jsonl)
    assert data_path.exists(), data_path

    # tokenizer (+ special tokens)
    tok = AutoTokenizer.from_pretrained(cfg.BASE_ID)
    tok.add_special_tokens({"additional_special_tokens": ["<|json|>", "</|json|>"]})
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    # model + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_ID, quantization_config=cfg.BNB_CFG, device_map="auto"
    )
    model.resize_token_embeddings(len(tok))
    model = get_peft_model(
        model,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none", task_type="CAUSAL_LM",
        ),
    )

    # build hf Dataset in-memory
    pairs = []
    with data_path.open() as f:
        for line in f:
            pairs.append(build_pair(json.loads(line), tok))
    ds = Dataset.from_list(pairs)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(cfg.CKPT_MOVES),
            num_train_epochs=1,
            per_device_train_batch_size=16,
            learning_rate=5e-4,
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
    model.save_pretrained(cfg.CKPT_MOVES)
    tok.save_pretrained(cfg.CKPT_MOVES)
    print(f"✓ Phase 1 saved to {cfg.CKPT_MOVES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to best-move JSONL")
    run(ap.parse_args())
