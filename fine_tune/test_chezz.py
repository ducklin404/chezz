#!/usr/bin/env python3
"""
Single-FEN check:
* loads base model in fp16
* attaches LoRA adapter from CKPT_FINAL
* generates one JSON answer
"""

from __future__ import annotations
import argparse, textwrap, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import config as cfg


def load_model():
    tok = AutoTokenizer.from_pretrained(cfg.BASE_ID)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": ["<|json|>", "</|json|>"]})

    base = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_ID, torch_dtype=torch.float16
    )
    base.resize_token_embeddings(len(tok))

    model = PeftModel.from_pretrained(base, cfg.CKPT_FINAL).half().to(cfg.DEVICE).eval()
    return tok, model


def run(fen: str):
    tok, model = load_model()
    prompt = (
        cfg.SYSTEM
        + f"<|user|>FEN: {fen}\n\n"
        + "<|assistant|><|json|>"
    )
    with torch.inference_mode():
        out = model.generate(
            **tok(prompt, return_tensors="pt").to(cfg.DEVICE),
            max_new_tokens=64,
            pad_token_id=tok.pad_token_id,
            do_sample=True,
        )
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True, help="FEN string to test")
    run(ap.parse_args().fen)
