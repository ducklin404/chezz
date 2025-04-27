"""
Global settings shared by every Chezz script.
The SYSTEM prompt & JSON schema are copied verbatim from chezz.ipynb.
"""

from __future__ import annotations
from pathlib import Path
import torch
from transformers import BitsAndBytesConfig

# ── hardware ────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
BNB_CFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=DTYPE,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ── model / checkpoints ─────────────────────────────────────────────────────
BASE_ID     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"      # same as ipynb
OUT_DIR     = Path("outputs")
CKPT_MOVES  = OUT_DIR / "phase1_moves"
CKPT_FINAL  = OUT_DIR / "phase2_expl_taunt"

# ── JSON schema & system preamble  (straight copy) ──────────────────────────
SCHEMA = """{
  "from": "<square>",        # e.g. "e2"
  "to":   "<square>",        # e.g. "e4"
  "piece": "<piece>",        # "pawn","knight",…
  "explanation": "<text>",   # short rationale
  "taunt": "<text>"          # optional cheeky comment
}"""

SYSTEM = (
    "<|system|> You are **ChezzBot-β**, a dry-humoured, mildly anxious chess coach "
    "who is utterly convinced every move you pick is textbook-perfect. You always play "
    "as the side to move; your opponent is the other colour whom you tease with sarcastic digs. "
    "Explain your move in ≤25 words using real chess ideas, then taunt the user in ≤15 words. "
    "Respond *only* with JSON: " + SCHEMA + " "
)

# ── generation defaults for quick tests ─────────────────────────────────────
GEN_KWARGS = dict(max_new_tokens=64, temperature=0.7, top_p=0.9, do_sample=True)
