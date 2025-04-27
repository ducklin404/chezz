# Chezz – TinyLlama meets trash-talking chess 🏰🐴💬

> A two-stage LoRA fine-tune that teaches a 1.1 B-parameter TinyLlama to  
> 1. **play strong chess** (predict the engine’s best move) and  
> 2. **explain that move & fire off a playful taunt**, all in one JSON blob.


A **full dev log write-up** lives here: <https://asilentpond.com/projects/chezz>


---



## Folder layout

```text
chezz/
├── chezz.ipynb               # the full notebook that was used
├── config.py                 # global settings, SYSTEM prompt, JSON schema
├── train_phase1.py           # Phase 1 fine-tune on FEN → best-move pairs
├── train_phase2.py           # Phase 2 fine-tune (adds explanation + taunt)
├── test_chezz.py             # single-FEN sanity check (base + LoRA)
└── requirements.txt          # transformers, datasets, peft, etc.
```

---

## Dataset

Both training splits live in a single HuggingFace dataset:

<https://huggingface.co/datasets/ducklin404/chezz_dataset/tree/main>

| File in Kaggle     | Used in phase | Rows | Columns |
|--------------------|---------------|------|------------------------------------------------|
| `train_500k.jsonl.gz`   | Phase 1       | ~500 k | `fen`,`from`,`to`,`piece` |
| `final_humor.jsonl`   | Phase 2       | ~5 k | `fen`,`from`,`to`,`piece`,`explanation`,`taunt` |

Download the archive, unzip anywhere, and pass the JSONL paths to the training scripts.

---

## Quick-start (GPU recommended)

```bash
git clone https://github.com/ducklin404/chezz.git
cd chezz/fine_tune
python -m venv .venv && source .venv/bin/activate     # optional but tidy
pip install -r requirements.txt                       # HF Transformers, PEFT…

# -------- Phase 1 – best move only --------
python train_phase1_moves.py \
    --jsonl /path/to/moves.jsonl

# -------- Phase 2 – explanation + taunt ----
python train_phase2_expl_taunt.py \
    --jsonl /path/to/humor.jsonl

# -------- Test a random position ----------
python test_chezz.py \
    --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
```

> **Tip:** The scripts automatically load the TinyLlama base weights in `fp16`, then attach the LoRA adapter saved from the previous phase.

---

## How it works (concise)

1. **Prompt design** – We wrap every training example in the same role-tagged prompt you see in the notebook:  
   `<|system|> … <|user|>FEN: … <|assistant|><|json|>`.
2. **Loss masking** – Only the JSON completion tokens count toward the loss, never the prompt.
3. **LoRA** – 4-bit quantised TinyLlama + Rank-16 adapters keeps VRAM under ≈10 GB.
4. **Phase 1** learns move selection; **Phase 2** keeps those weights frozen and nudges the adapter so the model *also* justifies the move and roasts the opponent.
5. **Inference** – Base model (fp16) + adapter loaded with `PeftModel`; generation is batched for speed.

---

## License

I don't know what to put here :>