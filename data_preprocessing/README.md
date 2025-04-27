# Chess-Sampling Utilities

A small toolbox for turning gargantuan chess‐engine evaluation dumps into
training-ready JSONL datasets — complete with optional humour prompts and
FEN-synchronisation helpers.

| Script | Purpose |
| ------ | ------- |
| `preprocessing.py`<br/>(*aka* **stream_chess_sampling.py**) | One-pass **reservoir-samples** high-quality positions straight from a CSV and writes a gzipped JSONL corpus ready for LLM ingestion. |
| `reduce_sample_size.py` | Uniformly down-samples an **existing** JSONL/JSONL.GZ file to a smaller size. |
| `gen_humor.py`<br/>(*aka* **make_humor_set.py**) | Builds a “skeleton” humour-alignment set: FEN + best move JSON with empty `explanation`/`taunt` fields for a human author/AI to fill. |
| `scan_error.py`<br/>(*aka* **sync_fen.py**) | Keeps two JSONL files in sync by copying the authoritative `fen` field from one to the other after a chosen index. |

---
