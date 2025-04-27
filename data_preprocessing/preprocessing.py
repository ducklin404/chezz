#!/usr/bin/env python3
"""stream_chess_sampling.py – Sample high‑quality chess positions and export
as gzipped JSONL ready for model training.

Key features
------------
* **Single‑pass reservoir sampling** – keeps memory usage bounded while
  producing an (almost) uniformly random subset of the input CSV.
* **Pluggable progress reporting** – shows a *tqdm* bar when the library is
  installed, otherwise falls back to plain‑text tickers via the standard
  :pydata:`logging` module.
* **Fully typed, documented and test‑friendly** – every public function is
  annotated, has a docstring and avoids side effects that would complicate
  unit testing.

The input CSV is assumed to contain (at minimum) the following columns::

    fen,line,depth

Where ``line`` holds the principal‑variation string beginning with a UCI move.
The output JSON Lines file contains one record per sampled position::

    {
      "text": "FEN: …\nBest Move JSON:{\"from\":\"e2\",…}"
    }

Run ``python stream_chess_sampling.py --help`` for every available CLI option.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import chess  # pip install python-chess

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover – we gracefully degrade
    tqdm = None  # type: ignore

__all__ = [
    "Config",
    "build_reservoir",
    "write_jsonl",
    "first_uci",
    "piece_name",
    "main",
]

PIECE_NAMES: dict[int, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

# ---------------------------------------------------------------------------
# Data‑class configuration ---------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Config:
    """Immutable runtime settings."""

    input_csv: Path
    output_jsonl_gz: Path

    sample_size: int = 1_000_000
    depth_min: int = 20
    rng_seed: int = 42
    total_lines_hint: int = 200_000_000
    log_every_n: int = 1_000_000

    def __post_init__(self) -> None:  # noqa: D401 – simple style
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.depth_min <= 0:
            raise ValueError("depth_min must be positive")

# ---------------------------------------------------------------------------
# Helper functions -----------------------------------------------------------
# ---------------------------------------------------------------------------

def first_uci(pv: str | None) -> str | None:
    """Return the first plausible 4‑ or 5‑character UCI move from *pv*.

    Parameters
    ----------
    pv
        A space‑delimited string of moves (principal variation) or *None*.

    Returns
    -------
    str | None
        The first UCI move if it has a valid length, otherwise *None*.
    """
    if not pv:
        return None
    move = pv.split()[0]
    return move if 4 <= len(move) <= 5 else None


def piece_name(board: chess.Board, uci: str) -> str:
    """Return lowercase English name of the piece that starts *uci* on *board*."""
    mv = chess.Move.from_uci(uci)
    piece = board.piece_at(mv.from_square)
    return PIECE_NAMES.get(piece.piece_type, "unknown") if piece else "unknown"

# ---------------------------------------------------------------------------
# Core logic -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_reservoir(cfg: Config) -> Tuple[List[Tuple[str, str]], int]:
    """Stream *cfg.input_csv* and perform reservoir sampling.

    The function keeps at most *cfg.sample_size* entries in memory and returns
    a list of ``(fen, uci)`` tuples.

    Returns
    -------
    reservoir
        Samples collected from the CSV.
    n_seen
        Number of rows that satisfied the *depth_min* filter (useful for stats).
    """
    random.seed(cfg.rng_seed)
    reservoir: List[Tuple[str, str]] = []
    n_seen = 0
    start_time = time.monotonic()

    logger = logging.getLogger(__name__)
    logger.debug("Opening CSV: %s", cfg.input_csv)

    with cfg.input_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        iterator: Iterable[dict[str, str]]  # type: ignore[assignment]
        if tqdm is not None:
            iterator = tqdm(
                reader,
                total=cfg.total_lines_hint,
                unit="lines",
                miniters=cfg.log_every_n,
                desc="Scanning CSV",
            )
        else:
            iterator = reader

        for row in iterator:  # type: ignore[arg-type]
            try:
                if int(row["depth"]) < cfg.depth_min:
                    continue
            except (KeyError, ValueError):
                continue  # malformed record – skip silently

            uci = first_uci(row.get("line"))
            if not uci:
                continue

            n_seen += 1
            if len(reservoir) < cfg.sample_size:
                reservoir.append((row["fen"], uci))
            else:
                j = random.randrange(n_seen)
                if j < cfg.sample_size:
                    reservoir[j] = (row["fen"], uci)

            # Poor‑man's progress when tqdm is unavailable
            if tqdm is None and n_seen % cfg.log_every_n == 0:
                pct = 100 * len(reservoir) / cfg.sample_size
                elapsed = time.monotonic() - start_time
                speed = n_seen / max(elapsed, 1)
                logger.info(
                    "[%d] rows (%.0f/s) – reservoir %.1f%%", n_seen, speed, pct
                )

    return reservoir, n_seen


def write_jsonl(reservoir: Iterable[Tuple[str, str]], cfg: Config) -> None:
    """Serialize *reservoir* to *cfg.output_jsonl_gz* in JSON Lines format."""
    logger = logging.getLogger(__name__)
    cfg.output_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)
    total = len(reservoir)  # type: ignore[arg-type]
    logger.debug("Writing %d examples to %s", total, cfg.output_jsonl_gz)

    with gzip.open(cfg.output_jsonl_gz, "wt", encoding="utf-8") as gz:
        iterator: Iterable[Tuple[int, Tuple[str, str]]]
        iterator = enumerate(reservoir, 1)
        if tqdm is not None:
            iterator = tqdm(iterator, total=total, unit="examples", desc="Writing")

        for idx, (fen, uci) in iterator:
            board = chess.Board(fen)
            record = {
                "text": (
                    f"FEN: {fen}\nBest Move JSON:" +
                    json.dumps(
                        {
                            "from": uci[:2],
                            "to": uci[2:4],
                            "piece": piece_name(board, uci),
                            "explanation": "",
                            "taunt": "",
                        },
                        separators=(",", ":"),
                    )
                )
            }
            gz.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Finished writing → %s", cfg.output_jsonl_gz)

# ---------------------------------------------------------------------------
# CLI glue -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def _parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Reservoir‑sample chess positions and export as JSONL‑GZ",
    )
    parser.add_argument("input_csv", type=Path, help="Path to input CSV file")
    parser.add_argument("output_jsonl_gz", type=Path, help="Path to output .jsonl.gz")
    parser.add_argument("--sample-size", type=int, default=1_000_000, help="Number of examples to sample (default: 1M)")
    parser.add_argument("--depth-min", type=int, default=20, help="Minimum engine depth to keep a row (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--total-lines-hint", type=int, default=200_000_000, help="Estimated total lines for progress bars (default: 200M)")
    parser.add_argument("--log-every-n", type=int, default=1_000_000, help="Print ticker every N lines when tqdm is absent (default: 1M)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    ns = parser.parse_args(argv)

    _configure_logging(ns.verbose)

    return Config(
        input_csv=ns.input_csv.expanduser(),
        output_jsonl_gz=ns.output_jsonl_gz.expanduser(),
        sample_size=ns.sample_size,
        depth_min=ns.depth_min,
        rng_seed=ns.seed,
        total_lines_hint=ns.total_lines_hint,
        log_every_n=ns.log_every_n,
    )


# ---------------------------------------------------------------------------
# Public entry‑point ---------------------------------------------------------
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: D401 – imperative style
    """Program entry‑point used by both CLI and :pyfunc:`python -m` invocations."""
    cfg = _parse_args(argv)
    logger = logging.getLogger(__name__)

    logger.info("Streaming & sampling …")
    reservoir, n_seen = build_reservoir(cfg)
    logger.info("Pass complete – scanned %,d lines, collected %,d examples", n_seen, len(reservoir))

    logger.info("Writing JSONL …")
    write_jsonl(reservoir, cfg)


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nAborted by user. Bye!\n")
