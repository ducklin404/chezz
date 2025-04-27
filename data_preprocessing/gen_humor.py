#!/usr/bin/env python3
"""
make_humor_set.py
=================

Generate a “skeleton” humour‑alignment dataset by uniformly sampling positions
from a large engine output CSV.

For every qualifying row (`depth` ≥ *DEPTH_MIN*) the script extracts:
    • the Forsyth‑Edwards Notation (`fen`)
    • the first principal‑variation move in UCI (`line` column)

The sample is written to a JSON Lines file whose objects look like

    {
        "fen": "<FEN>",
        "best_move_json": {
            "from": "e2",
            "to":   "e4",
            "piece": "pawn",
            "explanation": "",
            "taunt": ""
        }
    }

The human author can then fill the *explanation* and *taunt* fields manually.

Example
-------

    $ python make_humor_set.py --input evals.csv --output humor.jsonl \
        --sample-size 5000 --depth-min 20

Requirements
------------
    • python-chess  (pip install python-chess)
    • tqdm          (optional, for a progress bar)

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import chess  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore


# --------------------------------------------------------------------------- #
#                               Constants & Types                             #
# --------------------------------------------------------------------------- #

PIECE_NAMES: dict[int, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

CsvRow = dict[str, str]
FenUciPair = Tuple[str, str]


# --------------------------------------------------------------------------- #
#                               Utility Functions                             #
# --------------------------------------------------------------------------- #

def first_uci(pv: str) -> str | None:
    """
    Return the first token of a principal‑variation string *pv* iff it looks like
    a valid UCI move (4–5 characters). Otherwise return *None*.
    """
    if not pv:
        return None
    move = pv.split(maxsplit=1)[0]
    return move if 4 <= len(move) <= 5 else None


def piece_name(board: chess.Board, uci: str) -> str:
    """
    Map *uci*’s *from* square to a human‑readable piece name according to *board*.

    Falls back to ``"unknown"`` if the square is empty or unrecognised.
    """
    move = chess.Move.from_uci(uci)
    piece = board.piece_at(move.from_square)
    return PIECE_NAMES.get(piece.piece_type, "unknown") if piece else "unknown"


# --------------------------------------------------------------------------- #
#                           Core Reservoir Sampling                           #
# --------------------------------------------------------------------------- #

def reservoir_sample(
    rows: Iterable[CsvRow],
    sample_size: int,
    depth_min: int,
    loop_limit: int,
    seed: int,
    print_every_n: int = 1_000_000,
) -> List[FenUciPair]:
    """
    Uniformly reservoir‑sample *(fen, uci)* pairs from *rows* such that
    ``int(row["depth"]) >= depth_min``.

    Parameters
    ----------
    rows :
        An iterable of CSV dictionaries.
    sample_size :
        Desired size of the output reservoir.
    depth_min :
        Minimum search depth that a row must satisfy to be considered.
    loop_limit :
        Hard cap on how many rows to scan before aborting (safety valve).
    seed :
        RNG seed for reproducibility.
    print_every_n :
        Print a heartbeat when *tqdm* is unavailable.

    Returns
    -------
    list[tuple[str, str]]
        A uniformly sampled list of *(fen, uci)* pairs.
    """
    random.seed(seed)
    reservoir: List[FenUciPair] = []
    n_seen = 0
    tic = time.time()

    iterator: Iterable[CsvRow]
    if tqdm is not None:
        iterator = tqdm(rows, unit="rows", desc="Scanning CSV")
    else:
        iterator = rows

    for row in iterator:
        if n_seen >= loop_limit:
            logging.warning(
                "Loop limit of %d reached – sampling terminated early.", loop_limit
            )
            break

        try:
            if int(row["depth"]) < depth_min:
                continue
        except (KeyError, ValueError):
            # Missing or malformed depth → skip
            continue

        uci = first_uci(row.get("line", ""))
        if not uci:  # malformed PV
            continue

        n_seen += 1

        if len(reservoir) < sample_size:
            reservoir.append((row["fen"], uci))
        else:
            j = random.randrange(n_seen)
            if j < sample_size:
                reservoir[j] = (row["fen"], uci)

        # Fallback ticker
        if tqdm is None and n_seen % print_every_n == 0:
            pct = 100 * len(reservoir) / sample_size
            speed = n_seen / max(time.time() - tic, 1.0)
            logging.info(
                "[%s rows]  %.0f rows/s  |  reservoir %.1f%%",
                f"{n_seen:_}",
                speed,
                pct,
            )

    return reservoir


# --------------------------------------------------------------------------- #
#                               Output Writers                                #
# --------------------------------------------------------------------------- #

def write_jsonl(reservoir: List[FenUciPair], output_path: Path) -> None:
    """
    Serialize *reservoir* to *output_path* in JSON Lines format.

    Each line is a JSON object as shown in the module docstring with empty
    ``explanation`` and ``taunt`` fields.
    """
    iterator: Iterable[Tuple[int, FenUciPair]] = enumerate(reservoir, 1)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(reservoir), unit="ex", desc="Writing JSONL")

    with output_path.open("w", encoding="utf-8") as fp:
        for _, (fen, uci) in iterator:
            board = chess.Board(fen)
            obj = {
                "fen": fen,
                "best_move_json": {
                    "from": uci[:2],
                    "to": uci[2:4],
                    "piece": piece_name(board, uci),
                    "explanation": "",
                    "taunt": "",
                },
            }
            fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
#                           Command‑line Interface                            #
# --------------------------------------------------------------------------- #

def parse_arguments() -> argparse.Namespace:
    """Parse and return command‑line arguments."""
    parser = argparse.ArgumentParser(description="Sample a humour‑alignment dataset.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("evals.csv"),
        metavar="CSV",
        help="Path to source CSV file (default: evals.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("humor_skeleton.jsonl"),
        metavar="JSONL",
        help="Destination JSONL path (default: humor_skeleton.jsonl)",
    )
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=5000,
        help="Number of examples to sample (default: 5000)",
    )
    parser.add_argument(
        "--depth-min",
        type=int,
        default=20,
        metavar="DEPTH",
        help="Minimum search depth required (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="PRNG seed (default: 123)",
    )
    parser.add_argument(
        "--loop-limit",
        type=int,
        default=5_000_000,
        help="Maximum rows to scan before aborting (default: 5,000,000)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity (default: INFO)",
    )

    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Configure the root logger once per process."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
#                                  Entrypoint                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    """Main program entrypoint wrapped in KeyboardInterrupt handling."""
    args = parse_arguments()
    configure_logging(args.log_level)

    logging.info("Reading CSV from %s", args.input)
    try:
        with args.input.open(newline="") as fp:
            reader = csv.DictReader(fp)
            reservoir = reservoir_sample(
                rows=reader,
                sample_size=args.sample_size,
                depth_min=args.depth_min,
                loop_limit=args.loop_limit,
                seed=args.seed,
            )
    except FileNotFoundError:
        logging.error("Input file %s not found.", args.input)
        sys.exit(1)
    except csv.Error as exc:
        logging.error("CSV parse error: %s", exc)
        sys.exit(1)

    logging.info("Sampled %d positions.", len(reservoir))
    logging.info("Writing JSONL to %s", args.output)
    try:
        write_jsonl(reservoir, args.output)
    except OSError as exc:
        logging.error("Failed to write JSONL: %s", exc)
        sys.exit(1)

    logging.info("Finished. You can now fill in 'explanation' and 'taunt' fields.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)
