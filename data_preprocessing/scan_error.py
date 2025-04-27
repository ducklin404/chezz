#!/usr/bin/env python3
"""
Utility for synchronizing the *fen* field between two JSON-Lines (*.jsonl*) files.

Typical use-case:
    A reference “skeleton” file (`b_path`) is regarded as ground-truth,
    while a derived file (`a_path`) needs its *fen* values kept in sync
    from a certain line onwards.

Running as a script::

    $ python sync_fen.py final_humor.jsonl humor_skeleton.jsonl -s 4300 -vv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List


# ────────────────────────────── I/O HELPERS ──────────────────────────────


def read_jsonl(path: Path) -> List[Dict]:
    """Return the contents of *path* as a list of dicts.

    Args:
        path: Path to a valid JSON-Lines file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If a line cannot be parsed.
    """
    with path.open(encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def write_jsonl(path: Path, data: List[Dict]) -> None:
    """Write *data* to *path* in JSON-Lines format."""
    with path.open("w", encoding="utf-8") as fp:
        for obj in data:
            fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ───────────────────────────── CORE FUNCTION ─────────────────────────────


def update_fen_field(
    a_path: Path,
    b_path: Path,
    *,
    start_index: int = 700,
) -> bool:
    """Synchronise the `fen` field of *a_path* with that of *b_path*.

    Args:
        a_path: File to update **in-place**.
        b_path: Reference file.
        start_index: First (0-based) index to compare.

    Returns:
        True if *a_path* was rewritten, False otherwise.
    """
    a_list = read_jsonl(a_path)
    b_list = read_jsonl(b_path)

    changed = False
    max_i = min(len(a_list), len(b_list))

    for i in range(start_index, max_i):
        a_fen = a_list[i].get("fen")
        b_fen = b_list[i].get("fen")

        if a_fen != b_fen:
            logging.debug("Line %d differs – %r → %r", i, a_fen, b_fen)
            a_list[i]["fen"] = b_fen
            changed = True

    if changed:
        write_jsonl(a_path, a_list)

    return changed


# ────────────────────────────── CLI & ENTRY ──────────────────────────────


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronise the 'fen' field between two JSON-Lines files."
    )
    parser.add_argument(
        "a_path",
        type=Path,
        help="JSONL file that will be updated in-place.",
    )
    parser.add_argument(
        "b_path",
        type=Path,
        help="Reference JSONL file whose 'fen' values are authoritative.",
    )
    parser.add_argument(
        "-s",
        "--start-index",
        type=int,
        default=700,
        metavar="N",
        help="0-based index at which to begin comparison (default: 700).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (-v or -vv).",
    )
    return parser


def configure_logging(verbosity: int) -> None:
    """Configure root logger based on *verbosity* count (0, 1, 2…)."""
    level = logging.WARNING  # default
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname).1s %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = build_cli().parse_args()
    configure_logging(args.verbose)

    try:
        if update_fen_field(args.a_path, args.b_path, start_index=args.start_index):
            logging.info("Updated %s", args.a_path)
        else:
            logging.info("No changes needed – %s already in sync.", args.a_path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
