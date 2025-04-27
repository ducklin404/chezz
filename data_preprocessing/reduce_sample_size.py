#!/usr/bin/env python3
"""
Down-sample a large JSON-Lines corpus using reservoir sampling.

Example
-------
    $ python sample_jsonl.py train_1m.jsonl.gz train_500k.jsonl.gz -k 500000 -s 42 -vv
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
from pathlib import Path
from typing import List


# ────────────────────────────  CORE LOGIC  ──────────────────────────────


def reservoir_sample(
    src_path: Path,
    *,
    k: int,
    seed: int | None = None,
) -> List[str]:
    """Return *k* uniformly-sampled JSONL lines from *src_path*.

    The file may be plain text or gzipped (extension *.gz*).

    Args:
        src_path: Path to the source JSONL / JSONL.GZ file.
        k: Reservoir size.
        seed: Optional RNG seed for reproducibility.

    Raises:
        ValueError: If *k* is not positive.
        json.JSONDecodeError: If any line is not valid JSON.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    rng = random.Random(seed)
    reservoir: List[str] = []
    total: int = 0

    opener = gzip.open if src_path.suffix == ".gz" else open
    with opener(src_path, "rt", encoding="utf-8") as fp:
        for line in fp:
            # Validate JSON early so failures surface before writing output
            json.loads(line)

            total += 1
            if len(reservoir) < k:
                reservoir.append(line)
            else:
                j = rng.randrange(total)
                if j < k:
                    reservoir[j] = line

            if total % 1_000_000 == 0:
                logging.debug("Scanned %,d lines …", total)

    logging.info("Reservoir sampling complete: %,d lines scanned", total)
    return reservoir


def write_jsonl(
    dst_path: Path,
    lines: List[str],
) -> None:
    """Write *lines* to *dst_path* in JSON-Lines format (gzipped if *.gz*)."""
    opener = gzip.open if dst_path.suffix == ".gz" else open
    with opener(dst_path, "wt", encoding="utf-8") as fp:
        fp.writelines(lines)
    logging.info("Wrote %,d lines → %s", len(lines), dst_path)


# ─────────────────────────────  CLI  ────────────────────────────────────


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Uniformly down-sample a JSON-Lines file via reservoir sampling.",
    )
    p.add_argument("src", type=Path, help="Source JSONL or JSONL.GZ file")
    p.add_argument("dst", type=Path, help="Destination JSONL/JSONL.GZ file")
    p.add_argument(
        "-k",
        "--samples",
        type=int,
        default=500_000,
        metavar="N",
        help="Number of lines to retain (default: 500 000)",
    )
    p.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducible sampling",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v or -vv)",
    )
    return p


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING  # 0 flags
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname).1s %(asctime)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = build_cli().parse_args()
    configure_logging(args.verbose)

    try:
        sample = reservoir_sample(args.src, k=args.samples, seed=args.seed)
        write_jsonl(args.dst, sample)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
