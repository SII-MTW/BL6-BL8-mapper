"""Strict ANARCII IMGT annotation, with no framework-only fallback."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache

CDR_RANGES = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}
_NUMBERER = None


def _number_in_process(sequence: str) -> tuple[tuple[tuple[int, str], ...], str]:
    global _NUMBERER
    from anarcii import Anarcii

    if not sequence:
        raise ValueError("Empty antibody sequence")
    if _NUMBERER is None:
        _NUMBERER = Anarcii(
            seq_type="antibody", mode="accuracy", batch_size=1, cpu=True, ncpu=1, verbose=False
        )
    _NUMBERER.number({"query": sequence})
    item = _NUMBERER.to_scheme("imgt").get("query") or {}
    chain_type = str(item.get("chain_type", ""))
    if chain_type not in {"H", "K", "L"}:
        raise ValueError("ANARCII did not identify an antibody heavy/light variable domain")
    raw = int(item.get("query_start") or 0)
    regions = {}
    for position, amino_acid in item.get("numbering") or []:
        if amino_acid == "-":
            continue
        if raw >= len(sequence) or sequence[raw] != amino_acid:
            raise ValueError("ANARCII sequence/position round-trip mismatch")
        region = "framework"
        for name, (lo, hi) in CDR_RANGES.items():
            if lo <= int(position[0]) <= hi:
                region = ("h" if chain_type == "H" else "l") + name
                break
        regions[raw] = region
        raw += 1
    required = {("h" if chain_type == "H" else "l") + name for name in CDR_RANGES}
    if not required.issubset(set(regions.values())):
        raise ValueError("Incomplete IMGT CDR1/CDR2/CDR3 annotation")
    return tuple(regions.items()), chain_type


@lru_cache(maxsize=2048)
def number_sequence(sequence: str) -> tuple[tuple[tuple[int, str], ...], str]:
    """Keep PyTorch and PyMOL OpenMP runtimes separate on Windows."""
    if os.name != "nt":
        return _number_in_process(sequence)
    completed = subprocess.run(
        [sys.executable, "-m", "bl8_mapper.numbering"],
        input=json.dumps(sequence),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode:
        raise ValueError(f"ANARCII worker failed: {completed.stderr.strip()[-1200:]}")
    positions, chain_type = json.loads(completed.stdout)
    return tuple((int(index), region) for index, region in positions), chain_type


def parse_chain_regions(sequences: dict[str, str]) -> tuple[dict, dict, dict]:
    regions, spans, types = {}, {}, {}
    for chain, sequence in sequences.items():
        positions, chain_type = number_sequence(sequence)
        if chain == "B" and chain_type != "H" or chain == "C" and chain_type not in {"K", "L"}:
            raise ValueError(f"Antibody chain-role mismatch on canonical chain {chain}")
        regions[chain] = dict(positions)
        types[chain] = chain_type
        for region in set(regions[chain].values()) - {"framework"}:
            indexes = [i + 1 for i, name in positions if name == region]
            spans[f"{region}_span"] = f"{min(indexes)}-{max(indexes)}"
    return regions, spans, types


if __name__ == "__main__":
    from contextlib import redirect_stdout

    sequence = json.loads(sys.stdin.read())
    with redirect_stdout(sys.stderr):
        numbered = _number_in_process(sequence)
    print(json.dumps(numbered))
