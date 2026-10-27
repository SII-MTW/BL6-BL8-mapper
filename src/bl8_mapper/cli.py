"""Command-line tools for frozen BL6 and BL8 queries."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import pandas as pd

from .reference import load_reference_space


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--reference-dir", type=Path, help="Alternative complete reference asset directory"
    )
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="Verify frozen assets, counts, and checksums")
    for name in ("features", "states"):
        p = sub.add_parser(name, help=f"Map an input {name} CSV")
        p.add_argument("--input", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--level", choices=(6, 8), type=int, default=8)
        p.add_argument("--id-column", default="record_id")
        p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("structures", help="Extract prediction-compatible features from PDB/mmCIF")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--structure", type=Path)
    group.add_argument("--manifest", type=Path)
    p.add_argument("--antigen-chain")
    p.add_argument("--heavy-chain")
    p.add_argument("--light-chain")
    p.add_argument("--model-index", type=int, default=1)
    p.add_argument("--record-id", default="query")
    p.add_argument("--level", choices=(6, 8), type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return root


def _protect(paths: list[Path], overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}; use --overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)


def _versions() -> dict:
    result = {"python": platform.python_version()}
    for name in (
        "bl6-bl8-mapper",
        "numpy",
        "pandas",
        "gemmi",
        "scipy",
        "anarcii",
        "pymol-open-source",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed as a distribution"
    return result


def _run_structures(args, reference) -> int:
    from .structures import close_structure_session, map_structure, write_provenance

    if args.manifest:
        rows = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
        required = {"record_id", "path", "antigen_chain", "heavy_chain"}
        if not required.issubset(rows):
            raise ValueError(f"Manifest requires {sorted(required)}")
        reference._ids(rows, "record_id")
        records = rows.to_dict("records")
        base = args.manifest.resolve().parent
    else:
        if not args.antigen_chain or not args.heavy_chain:
            raise ValueError("--structure requires --antigen-chain and --heavy-chain")
        records = [
            {
                "record_id": args.record_id,
                "path": str(args.structure.resolve()),
                "antigen_chain": args.antigen_chain,
                "heavy_chain": args.heavy_chain,
                "light_chain": args.light_chain or "",
                "model_index": args.model_index,
            }
        ]
        base = Path.cwd()
    features_path = args.output.with_suffix(".features.csv")
    errors_path = args.output.with_suffix(".errors.csv")
    manifest_path = args.output.with_suffix(".run.json")
    _protect([args.output, features_path, errors_path, manifest_path], args.overwrite)
    assignments, features, errors, provenance = [], [], [], []
    try:
        for row in records:
            identity = row["record_id"]
            path = Path(row["path"])
            if not path.is_absolute():
                path = base / path
            try:
                mapped, values, notes = map_structure(
                    path,
                    row["antigen_chain"],
                    row["heavy_chain"],
                    row.get("light_chain") or None,
                    identity,
                    int(row.get("model_index") or 1),
                    reference,
                    args.level,
                )
                assignments.append(mapped)
                features.append({"record_id": identity, **values})
                provenance.append({"record_id": identity, **notes})
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                errors.append(
                    {"record_id": identity, "error_type": type(exc).__name__, "message": str(exc)}
                )
    finally:
        close_structure_session()
    if assignments:
        pd.concat(assignments, ignore_index=True).to_csv(args.output, index=False)
        pd.DataFrame(features).to_csv(features_path, index=False)
    pd.DataFrame(errors, columns=["record_id", "error_type", "message"]).to_csv(
        errors_path, index=False
    )
    write_provenance(
        manifest_path,
        {
            "versions": _versions(),
            "reference": reference.describe(),
            "submitted": len(records),
            "mapped": len(assignments),
            "failed": len(errors),
            "interfaces": provenance,
        },
    )
    print(
        json.dumps({"submitted": len(records), "mapped": len(assignments), "failed": len(errors)})
    )
    return 2 if errors else 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        reference = load_reference_space(args.reference_dir)
        if args.command == "validate":
            print(json.dumps(reference.describe(), indent=2))
        elif args.command in {"features", "states"}:
            _protect([args.output], args.overwrite)
            rows = pd.read_csv(args.input, dtype={args.id_column: str}, low_memory=False)
            method = reference.project if args.command == "features" else reference.lookup_states
            result = method(rows, level=args.level, id_column=args.id_column)
            result.to_csv(args.output, index=False)
            print(json.dumps(result[f"bl{args.level}_status"].value_counts().to_dict()))
        else:
            return _run_structures(args, reference)
    except (ValueError, OSError, ImportError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0
