"""Prediction-compatible feature formulas used for corrected target projection.

Internal formal_* numeric keys are frozen feature identifiers. This module does
not calculate strict hydrogen bonds: its element/distance heuristic has no
explicit-hydrogen angle criterion. Whole-interface geometry and fixed loop
secondary-structure placeholders follow the existing prediction convention.
See docs/measurement_contract.md before interpreting covered or OOV.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np
import pymol2
from scipy.spatial import cKDTree

from .numbering import parse_chain_regions

_PYMOL_SESSION: pymol2.PyMOL | None = None

AA_CLASSES = {
    "basic": set("RKH"),
    "acidic": set("DE"),
    "polar": set("NQSTC"),
    "hydrophobic": set("AVILMGP"),
    "aromatic": set("FYW"),
}

AA_TO_CLASS = {aa: cls for cls, aas in AA_CLASSES.items() for aa in aas}

AA20 = set(AA_TO_CLASS)

AA3 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}

CDR_KEYS = ["hcdr1", "hcdr2", "hcdr3", "lcdr1", "lcdr2", "lcdr3"]

CDR_SET = set(CDR_KEYS)

CDR3_SET = {"hcdr3", "lcdr3"}

CDR_RANGES = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}

BACKBONE = {"N", "CA", "C", "O", "OXT"}

CONTACT_CUTOFF = 4.5

CLOSE_CUTOFF = 3.5

CLASH_CUTOFF = 2.0


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def one_letter(name: str) -> str:
    return AA3.get(name.upper(), "X")


def residue_key(residue: gemmi.Residue) -> str:
    ins = residue.seqid.icode.strip()
    return f"{residue.seqid.num}{ins}" if ins else str(residue.seqid.num)


def atom_xyz(atom: gemmi.Atom) -> np.ndarray:
    return np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)


def chain_residues(chain: gemmi.Chain) -> list[gemmi.Residue]:
    return [r for r in chain if r.find_atom("CA", "*") is not None]


def chain_sequence(residues: list[gemmi.Residue]) -> str:
    return "".join(one_letter(r.name) for r in residues)


def centroid(residue: gemmi.Residue) -> np.ndarray:
    coords = [atom_xyz(a) for a in residue if a.element.name.upper() != "H"]
    return np.mean(coords, axis=0) if coords else atom_xyz(residue.find_atom("CA", "*"))


def ca(residue: gemmi.Residue) -> np.ndarray:
    atom = residue.find_atom("CA", "*")
    return atom_xyz(atom) if atom else np.full(3, np.nan)


def residue_records(structure: gemmi.Structure) -> dict[str, list[gemmi.Residue]]:
    return {str(c.name): chain_residues(c) for c in structure[0] if chain_residues(c)}


def heavy_atoms(
    cid: str, residues: list[gemmi.Residue], region_map: dict[int, str], role: str
) -> list[dict[str, object]]:
    rows = []
    for index, residue in enumerate(residues):
        region = region_map.get(index, "antigen" if role == "antigen" else "framework")
        for atom in residue:
            element = atom.element.name.upper()
            if element == "H":
                continue
            rows.append(
                {
                    "chain": cid,
                    "resindex": index + 1,
                    "reskey": residue_key(residue),
                    "resname": residue.name.upper(),
                    "aa": one_letter(residue.name),
                    "atom": atom.name.upper(),
                    "element": element,
                    "region": region,
                    "backbone": atom.name.upper() in BACKBONE,
                    "coord": atom_xyz(atom),
                }
            )
    return rows


def is_acidic(atom: dict[str, object]) -> bool:
    return (str(atom["resname"]), str(atom["atom"])) in {
        ("ASP", "OD1"),
        ("ASP", "OD2"),
        ("GLU", "OE1"),
        ("GLU", "OE2"),
    }


def is_basic(atom: dict[str, object]) -> bool:
    return (str(atom["resname"]), str(atom["atom"])) in {
        ("LYS", "NZ"),
        ("ARG", "NE"),
        ("ARG", "NH1"),
        ("ARG", "NH2"),
        ("HIS", "ND1"),
        ("HIS", "NE2"),
    }


def prediction_hbond_proxy(a: dict[str, object], b: dict[str, object], d: float) -> bool:
    # Prediction-compatible element/distance heuristic; no D-H-A angle is measured.
    # The resulting field is explicitly marked as prediction-derived below.
    if d > 3.5 or a["backbone"] and b["backbone"]:
        return False
    donor_a = a["element"] in {"N", "O", "S"} and not is_acidic(a)
    donor_b = b["element"] in {"N", "O", "S"} and not is_acidic(b)
    acceptor_a = a["element"] in {"N", "O", "S"} and not is_basic(a)
    acceptor_b = b["element"] in {"N", "O", "S"} and not is_basic(b)
    return bool((donor_a and acceptor_b) or (donor_b and acceptor_a))


def chemistry(a: dict[str, object], b: dict[str, object], d: float) -> dict[str, int]:
    aa, bb = str(a["resname"]), str(b["resname"])
    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO", "CYS"}
    aromatic = {"PHE", "TYR", "TRP", "HIS"}
    hyd = int(
        d <= CONTACT_CUTOFF
        and aa in hydrophobic
        and bb in hydrophobic
        and not a["backbone"]
        and not b["backbone"]
        and a["element"] in {"C", "S"}
        and b["element"] in {"C", "S"}
    )
    arom = int(
        d <= 5.0 and aa in aromatic and bb in aromatic and not a["backbone"] and not b["backbone"]
    )
    catpi = int(d <= 6.0 and ((is_basic(a) and bb in aromatic) or (is_basic(b) and aa in aromatic)))
    salt = int(d <= 4.0 and ((is_basic(a) and is_acidic(b)) or (is_basic(b) and is_acidic(a))))
    hbond = int(prediction_hbond_proxy(a, b, d))
    return {
        "close": int(d <= CLOSE_CUTOFF),
        "hbond": hbond,
        "salt": salt,
        "hydrophobic": hyd,
        "aromatic": arom,
        "cation_pi": catpi,
        "backbone": int(a["backbone"] or b["backbone"]),
        "clash": int(d <= CLASH_CUTOFF),
    }


def compute_pymol_sasa(
    path: Path,
    binder: list[str],
    antigen: str,
    region_maps: dict[str, dict[int, str]],
    chains: dict[str, list[gemmi.Residue]],
) -> dict[str, float]:
    """Compute PyMOL dot-solvent SASA/BSA with the SAbDAb definition."""
    global _PYMOL_SESSION
    if _PYMOL_SESSION is None:
        _PYMOL_SESSION = pymol2.PyMOL()
        _PYMOL_SESSION.start()
    cmd = _PYMOL_SESSION.cmd
    cmd.delete("all")
    cmd.load(str(path), "src")
    cmd.remove("src and not polymer.protein")
    binder_expr = "+".join(binder)
    cmd.create("ab_free", f"src and chain {binder_expr} and polymer.protein")
    cmd.create("ag_free", f"src and chain {antigen} and polymer.protein")
    cmd.create("complex", f"src and chain {binder_expr}+{antigen} and polymer.protein")
    cmd.set("dot_solvent", 1)
    cmd.set("dot_density", 1)

    def atom_region(atom: object) -> str:
        chain = clean(getattr(atom, "chain", ""))
        resi = clean(getattr(atom, "resi", ""))
        # PyMOL uses the residue label as displayed in resi. Prediction
        # CIFs are normally sequential and therefore match raw indexes.
        try:
            index = int(re.match(r"-?\d+", resi).group(0)) - 1
        except (AttributeError, ValueError):
            return "framework"
        return region_maps.get(chain, {}).get(index, "framework")

    def area(selection: str, collect_regions: bool = False) -> tuple[float, dict[str, float]]:
        # Per-atom SASA values are only needed for the antibody selections.
        # Avoiding load_b/get_model for antigen-only and whole-complex areas
        # preserves the numerical definition while substantially reducing the
        # cost of large prediction cohorts.
        value = float(cmd.get_area(selection, load_b=1 if collect_regions else 0))
        if not collect_regions:
            return value, {}
        by_region: dict[str, float] = Counter()
        for atom in cmd.get_model(selection).atom:
            by_region[atom_region(atom)] += float(atom.b)
        return value, dict(by_region)

    ab_free, ab_free_regions = area("ab_free", collect_regions=True)
    ag_free, _ = area("ag_free")
    complex_bound, _ = area("complex")
    ab_bound, ab_bound_regions = area("complex and chain " + binder_expr, collect_regions=True)
    cmd.delete("all")
    out: dict[str, float] = {
        "sasa_antibody_free_A2": ab_free,
        "sasa_antibody_bound_A2": ab_bound,
        "sasa_antigen_free_A2": ag_free,
        "sasa_complex_bound_A2": complex_bound,
        "bsa_antibody_A2": ab_free - ab_bound,
        "bsa_A2": (ab_free + ag_free - complex_bound) / 2.0,
        "bsa_per_antibody_free_sasa": (ab_free - ab_bound) / ab_free if ab_free else np.nan,
    }
    for cdr in CDR_KEYS:
        free = float(ab_free_regions.get(cdr, 0.0))
        bound = float(ab_bound_regions.get(cdr, 0.0))
        out[f"{cdr}_sasa_free_A2"] = free
        out[f"{cdr}_sasa_bound_A2"] = bound
        out[f"{cdr}_bsa_A2"] = free - bound
    return out


def pca_shape(points: list[np.ndarray]) -> dict[str, float]:
    x = np.asarray([p for p in points if np.isfinite(p).all()], dtype=float)
    if len(x) == 0:
        return {"rg": np.nan, "span": np.nan, "lin": np.nan, "plan": np.nan, "thick": np.nan}
    centered = x - x.mean(axis=0)
    rg = float(np.sqrt((centered * centered).sum(axis=1).mean()))
    span = float(max((np.linalg.norm(a - b) for a, b in combinations(x, 2)), default=0.0))
    if len(x) < 3:
        return {"rg": rg, "span": span, "lin": np.nan, "plan": np.nan, "thick": np.nan}
    vals = np.maximum(np.linalg.eigvalsh(np.cov(centered.T)), 0.0)
    total = float(vals.sum())
    return {
        "rg": rg,
        "span": span,
        "lin": float(vals[-1] / total) if total else np.nan,
        "plan": float(vals[0] / total) if total else np.nan,
        "thick": float(np.sqrt(vals[0])),
    }


def entropy(values: list[float]) -> float:
    x = np.asarray([v for v in values if v > 0], dtype=float)
    if len(x) <= 1:
        return 0.0
    p = x / x.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(x)))


def normalized_entropy(values: list[float], denominator_bins: int) -> float:
    """Match the SAbDAb entropy convention, including zero bins."""
    x = np.asarray([float(v) for v in values if float(v) > 0], dtype=float)
    if len(x) == 0 or denominator_bins <= 1:
        return 0.0
    p = x / x.sum()
    return float(-(p * np.log(p)).sum() / math.log(denominator_bins))


def loop_features(points: list[np.ndarray], ss: str = "L") -> dict[str, float]:
    x = np.asarray([p for p in points if np.isfinite(p).all()], dtype=float)
    if len(x) < 2:
        return {
            "contour": np.nan,
            "end": np.nan,
            "ext": np.nan,
            "curv": np.nan,
            "turn": np.nan,
            "n": float(len(x)),
            "line": np.nan,
            "plan": np.nan,
            "rg": np.nan,
            "thick": np.nan,
            "ssH": np.nan,
            "ssL": np.nan,
            "ssS": np.nan,
            "tip": np.nan,
        }
    seg = np.linalg.norm(np.diff(x, axis=0), axis=1)
    contour = float(seg.sum())
    end = float(np.linalg.norm(x[-1] - x[0]))
    vec = np.diff(x, axis=0)
    turns = []
    for u, v in zip(vec[:-1], vec[1:]):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu and nv:
            turns.append(math.degrees(math.acos(np.clip(float(np.dot(u, v) / (nu * nv)), -1, 1))))
    axis = x[-1] - x[0]
    axis_n = np.linalg.norm(axis)
    tip = (
        float(max((np.linalg.norm(np.cross(p - x[0], axis)) / axis_n for p in x), default=np.nan))
        if axis_n
        else np.nan
    )
    shape = pca_shape(list(x))
    return {
        "contour": contour,
        "end": end,
        "ext": end / contour if contour else np.nan,
        "curv": 1.0 - end / contour if contour else np.nan,
        "turn": float(np.mean(turns)) if turns else np.nan,
        "n": float(len(x)),
        "line": shape["lin"],
        "plan": shape["plan"],
        "rg": shape["rg"],
        "thick": shape["thick"],
        "ssH": float(ss.count("H") / len(ss)),
        "ssL": float(ss.count("L") / len(ss)),
        "ssS": float(ss.count("S") / len(ss)),
        "tip": tip,
    }


def sequence_features(seqs: dict[str, str], regions: dict[str, dict[int, str]]) -> dict[str, float]:
    partitions = {name: Counter() for name in ("variable", "framework", "cdr", "cdr3")}
    region_counts = Counter()
    for chain, seq in seqs.items():
        reg = regions.get(chain, {})
        for i, aa in enumerate(seq):
            cls = AA_TO_CLASS.get(aa)
            if cls is None:
                continue
            region = reg.get(i, "framework")
            partitions["variable"][cls] += 1
            if region == "framework":
                partitions["framework"][cls] += 1
            if region in CDR_SET:
                partitions["cdr"][cls] += 1
                region_counts[region] += 1
            if region in CDR3_SET:
                partitions["cdr3"][cls] += 1
    out: dict[str, float] = {}
    for partition in ("variable", "framework", "cdr", "cdr3"):
        total = sum(partitions[partition].values())
        for cls in AA_CLASSES:
            out[f"sequence_{partition}_{cls}_fraction"] = (
                partitions[partition][cls] / total if total else np.nan
            )
    for cls in AA_CLASSES:
        out[f"sequence_cdr_{cls}_enrichment_vs_framework"] = (
            out[f"sequence_cdr_{cls}_fraction"] - out[f"sequence_framework_{cls}_fraction"]
        )
    variable_total = sum(partitions["variable"].values())
    cdr_total = sum(partitions["cdr"].values())
    cdr3_total = sum(partitions["cdr3"].values())
    out["sequence_cdr_fraction_variable"] = cdr_total / variable_total if variable_total else np.nan
    out["sequence_cdr3_fraction_variable"] = (
        cdr3_total / variable_total if variable_total else np.nan
    )
    out["sequence_cdr3_fraction_cdr"] = cdr3_total / cdr_total if cdr_total else np.nan
    for region in CDR_KEYS:
        out[f"sequence_{region}_fraction_cdr"] = (
            region_counts[region] / cdr_total if cdr_total else np.nan
        )
    out["sequence_variable_residue_count_qc"] = float(variable_total)
    out["sequence_has_mapped_variable_domain_qc"] = float(variable_total > 0)
    return out


def aa_pair_features(pairs: list[dict[str, object]]) -> dict[str, float]:
    out = {}
    # The reference AA-pair table is based on close_pairs (<=3.5 A), not all
    # loose 4.5 A interface pairs.
    close_pairs = [p for p in pairs if p["close"]]
    total = float(len(close_pairs))
    cdr = [p for p in close_pairs if str(p["ab_region"]) in CDR_SET]
    for ab_class in AA_CLASSES:
        for ag_class in AA_CLASSES:
            all_n = sum(
                AA_TO_CLASS.get(str(p["ab_aa"])) == ab_class
                and AA_TO_CLASS.get(str(p["ag_aa"])) == ag_class
                for p in close_pairs
            )
            cdr_n = sum(
                AA_TO_CLASS.get(str(p["ab_aa"])) == ab_class
                and AA_TO_CLASS.get(str(p["ag_aa"])) == ag_class
                for p in cdr
            )
            out[f"aa_pair_all_{ab_class}_to_{ag_class}_fraction"] = all_n / total if total else 0.0
            out[f"aa_pair_cdr_{ab_class}_to_{ag_class}_fraction"] = cdr_n / len(cdr) if cdr else 0.0

    def mass(scope: str, pairs_for_mass: list[tuple[str, str]]) -> float:
        data = close_pairs if scope == "all" else cdr
        denom = len(data)
        return (
            sum(
                1
                for p in data
                if (AA_TO_CLASS.get(str(p["ab_aa"])), AA_TO_CLASS.get(str(p["ag_aa"])))
                in pairs_for_mass
            )
            / denom
            if denom
            else 0.0
        )

    classes = list(AA_CLASSES)
    matrix = [(a, b) for a in classes for b in classes]
    for scope in ("all", "cdr"):
        out[f"aa_pair_{scope}_matrix_entropy"] = entropy(
            [
                sum(
                    1
                    for p in (close_pairs if scope == "all" else cdr)
                    if (AA_TO_CLASS.get(str(p["ab_aa"])), AA_TO_CLASS.get(str(p["ag_aa"]))) == cell
                )
                for cell in matrix
            ]
        )
        out[f"aa_pair_{scope}_opposite_charge_fraction"] = mass(
            scope, [("basic", "acidic"), ("acidic", "basic")]
        )
        out[f"aa_pair_{scope}_like_charge_fraction"] = mass(
            scope, [("basic", "basic"), ("acidic", "acidic")]
        )
        out[f"aa_pair_{scope}_polar_polar_fraction"] = mass(scope, [("polar", "polar")])
        out[f"aa_pair_{scope}_nonpolar_nonpolar_fraction"] = mass(
            scope,
            [
                (a, b)
                for a, b in matrix
                if a in {"hydrophobic", "aromatic"} and b in {"hydrophobic", "aromatic"}
            ],
        )
        out[f"aa_pair_{scope}_cation_aromatic_fraction"] = mass(
            scope, [("basic", "aromatic"), ("aromatic", "basic")]
        )
    cdr3 = [p for p in close_pairs if str(p["ab_region"]) in CDR3_SET]
    out["aa_pair_cdr_close_fraction_total"] = len(cdr) / len(close_pairs) if close_pairs else 0.0
    out["aa_pair_cdr3_close_fraction_total"] = len(cdr3) / len(close_pairs) if close_pairs else 0.0
    out["aa_pair_total_close_pairs_qc"] = float(len(close_pairs))
    out["aa_pair_has_close_contact_qc"] = float(bool(close_pairs))
    return out


def structure_features(path: Path) -> tuple[dict[str, float], dict[str, object]]:
    structure = gemmi.read_structure(str(path))
    chains = residue_records(structure)
    ag = "A"
    vh, vl = "B", ("C" if "C" in chains else None)
    if vh is None:
        raise ValueError("No antibody variable chain found")
    binder = [c for c in [vh, vl] if c is not None and c in chains]
    seqs = {c: chain_sequence(chains[c]) for c in binder}
    region_maps, spans, chain_types = parse_chain_regions(seqs)
    ag_atoms = heavy_atoms(ag, chains[ag], {}, "antigen")
    ab_atoms = [
        a for c in binder for a in heavy_atoms(c, chains[c], region_maps.get(c, {}), "antibody")
    ]
    tree = cKDTree(np.asarray([a["coord"] for a in ag_atoms]))
    pairs = []
    for a in ab_atoms:
        for j in tree.query_ball_point(a["coord"], CONTACT_CUTOFF):
            g = ag_atoms[j]
            d = float(np.linalg.norm(a["coord"] - g["coord"]))
            flags = chemistry(a, g, d)
            pairs.append(
                {
                    "ab_aa": a["aa"],
                    "ag_aa": g["aa"],
                    "ab_region": a["region"],
                    "ab_chain": a["chain"],
                    "ab_resindex": a["resindex"],
                    "ag_resindex": g["resindex"],
                    "d": d,
                    **flags,
                }
            )
    if not pairs:
        raise ValueError("No antibody-antigen contacts within 4.5 A")

    residue_pair_groups: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for pair in pairs:
        key = (str(pair["ab_chain"]), int(pair["ab_resindex"]), int(pair["ag_resindex"]))
        residue_pair_groups.setdefault(key, []).append(pair)
    residue_pair_rows = [
        {
            "ab_chain": key[0],
            "ab_resindex": key[1],
            "ag_resindex": key[2],
            "atom_pair_count": len(group),
            "min_distance": min(float(p["d"]) for p in group),
        }
        for key, group in residue_pair_groups.items()
    ]

    def sumflag(name: str) -> float:
        return float(sum(p[name] for p in pairs))

    close = sumflag("close")
    atom = float(len(pairs))
    unique_ab = {(p["ab_chain"], p["ab_resindex"]) for p in pairs}
    unique_ag = {p["ag_resindex"] for p in pairs}
    atom_by_region = {
        r: sum(1 for p in pairs if p["ab_region"] == r) for r in CDR_KEYS + ["framework"]
    }
    shares = {r: atom_by_region[r] / atom if atom else 0.0 for r in atom_by_region}
    ag_unique = sorted(unique_ag)
    ab_unique = sorted(unique_ab, key=lambda x: (x[0], x[1]))
    ag_shape = pca_shape([ca(chains[ag][i - 1]) for i in ag_unique if 0 < i <= len(chains[ag])])
    ab_shape = pca_shape(
        [ca(chains[c][i - 1]) for c, i in ab_unique if c in chains and 0 < i <= len(chains[c])]
    )
    # One global interface patch and regional local patches approximate the
    # same shape descriptors used in SAbDAb; all definitions are explicit.
    ag_span = max(ag_unique) - min(ag_unique) + 1 if ag_unique else 0
    ab_index = [i for _, i in ab_unique]
    ab_span = max(ab_index) - min(ab_index) + 1 if ab_index else 0
    ag_segments = 1 + sum(b != a + 1 for a, b in zip(ag_unique, ag_unique[1:])) if ag_unique else 0
    ab_segments = (
        1 + sum(b != a + 1 for a, b in zip(sorted(set(ab_index)), sorted(set(ab_index))[1:]))
        if ab_index
        else 0
    )
    ag_long = max(
        (
            len(list(g))
            for _, g in __import__("itertools").groupby(enumerate(ag_unique), lambda z: z[1] - z[0])
        ),
        default=0,
    )
    ab_sorted = sorted(set(ab_index))
    ab_long = max(
        (
            len(list(g))
            for _, g in __import__("itertools").groupby(enumerate(ab_sorted), lambda z: z[1] - z[0])
        ),
        default=0,
    )
    # B-factors in prediction CIFs are generally pLDDT-like confidence values.
    ca_b = {}
    for c in [*binder]:
        ca_b[c] = [float(r.find_atom("CA", "*").b_iso) for r in chains[c] if r.find_atom("CA", "*")]
    fw_b = [
        v
        for c in binder
        for i, v in enumerate(ca_b[c])
        if region_maps.get(c, {}).get(i) == "framework"
    ]
    fw_mean = float(np.mean(fw_b)) if fw_b else np.nan
    fw_sd = float(np.std(fw_b)) if fw_b else np.nan
    weighted_mean_min = float(
        np.average(
            [r["min_distance"] for r in residue_pair_rows],
            weights=[r["atom_pair_count"] for r in residue_pair_rows],
        )
    )
    features: dict[str, float] = {
        "contact_atom_pairs_raw": atom,
        "contact_close_pairs_raw": close,
        "contact_residue_pairs_raw": float(len(residue_pair_rows)),
        "contact_min_distance_raw": float(min(r["min_distance"] for r in residue_pair_rows)),
        "contact_mean_min_distance_raw": weighted_mean_min,
        "raw_atom_pair_count": atom,
        "raw_close_pairs": close,
        "raw_salt_bridge_pairs": sumflag("salt"),
        "raw_hydrophobic_pairs": sumflag("hydrophobic"),
        "raw_aromatic_pairs": sumflag("aromatic"),
        "raw_cation_pi_pairs": sumflag("cation_pi"),
        "raw_backbone_pairs": sumflag("backbone"),
        "raw_clash_pairs": sumflag("clash"),
        "chem_salt_bridge_pairs_density": sumflag("salt") / atom if atom else 0.0,
        "chem_hydrophobic_pairs_density": sumflag("hydrophobic") / atom if atom else 0.0,
        "chem_aromatic_pairs_density": sumflag("aromatic") / atom if atom else 0.0,
        "chem_cation_pi_pairs_density": sumflag("cation_pi") / atom if atom else 0.0,
        "chem_backbone_pairs_density": sumflag("backbone") / atom if atom else 0.0,
        "chem_clash_pairs_density": sumflag("clash") / atom if atom else 0.0,
        "formal_hbond_strict_count": sumflag("hbond"),
        "formal_hbond_relaxed_count": sumflag("hbond"),
        "formal_direct_hbond_strict_count": sumflag("hbond"),
        "formal_water_hbond_strict_count": 0.0,
        "formal_hbond_density": sumflag("hbond") / atom if atom else 0.0,
        "formal_direct_hbond_density": sumflag("hbond") / atom if atom else 0.0,
        "formal_water_hbond_density": 0.0,
        "formal_cdr3_hbond_strict_count": sum(
            p["hbond"] for p in pairs if p["ab_region"] in CDR3_SET
        ),
        "formal_hcdr3_hbond_strict_count": sum(
            p["hbond"] for p in pairs if p["ab_region"] == "hcdr3"
        ),
        "formal_lcdr3_hbond_strict_count": sum(
            p["hbond"] for p in pairs if p["ab_region"] == "lcdr3"
        ),
        "geometry_epitope_n": float(len(unique_ag)),
        "geometry_epitope_span": float(ag_span),
        "geometry_epitope_segments": float(ag_segments),
        "geometry_epitope_longest_run": float(ag_long),
        "geometry_epitope_contiguity": ag_long / len(unique_ag) if unique_ag else np.nan,
        "geometry_paratope_n": float(len(unique_ab)),
        "geometry_paratope_span": float(ab_span),
        "geometry_paratope_segments": float(ab_segments),
        "geometry_paratope_longest_run": float(ab_long),
        "geometry_paratope_contiguity": ab_long / len(unique_ab) if unique_ab else np.nan,
        "geometry_antigen_patch_radius_gyration_mean": ag_shape["rg"],
        "geometry_antigen_patch_radius_gyration_median": ag_shape["rg"],
        "geometry_antigen_patch_span_mean": ag_shape["span"],
        "geometry_antigen_patch_span_median": ag_shape["span"],
        "geometry_antigen_patch_linearity_mean": ag_shape["lin"],
        "geometry_antigen_patch_linearity_median": ag_shape["lin"],
        "geometry_antigen_patch_planarity_mean": ag_shape["plan"],
        "geometry_antigen_patch_planarity_median": ag_shape["plan"],
        "geometry_antigen_patch_thickness_mean": ag_shape["thick"],
        "geometry_antigen_patch_thickness_median": ag_shape["thick"],
        "geometry_antibody_patch_radius_gyration_mean": ab_shape["rg"],
        "geometry_antibody_patch_radius_gyration_median": ab_shape["rg"],
        "geometry_antibody_patch_span_mean": ab_shape["span"],
        "geometry_antibody_patch_span_median": ab_shape["span"],
        "geometry_shape_complementarity_proxy_mean": (close / atom) * (1 - sumflag("clash") / atom)
        if atom
        else np.nan,
        "geometry_shape_complementarity_proxy_median": (close / atom)
        * (1 - sumflag("clash") / atom)
        if atom
        else np.nan,
        "geometry_surface_closure_proxy_mean": len(pairs) / (len(unique_ab) * len(unique_ag))
        if unique_ab and unique_ag
        else np.nan,
        "geometry_surface_closure_proxy_median": len(pairs) / (len(unique_ab) * len(unique_ag))
        if unique_ab and unique_ag
        else np.nan,
        "geometry_approach_angle_abs_cos_mean": np.nan,
        "geometry_approach_angle_abs_cos_median": np.nan,
        "geometry_directional_lock_fraction_mean": (
            sumflag("hbond") + sumflag("salt") + sumflag("cation_pi")
        )
        / atom
        if atom
        else np.nan,
        "geometry_directional_lock_fraction_median": (
            sumflag("hbond") + sumflag("salt") + sumflag("cation_pi")
        )
        / atom
        if atom
        else np.nan,
        "geometry_packing_fraction_mean": (sumflag("hydrophobic") + sumflag("aromatic")) / atom
        if atom
        else np.nan,
        "geometry_packing_fraction_median": (sumflag("hydrophobic") + sumflag("aromatic")) / atom
        if atom
        else np.nan,
        "geometry_local_epitope_contiguity_mean": ag_long / len(unique_ag) if unique_ag else np.nan,
        "geometry_local_epitope_contiguity_median": ag_long / len(unique_ag)
        if unique_ag
        else np.nan,
        "geometry_local_epitope_coverage_mean": len(unique_ag) / ag_span if ag_span else np.nan,
        "geometry_local_epitope_coverage_median": len(unique_ag) / ag_span if ag_span else np.nan,
        "geometry_local_epitope_n_segments_mean": float(ag_segments),
        "geometry_local_epitope_n_segments_median": float(ag_segments),
        "geometry_local_epitope_longest_run_mean": float(ag_long),
        "geometry_local_epitope_longest_run_median": float(ag_long),
        "geometry_local_paratope_contiguity_mean": ab_long / len(unique_ab)
        if unique_ab
        else np.nan,
        "geometry_local_paratope_contiguity_median": ab_long / len(unique_ab)
        if unique_ab
        else np.nan,
        "geometry_n_patches": 1.0,
        "bsa_A2": np.nan,
        "bsa_antibody_A2": np.nan,
        "bsa_per_antibody_free_sasa": np.nan,
        "framework_mean_bfactor": fw_mean,
        "framework_sd_bfactor": fw_sd,
        "n_antigen_ca": float(len(chains[ag])),
        "n_antigen_chains": 1.0,
        "n_binder_ca": float(sum(len(chains[c]) for c in binder)),
        "n_binder_chains": float(len(binder)),
        "sasa_antibody_bound_A2": np.nan,
        "sasa_antibody_free_A2": np.nan,
        "sasa_antigen_free_A2": np.nan,
        "sasa_complex_bound_A2": np.nan,
    }
    for cdr in CDR_KEYS:
        chain = vh if cdr.startswith("h") else vl
        points = (
            [
                ca(chains[chain][i])
                for i, r in region_maps.get(chain, {}).items()
                if r == cdr and i < len(chains.get(chain, []))
            ]
            if chain
            else []
        )
        lf = loop_features(points, "L")
        pre = cdr
        features.update(
            {
                f"{pre}_contour_A": lf["contour"],
                f"{pre}_curvature_proxy": lf["curv"],
                f"{pre}_end_to_end_A": lf["end"],
                f"{pre}_extension_ratio": lf["ext"],
                f"{pre}_mean_turn_angle_deg": lf["turn"],
                f"{pre}_n_ca": lf["n"],
                f"{pre}_pca_linearity": lf["line"],
                f"{pre}_pca_planarity": lf["plan"],
                f"{pre}_rg_A": lf["rg"],
                f"{pre}_ss_H_fraction": lf["ssH"],
                f"{pre}_ss_L_fraction": lf["ssL"],
                f"{pre}_ss_S_fraction": lf["ssS"],
                f"{pre}_thickness_A": lf["thick"],
                f"{pre}_tip_protrusion_A": lf["tip"],
            }
        )
        features[f"{pre}_bsa_A2"] = np.nan
        features[f"{pre}_sasa_bound_A2"] = np.nan
        features[f"{pre}_sasa_free_A2"] = np.nan
        bf = (
            [
                ca_b.get(chain, [np.nan] * len(chains.get(chain, [])))[i]
                for i, r in region_maps.get(chain, {}).items()
                if r == cdr and i < len(ca_b.get(chain, []))
            ]
            if chain
            else []
        )
        features[f"{pre}_mean_bfactor"] = float(np.mean(bf)) if bf else np.nan
        features[f"{pre}_bfactor_z_vs_framework"] = (
            (features[f"{pre}_mean_bfactor"] - fw_mean) / fw_sd
            if bf and fw_sd and np.isfinite(fw_sd)
            else np.nan
        )
        # Topology names in the reference table are residue-based.
        features[f"topology_{cdr}_atom_pairs"] = float(
            sum(1 for p in pairs if p["ab_region"] == cdr)
        )
        features[f"topology_{cdr}_close_pairs"] = float(
            sum(p["close"] for p in pairs if p["ab_region"] == cdr)
        )
        features[f"topology_{cdr}_share"] = shares[cdr]
        features[f"topology_{cdr}_residue_count"] = float(
            len({p["ab_resindex"] for p in pairs if p["ab_region"] == cdr})
        )
    for r in ["framework"]:
        features[f"topology_{r}_atom_pairs"] = float(sum(1 for p in pairs if p["ab_region"] == r))
        features[f"topology_{r}_close_pairs"] = float(
            sum(p["close"] for p in pairs if p["ab_region"] == r)
        )
        features[f"topology_{r}_share"] = shares[r]
        features[f"topology_{r}_residue_count"] = float(
            len({p["ab_resindex"] for p in pairs if p["ab_region"] == r})
        )
    region_vals = [shares[r] for r in CDR_KEYS + ["framework"]]
    features["topology_entropy"] = normalized_entropy(region_vals, 7)
    features["topology_n_active_regions"] = float(sum(v >= 0.02 for v in region_vals))
    for i in [1, 2, 3]:
        features[f"topology_cdr{i}_share"] = sum(
            shares.get(f"hcdr{i}", 0.0) + shares.get(f"lcdr{i}", 0.0) for _ in [0]
        )
    features.update(aa_pair_features(pairs))
    features.update(sequence_features(seqs, region_maps))
    sasa = compute_pymol_sasa(path, binder, ag, region_maps, chains)
    features.update(sasa)
    # Prediction CIFs do not normally contain crystallographic waters; this is
    # recorded separately from formal direct hydrogen-bond geometry.
    status = {
        "antigen_chain": ag,
        "vh_chain": vh,
        "vl_chain": vl,
        "binder_chains": ";".join(binder),
        "hbond_measurement": "prediction_element_distance_proxy_no_DHA_angle",
        "has_contacts": bool(pairs),
    }
    return features, status
