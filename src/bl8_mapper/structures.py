"""Explicit entity selection and prediction-compatible structure projection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .reference import ReferenceSpace


def canonicalize_structure(
    source: Path,
    destination: Path,
    antigen_chain: str,
    heavy_chain: str,
    light_chain: str | None = None,
    model_index: int = 1,
) -> dict:
    """Select one protein antigen and one Fv/VHH, preserving residue provenance.

    Internal chain names A/B/C and sequential residue IDs make selections
    independent of source chain labels, insertion codes, and extra assemblies.
    Each atom name contributes its highest-occupancy alternate conformer.
    """
    import gemmi

    from ._prediction_features import AA3
    from .numbering import number_sequence

    roles = [("antigen", antigen_chain, "A"), ("heavy", heavy_chain, "B")]
    if light_chain:
        roles.append(("light", light_chain, "C"))
    if any(not name for _, name, _ in roles) or len({name for _, name, _ in roles}) != len(roles):
        raise ValueError("Antigen, heavy, and light selections must be distinct non-empty chains")
    structure = gemmi.read_structure(str(source))
    if not 1 <= model_index <= len(structure):
        raise ValueError(f"model_index must be between 1 and {len(structure)}")
    source_model = structure[model_index - 1]
    result = gemmi.Structure()
    result.name = "selected_interface"
    model = gemmi.Model("1")
    residue_map, sequences = [], {}
    for role, source_name, canonical_name in roles:
        matches = [c for c in source_model if c.name == source_name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one chain {source_name!r}; found {len(matches)}")
        new_chain = gemmi.Chain(canonical_name)
        sequence = []
        for residue in matches[0]:
            if residue.name.upper() not in AA3 or residue.find_atom("CA", "*") is None:
                continue
            atoms = {}
            for atom in residue:
                if atom.element.name.upper() in {"H", "D"} or atom.occ <= 0:
                    continue
                previous = atoms.get(atom.name)
                if previous is None or (atom.occ, atom.altloc in {"\x00", "", "A"}) > (
                    previous.occ,
                    previous.altloc in {"\x00", "", "A"},
                ):
                    atoms[atom.name] = atom
            if "CA" not in atoms:
                continue
            new_residue = gemmi.Residue()
            new_residue.name = residue.name
            new_residue.seqid = gemmi.SeqId(len(sequence) + 1, " ")
            new_residue.het_flag = "A"
            new_residue.entity_type = gemmi.EntityType.Polymer
            for atom in atoms.values():
                clone = atom.clone()
                clone.altloc = "\x00"
                new_residue.add_atom(clone)
            new_chain.add_residue(new_residue)
            sequence.append(AA3[residue.name.upper()])
            residue_map.append(
                {
                    "role": role,
                    "source_chain": source_name,
                    "source_residue": str(residue.seqid),
                    "source_resname": residue.name,
                    "canonical_chain": canonical_name,
                    "canonical_residue": len(sequence),
                }
            )
        if not sequence:
            raise ValueError(f"No supported protein CA residues on chain {source_name}")
        sequences[canonical_name] = "".join(sequence)
        if role != "antigen":
            annotation, chain_type = number_sequence(sequences[canonical_name])
            expected_types = {"H"} if role == "heavy" else {"L", "K"}
            if chain_type not in expected_types:
                raise ValueError(f"ANARCII disagrees with the selected {role} chain")
            if len(sequence) > 180 or len(sequence) - len(annotation) > 25:
                raise ValueError(
                    "Select isolated variable-domain chains; full Fab/IgG or fused scFv chains require preprocessing"
                )
        model.add_chain(new_chain)
    result.add_model(model)
    result.setup_entities()
    result.make_mmcif_document().write_file(str(destination))
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "antigen_chain": antigen_chain,
        "heavy_chain": heavy_chain,
        "light_chain": light_chain,
        "model_index": model_index,
        "residue_map": residue_map,
        "selected_protein_residues": len(residue_map),
        "selection_profile": "explicit_protein_chains_highest_occupancy_altloc",
    }


def extract_features(
    path: str | Path,
    antigen_chain: str,
    heavy_chain: str,
    light_chain: str | None = None,
    model_index: int = 1,
) -> tuple[dict, dict]:
    """Return 328-compatible inputs and explicit measurement/QC provenance.

    This is the corrected prediction projection profile, not a reconstruction
    of the original experimental patch and strict hydrogen-bond calculations.
    """
    from ._prediction_features import structure_features

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with TemporaryDirectory(prefix="bl8_mapper_") as temp:
        canonical = Path(temp) / "selected.cif"
        provenance = canonicalize_structure(
            source, canonical, antigen_chain, heavy_chain, light_chain, model_index
        )
        features, status = structure_features(canonical)
    provenance["feature_profile"] = "corrected_prediction_compatible"
    provenance["feature_status"] = status
    provenance["measurement_notes"] = [
        "Hydrogen-bond fields use an element/distance heuristic without D-H-A angles.",
        "Geometry uses one whole-interface patch; loop secondary structure uses the prediction placeholder.",
        "Water-mediated fields are compatibility zeros, not evidence that waters are biologically absent.",
        "B-factor fields retain the input values; prediction confidence is not experimental mobility.",
        "No model-native quality threshold is applied by this mapper.",
    ]
    return features, provenance


def map_structure(
    path: str | Path,
    antigen_chain: str,
    heavy_chain: str,
    light_chain: str | None = None,
    record_id: str = "query",
    model_index: int = 1,
    reference: ReferenceSpace | None = None,
    level: int = 8,
):
    """Map one explicitly selected interface; return assignments and provenance."""
    import pandas as pd

    ref = reference or ReferenceSpace()
    features, provenance = extract_features(
        path, antigen_chain, heavy_chain, light_chain, model_index
    )
    assignment = ref.project(pd.DataFrame([{"record_id": record_id, **features}]), level)
    assignment["mapping_profile"] = "corrected_prediction_compatible"
    return assignment, features, provenance


def close_structure_session() -> None:
    """Release the optional persistent PyMOL session after a batch."""
    from . import _prediction_features as core

    if core._PYMOL_SESSION is not None:
        core._PYMOL_SESSION.stop()
        core._PYMOL_SESSION = None


def write_provenance(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
