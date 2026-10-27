"""Frozen transforms, nearest state assignment, and exact BL6/BL8 lookup."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VIEWS = (
    "interface_topology",
    "interface_shape_geometry",
    "interaction",
    "cdr_conformation",
    "flexibility_disorder",
    "solvent_interface_area",
    "aa_pair_contact_composition",
    "antibody_sequence_composition",
)
BASE_VIEWS = VIEWS[:6]
DEFAULT_REFERENCE = Path(__file__).resolve().parent / "reference_space"


def normalize_state(value: object) -> str:
    """Accept integer cluster codes or their C-prefixed representation."""
    text = str(value).strip()
    raw = text[1:] if text.startswith("C") else text
    if not re.fullmatch(r"-?\d+(?:\.0+)?", raw):
        raise ValueError(f"Invalid state code: {text!r}")
    return f"C{int(float(raw))}"


@dataclass(frozen=True)
class ViewTransform:
    features: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = frame.loc[:, self.features].apply(pd.to_numeric, errors="raise").to_numpy(float)
        values[~np.isfinite(values)] = np.nan
        missing = np.isnan(values)
        clipped = np.clip(values, self.lower, self.upper)
        filled = np.where(missing, self.median, clipped)
        return (filled - self.mean) / self.scale, missing.mean(axis=1)


class ReferenceSpace:
    """Read-only reference assets with no fitting or training operations."""

    def __init__(self, directory: Path = DEFAULT_REFERENCE, verify: bool = True):
        self.directory = Path(directory)
        if verify:
            self.verify_assets()
        self.metadata = json.loads((self.directory / "release_metadata.json").read_text())
        self.feature_dictionary = pd.read_csv(self.directory / "feature_dictionary.csv")
        self.state_dictionary = pd.read_csv(self.directory / "state_dictionary.csv")
        self.catalogs = {n: pd.read_csv(self.directory / f"bl{n}_catalog.csv") for n in (6, 8)}
        self.lookup = {
            n: catalog.set_index("state_tuple")[f"bl{n}_id"].to_dict()
            for n, catalog in self.catalogs.items()
        }
        stats = json.loads((self.directory / "preprocessing_stats.json").read_text())["views"]
        self.transforms: dict[str, ViewTransform] = {}
        self.centers: dict[str, np.ndarray] = {}
        self.labels: dict[str, list[str]] = {}
        self.names: dict[str, dict[str, str]] = {}
        for view in VIEWS:
            item = stats[view]
            features = tuple(item["features"])
            arrays = [
                np.asarray(item[key], float)
                for key in ("lower", "upper", "median", "mean", "scale")
            ]
            if any(a.shape != (len(features),) or not np.isfinite(a).all() for a in arrays):
                raise ValueError(f"Invalid preprocessing arrays for {view}")
            if (arrays[-1] <= 0).any() or (arrays[0] > arrays[1]).any():
                raise ValueError(f"Invalid scale or clipping bounds for {view}")
            self.transforms[view] = ViewTransform(features, *arrays)
            centers = pd.read_csv(self.directory / f"{view}_state_centers_z.csv")
            self.centers[view] = centers.loc[:, features].to_numpy(float)
            self.labels[view] = centers.cluster.map(normalize_state).tolist()
            if not np.isfinite(self.centers[view]).all():
                raise ValueError(f"Non-finite state centers for {view}")
            rows = self.state_dictionary[self.state_dictionary.view.eq(view)]
            self.names[view] = rows.set_index("state_code").state_name.to_dict()
        self._validate_catalogs()

    def verify_assets(self) -> None:
        """Check all bundled reference hashes before loading numerical assets."""
        checksums = json.loads((self.directory / "checksums.json").read_text())
        for name, expected in checksums.items():
            path = self.directory / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Reference checksum mismatch: {name}")

    def _validate_catalogs(self) -> None:
        for n, count in ((6, 185), (8, 1498)):
            catalog = self.catalogs[n]
            if len(catalog) != count or len(self.lookup[n]) != count:
                raise ValueError(f"Expected {count} unique BL{n} catalog entries")
            tuples = catalog[[f"{v}_state_code" for v in VIEWS[:n]]].agg("|".join, axis=1)
            if not tuples.equals(catalog.state_tuple):
                raise ValueError(f"Catalog tuple/column mismatch for BL{n}")
            if catalog[f"bl{n}_id"].duplicated().any():
                raise ValueError(f"Duplicate BL{n} catalog IDs")
            if int(catalog.n_records.sum()) != 15396:
                raise ValueError(f"Inconsistent record support for BL{n}")
        prefixes = {"|".join(t.split("|")[:6]) for t in self.lookup[8]}
        if prefixes != set(self.lookup[6]):
            raise ValueError("BL6 is not the six-view prefix of the BL8 catalog")

    def feature_names(self, level: int = 8) -> list[str]:
        self._level(level)
        return [f for view in VIEWS[:level] for f in self.transforms[view].features]

    @staticmethod
    def _level(level: int) -> None:
        if level not in (6, 8):
            raise ValueError("level must be 6 or 8")

    @staticmethod
    def _ids(frame: pd.DataFrame, id_column: str) -> pd.Series:
        if frame.empty:
            raise ValueError("Input contains no interface records")
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate input column names")
        if id_column not in frame or frame[id_column].isna().any():
            raise ValueError(f"Missing {id_column} identifiers")
        ids = frame[id_column].astype(str).str.strip()
        if ids.eq("").any() or ids.duplicated().any():
            raise ValueError("Identifiers must be non-empty and unique per interface")
        return ids.reset_index(drop=True)

    def lookup_states(
        self, frame: pd.DataFrame, level: int = 8, id_column: str = "record_id"
    ) -> pd.DataFrame:
        """Query complete state tuples, rejecting unknown or malformed states."""
        self._level(level)
        out = pd.DataFrame({id_column: self._ids(frame, id_column)})
        for view in VIEWS[:level]:
            column = f"{view}_state_code"
            if column not in frame:
                raise ValueError(f"Missing state column: {column}")
            values = frame[column].map(normalize_state).reset_index(drop=True)
            unknown = set(values) - set(self.names[view])
            if unknown:
                raise ValueError(f"Unknown states in {view}: {sorted(unknown)}")
            out[column] = values
            out[f"{view}_state_name"] = values.map(self.names[view])
        for n in (6, 8) if level == 8 else (6,):
            key = out[[f"{v}_state_code" for v in VIEWS[:n]]].agg("|".join, axis=1)
            out[f"bl{n}_state_tuple"] = key
            out[f"bl{n}_id"] = key.map(self.lookup[n]).fillna(f"BL{n}_OOV")
            out[f"bl{n}_status"] = np.where(
                key.isin(self.lookup[n]), "covered", "OOV_new_combination"
            )
        out["has_qc_state"] = out.interaction_state_code.eq("C2")
        if level == 8:
            out["has_qc_state"] |= out.aa_pair_contact_composition_state_code.eq("C-1")
            out["has_qc_state"] |= out.antibody_sequence_composition_state_code.eq("C-1")
        return out

    def project(
        self, frame: pd.DataFrame, level: int = 8, id_column: str = "record_id"
    ) -> pd.DataFrame:
        """Apply frozen preprocessing and nearest centers; never refit the space.

        BL6 needs 230 features. BL8 needs 328 and also returns BL6 membership.
        Optional QC-count columns override AA-pair/sequence QC inference.
        """
        self._level(level)
        ids = self._ids(frame, id_column)
        required = self.feature_names(level)
        missing = [f for f in required if f not in frame]
        if missing:
            raise ValueError(f"Missing {len(missing)} feature columns: {missing[:8]}")
        rows = frame.reset_index(drop=True)
        state_frame = pd.DataFrame({id_column: ids})
        diagnostics = {}
        for view in VIEWS[:level]:
            transform = self.transforms[view]
            x, imputed = transform.transform(rows)
            if view in BASE_VIEWS and (imputed == 1).any():
                raise ValueError(f"Entire {view} view missing for one or more records")
            centers = self.centers[view]
            distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2) / np.sqrt(
                x.shape[1]
            )
            nearest = distances.argmin(axis=1)
            codes = np.asarray(self.labels[view], dtype=object)[nearest]
            nearest_distance = distances[np.arange(len(rows)), nearest]
            qc = np.zeros(len(rows), dtype=bool)
            if view == VIEWS[6]:
                column = "aa_pair_total_close_pairs_qc"
                if column in rows:
                    counts = pd.to_numeric(rows[column], errors="raise")
                else:
                    cells = [
                        f
                        for f in transform.features
                        if f.startswith("aa_pair_all_") and "_to_" in f and f.endswith("_fraction")
                    ]
                    counts = (
                        rows[cells].apply(pd.to_numeric, errors="raise").sum(axis=1, min_count=1)
                    )
                if counts.isna().any() or (counts < 0).any():
                    raise ValueError(
                        "Missing or negative AA-pair contact QC; supply the observed count"
                    )
                qc = counts.eq(0).to_numpy()
            elif view == VIEWS[7]:
                column = "sequence_variable_residue_count_qc"
                if column in rows:
                    counts = pd.to_numeric(rows[column], errors="raise")
                    if counts.isna().any() or (counts < 0).any():
                        raise ValueError("Missing or negative variable-domain QC count")
                    qc = counts.eq(0).to_numpy()
                else:
                    qc = imputed == 1
            codes[qc] = "C-1"
            nearest_distance[qc] = np.nan
            state_frame[f"{view}_state_code"] = codes
            diagnostics[f"{view}_nearest_state_distance"] = nearest_distance
            diagnostics[f"{view}_imputed_fraction"] = imputed
        out = self.lookup_states(state_frame, level, id_column)
        for name, values in diagnostics.items():
            out[name] = values
        out["mapping_profile"] = "frozen_feature_projection"
        return out

    def describe(self) -> dict:
        return {
            "release": self.metadata["release"],
            "n_reference_records": 15396,
            "n_pdb_ids": 8273,
            "bl6_signatures": 185,
            "bl8_signatures": 1498,
            "views": {
                v: {
                    "inputs": len(self.transforms[v].features),
                    "fitted_centers": len(self.labels[v]),
                }
                for v in VIEWS
            },
            "membership": "exact state tuple in the frozen catalog",
        }


def load_reference_space(directory: str | Path | None = None) -> ReferenceSpace:
    return ReferenceSpace(DEFAULT_REFERENCE if directory is None else Path(directory))
