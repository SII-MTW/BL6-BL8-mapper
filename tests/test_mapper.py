"""Core tests do not require PyMOL, ANARCII, or raw structure files."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from bl8_mapper import VIEWS, load_reference_space

ROOT = Path(__file__).resolve().parents[1]


class MapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load_reference_space()
        cls.features = pd.read_csv(ROOT / "examples/feature_vectors.csv")
        cls.expected = pd.read_csv(ROOT / "examples/expected_assignments.csv")

    def test_catalog_counts_and_nesting(self):
        self.assertEqual(len(self.ref.feature_names(6)), 230)
        self.assertEqual(len(self.ref.feature_names(8)), 328)
        self.assertEqual(len(self.ref.lookup[6]), 185)
        self.assertEqual(len(self.ref.lookup[8]), 1498)
        self.ref.verify_assets()

    def test_all_catalog_rows_roundtrip(self):
        for n in (6, 8):
            catalog = self.ref.catalogs[n].copy()
            catalog["record_id"] = catalog[f"bl{n}_id"]
            result = self.ref.lookup_states(catalog, level=n)
            self.assertTrue(result[f"bl{n}_status"].eq("covered").all())
            self.assertEqual(result[f"bl{n}_id"].tolist(), catalog[f"bl{n}_id"].tolist())

    def test_valid_oov_is_not_nearest_catalog(self):
        states = list(product(*(self.ref.labels[v] for v in VIEWS)))
        oov = next(row for row in states if "|".join(row) not in self.ref.lookup[8])
        frame = pd.DataFrame(
            [{"record_id": "oov", **{f"{v}_state_code": s for v, s in zip(VIEWS, oov)}}]
        )
        result = self.ref.lookup_states(frame)
        self.assertEqual(result.bl8_status.iloc[0], "OOV_new_combination")
        self.assertEqual(result.bl8_id.iloc[0], "BL8_OOV")
        frame.loc[0, "interaction_state_code"] = "C999"
        with self.assertRaisesRegex(ValueError, "Unknown states"):
            self.ref.lookup_states(frame)

    def test_feature_fixture_and_determinism(self):
        result = self.ref.project(self.features)
        pd.testing.assert_frame_equal(result, self.ref.project(self.features))
        columns = ["record_id", "bl6_id", "bl6_status", "bl8_id", "bl8_status", "bl8_state_tuple"]
        pd.testing.assert_frame_equal(result[columns], self.expected[columns])

    def test_bl6_requires_only_230_inputs(self):
        frame = self.features[["record_id", *self.ref.feature_names(6)]]
        result = self.ref.project(frame, level=6)
        full = self.ref.project(self.features)
        self.assertEqual(result.bl6_state_tuple.tolist(), full.bl6_state_tuple.tolist())
        self.assertNotIn("bl8_status", result)

    def test_projection_never_mutates_input(self):
        frame = self.features.copy(deep=True)
        frame.loc[0, self.ref.feature_names()[0]] = np.inf
        before = frame.copy(deep=True)
        self.ref.project(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_missing_columns_and_invalid_identifiers(self):
        with self.assertRaisesRegex(ValueError, "Missing .*feature columns"):
            self.ref.project(self.features.drop(columns=self.ref.feature_names()[0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            self.ref.project(pd.concat([self.features.iloc[:1]] * 2))
        with self.assertRaisesRegex(ValueError, "no interface"):
            self.ref.project(self.features.iloc[:0])
        with self.assertRaisesRegex(ValueError, "level must"):
            self.ref.project(self.features, level=7)

    def test_missing_base_view_is_not_a_biological_state(self):
        frame = self.features.iloc[:1].copy()
        frame[list(self.ref.transforms[VIEWS[0]].features)] = np.nan
        with self.assertRaisesRegex(ValueError, "Entire .* view missing"):
            self.ref.project(frame)

    def test_qc_unresolved_states(self):
        frame = self.features.iloc[:1].copy()
        frame["aa_pair_total_close_pairs_qc"] = 0
        frame["sequence_variable_residue_count_qc"] = 0
        result = self.ref.project(frame)
        self.assertEqual(result.aa_pair_contact_composition_state_code.iloc[0], "C-1")
        self.assertEqual(result.antibody_sequence_composition_state_code.iloc[0], "C-1")
        self.assertTrue(result.has_qc_state.iloc[0])
        self.assertTrue(pd.isna(result.aa_pair_contact_composition_nearest_state_distance.iloc[0]))

    def test_unavailable_aa_matrix_requires_qc_evidence(self):
        frame = self.features.iloc[:1].copy()
        columns = list(self.ref.transforms[VIEWS[6]].features)
        frame[columns] = np.nan
        if "aa_pair_total_close_pairs_qc" in frame:
            frame = frame.drop(columns="aa_pair_total_close_pairs_qc")
        with self.assertRaisesRegex(ValueError, "AA-pair contact QC"):
            self.ref.project(frame)

    def test_cli_example(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "mapped.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bl8_mapper",
                    "features",
                    "--input",
                    str(ROOT / "examples/feature_vectors.csv"),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.exists())
            self.assertEqual(sum(json.loads(completed.stdout).values()), len(self.features))

    def test_english_source_and_no_workstation_paths(self):
        for path in ROOT.rglob("*"):
            if not path.is_file() or any(
                p in path.parts for p in (".git", "__pycache__", "dist", "build", ".ruff_cache")
            ):
                continue
            if path.suffix not in {".py", ".md", ".json", ".csv", ".toml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8-sig")
            self.assertFalse(any("\u4e00" <= c <= "\u9fff" for c in text), str(path))
            self.assertNotIn("C:\\Users\\", text, str(path))


if __name__ == "__main__":
    unittest.main()
