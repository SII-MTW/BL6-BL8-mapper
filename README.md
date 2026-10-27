# BL6 / BL8 Frozen Interface Mapper

Map an antibody-antigen interface into the frozen BL6 and BL8 reference catalogs
and report whether its exact state combination is **covered** or **OOV**.
This repository contains mapping code and complete reference dictionaries only.
It has no plotting code, manuscript files, target-specific analyses, or training
and reclustering commands.

## Reference space

The frozen SAbDab-derived reference contains **15,396 interface records from
8,273 PDB entries**. One input row represents one explicitly selected interface,
not an entire multi-interface PDB file.

| Representation | Views | Numerical inputs | Observed signatures |
|---|---|---:|---:|
| BL6 | Topology; shape/geometry; interaction; CDR conformation; B-factor descriptors; SASA/interface area | 230 | 185 |
| BL8 | The six BL6 views plus AA-property-pair contacts and antibody sequence composition | 328 | 1,498 |

BL6 is the **first six states of BL8**, using the same frozen state assignments.
BL8 adds 64 AA-pair inputs and 34 sequence inputs. The sequence view describes
antibody amino-acid property composition, not antigen identity or sequence
similarity. Catalogs are observed combinations, not all mathematically possible
combinations of view states.

### What covered and OOV mean

For each view the mapper applies the stored preprocessing and chooses the
nearest frozen center. It concatenates the states in the documented view order
and performs an **exact tuple lookup** in the relevant catalog.

- `covered`: the complete tuple exists in the corresponding reference catalog.
- `OOV_new_combination`: the states are valid, but their combination is absent.
- An invalid structure or failed CDR annotation is an **error**, not OOV.
- `has_qc_state` flags clash-rich interaction or unresolved additional views.
  QC tuples can be present in the catalog; covered alone is not a quality gate.

There is no nearest-signature substitution, target-specific refitting, model
confidence threshold, or learned binder/nonbinder classifier. A BL8 OOV can
still be BL6 covered. Neither membership nor OOV proves binding, correctness,
novelty of mechanism, or affinity.

## Install

Python 3.10 or later is required. Feature and state-table mapping only need
NumPy and pandas:

```bash
python -m pip install .
bl8-map validate
```

Structure extraction additionally requires Gemmi, SciPy, ANARCII and an importable
`pymol2` installation. A separate environment is recommended; for example:

```bash
conda create -n bl8-map -c conda-forge python=3.11 pymol-open-source pip
conda activate bl8-map
python -m pip install '.[structures]'
python -c "import gemmi, scipy, anarcii, pymol2"
```

PyMOL is installed separately because its available binary distribution depends
on the operating system. ANARCII may obtain its model weights on first use;
follow its upstream installation instructions for offline environments.
Core catalog queries do not load ANARCII, PyMOL, PyTorch, or scikit-learn.

## Map a PDB or mmCIF structure

Select the antigen, heavy variable chain, and optional light variable chain
explicitly. The example chain labels below are examples, not automatic defaults:

```bash
bl8-map structures --structure complex.cif --antigen-chain A --heavy-chain B --light-chain C --output results/query.csv
```

For a VHH, omit `--light-chain`. For multiple models in a file, `--model-index 1`
selects the first coordinate model; use a separate record for each desired
interface. Do not infer all biological interfaces from a crystal assembly.

**Measurement scope:** structure extraction implements the corrected
prediction-compatible profile used for the study's target projections. It does
not reconstruct the original experimental multi-patch/strict-hydrogen-bond
pipeline. In particular, hydrogen-bond fields use a heavy-atom element/distance
heuristic without D-H-A angles, geometry uses a whole-interface patch, and
secondary-structure/water fields retain prediction-profile placeholders.
See [measurement_contract.md](docs/measurement_contract.md). Use independently
computed reference-compatible inputs through the feature-table interface when
strict experimental-method equivalence is required.

Supported structure selections are one protein antigen chain and separate
Fv heavy/light variable domains or one VHH domain. Full Fab/IgG chains, fused
scFv domains, non-protein antigens, and multi-chain antigen selections require
preprocessing and are not silently interpreted as equivalent inputs. Supply
only the desired variable-domain coordinates while retaining antigen contacts.
Incomplete or incorrectly assigned antibody domains fail annotation rather
than becoming framework-only interfaces.

For a batch, copy [examples/structures.csv](examples/structures.csv), replace
the example file paths and chain labels, and run:

```bash
bl8-map structures --manifest examples/structures.csv --output results/batch.csv
```

Paths in a manifest are relative to that manifest. Output files are:

- `batch.csv`: state codes/names, BL6/BL8 IDs, membership and distance diagnostics.
- `batch.features.csv`: extracted numerical inputs and available QC counts.
- `batch.errors.csv`: explicit failed records; failures receive no signature.
- `batch.run.json`: input SHA-256 hashes, chain/residue mappings, software
  versions, reference identity, and measurement notes.

A batch with any failed records exits with code 2 while retaining successful
results. Existing output files are protected unless `--overwrite` is passed.
Optional `--level 6` computes BL6 assignments only; the current structure
extractor still computes the shared feature set.

## Map a precomputed feature table

The input CSV needs a unique `record_id` and all numerical columns for the
chosen level. Required features and their definitions are in
[feature_dictionary.csv](src/bl8_mapper/reference_space/feature_dictionary.csv).
Missing columns cause errors; finite-feature gaps use frozen reference medians.
An entirely missing base view is rejected. QC-state handling is documented in
[reference_space.md](docs/reference_space.md).

```bash
bl8-map features --input examples/feature_vectors.csv --output results/example.csv
bl8-map features --input bl6_features.csv --level 6 --output results/bl6.csv
```

Python API:

```python
import pandas as pd
from bl8_mapper import load_reference_space

reference = load_reference_space()
features = pd.read_csv("examples/feature_vectors.csv", dtype={"record_id": str})
assignments = reference.project(features, level=8)
print(assignments[["record_id", "bl6_id", "bl6_status", "bl8_id", "bl8_status"]])
```

For structure input from Python:

```python
from bl8_mapper.structures import map_structure, close_structure_session

try:
    assignments, features, provenance = map_structure(
        "complex.cif", antigen_chain="A", heavy_chain="B", light_chain="C"
    )
finally:
    close_structure_session()
```

## Query state tuples directly

An input state CSV must contain `record_id` and the `*_state_code` columns in
the catalog. Unknown states are errors, not new combinations:

```bash
bl8-map states --input examples/state_queries.csv --output results/lookup.csv
```

Do not use only the displayed biological label, a 2D visualization, or a
distance cutoff as the catalog membership test.

## Complete dictionaries and numerical assets

All assets are packaged under [src/bl8_mapper/reference_space](src/bl8_mapper/reference_space):

| File | Contents |
|---|---|
| `bl6_catalog.csv` | All 185 BL6 IDs, tuples, view labels and reference support |
| `bl8_catalog.csv` | All 1,498 BL8 IDs, tuples, view labels, support and linked BL6 IDs |
| `state_dictionary.csv` | Every view state, biological label, support, and QC flag |
| `feature_dictionary.csv` | Exactly 328 included inputs with calculation, unit and interpretation |
| `*_state_centers_z.csv` | Eight frozen nearest-center models |
| `preprocessing_stats.json` | Ordered features, clipping bounds, medians, means and scales |
| `aa_pair_state_dictionary.csv`, `sequence_state_dictionary.csv` | Extended-view interpretation and discriminating features |
| `release_metadata.json`, `checksums.json` | Release identity, provenance and asset checksums |

Historical `formal_*` numerical column names are retained as immutable feature
identifiers so existing feature tables remain usable. The public view is called
`interaction`. No `hbond_like` field is used as a mapping input.

## Validation and development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m ruff check .
python -m ruff format --check .
python -m build
```

Tests query the complete catalogs, check BL6/BL8 nesting, distinguish valid OOV
from malformed states, exercise QC/missingness handling and run the example CLI.
[validation_report.json](docs/validation_report.json) records the release audit
against all 15,396 reference records and selected existing corrected structure
projections. Reference KMeans training labels are not silently rewritten when
a nearest-center query differs near a training boundary; see the audit counts.
In this audit, 15,386/15,396 complete tuples matched the historical training
labels exactly; all 15,396 remained covered. All 328 inputs and the resulting
BL8 tuple matched the existing corrected mapping for each of the three tested
target structures. These bounded checks do not establish equivalence to the
experimental feature-extraction pipeline.

## License and citation

Code and bundled derived mapping assets are distributed under the MIT license
in [LICENSE](LICENSE). PyMOL, ANARCII and other dependencies retain their own
licenses. Raw structures and model-native prediction outputs are not included.
Please cite the accompanying BL8 study when its bibliographic record becomes
available; no manuscript DOI is invented in this release.
