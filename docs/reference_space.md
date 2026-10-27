# Frozen projection and catalog membership

## View order and input counts

| Position | Identifier | Inputs | Fitted centers | Additional QC state |
|---|---|---:|---:|---|
| 1 | `interface_topology` | 36 | 5 | None |
| 2 | `interface_shape_geometry` | 47 | 2 | None |
| 3 | `interaction` | 24 | 3 | Clash-rich C2 is a fitted QC center |
| 4 | `cdr_conformation` | 84 | 3 | None |
| 5 | `flexibility_disorder` | 14 | 2 | None |
| 6 | `solvent_interface_area` | 25 | 2 | None |
| 7 | `aa_pair_contact_composition` | 64 | 8 | C-1: no close contact |
| 8 | `antibody_sequence_composition` | 34 | 2 | C-1: unmapped variable domain |

The first six views contain 230 inputs. BL8 contains 328 inputs. Raw state
labels such as C0 are local to a view and must not be compared across views.
Biological state names are descriptive summaries, not the computational key.
Historical labels containing "disorder" or "Hbond" do not independently
establish dynamics or strict hydrogen-bond geometry for a query structure.

## Numerical mapping

For each view, ordered features are clipped to the stored first/99th-percentile
bounds, missing entries are filled with stored clipped-reference medians, and
the result is standardized using stored means and population standard deviations.

```text
x_clipped[j] = min(upper[j], max(lower[j], x[j]))
x_filled[j]  = median[j] if x[j] is missing/nonfinite else x_clipped[j]
z[j]        = (x_filled[j] - mean[j]) / scale[j]
d(k)        = sqrt(sum_j((z[j] - center[k,j])**2) / n_view_features)
state       = argmin_k d(k)
```

Ties use the first stored center. No query data are used to estimate transforms
or centers. Per-view distances are RMS standardized distances; they are not
binding probabilities or an alternative definition of OOV. Clipping limits
their interpretation for values far outside the training range.

BL6 concatenates the first six states; BL8 concatenates all eight states. Exact
lookup in `bl6_catalog.csv` or `bl8_catalog.csv` determines membership. Thus a
covered BL8 must also have a covered BL6 prefix. BL6 coverage does not imply BL8
coverage. A query can assign known marginal states and still be OOV.

## QC and missing data

Use observed optional `aa_pair_total_close_pairs_qc` and
`sequence_variable_residue_count_qc` columns whenever available. A zero count
assigns C-1 for that view without claiming a biological cluster. Negative or
missing supplied counts are errors. If counts are absent, AA-pair QC is
inferred from the sum of the 25 all-contact matrix fractions; sequence QC is
inferred only when the entire sequence view is missing.

No-close-contact matrix zeros are an explicit QC encoding. A completely
unavailable AA-pair matrix must be accompanied by an observed QC count rather
than silently being converted to no contact. An entirely missing base view
is rejected. Partial missingness uses the frozen median and is exposed in
`*_imputed_fraction`. A structure with no 4.5-A interface contacts is rejected
by the structure extractor, rather than assigned a fictitious biological state.

Clash-rich C2 and C-1 combinations remain in the complete reference catalogs,
because these files describe the published full catalog. `has_qc_state` must
be checked separately before interpreting coverage as a usable pose.

## Stable IDs and reference support

BL6 and BL8 IDs preserve the existing manuscript catalog identifiers. Tuples
are normalized to pipe-separated C-prefixed states. `n_records` counts retained
interface records, while `n_pdb_ids` counts unique supporting PDB entries for
that signature. Do not sum signature-level PDB counts to obtain the global
unique PDB denominator. BL6 PDB support comes from the original BL6 dictionary,
not a sum of BL8 support counts. `effective_pdb_support` is the original
PDB-weighted support, not an independent-structure count.

The mapping space includes state centers and preprocessing, not merely a 2D
plot. Manifold coordinates are not necessary for membership tests and are not
distributed in this minimal package.
