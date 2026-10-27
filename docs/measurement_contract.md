# Structure projection measurement contract

The frozen reference catalog and transforms are unchanged. The structure entry
point reproduces the corrected prediction-side feature convention, with explicit
chain selection and strict ANARCII IMGT annotation. It is a projection profile;
it does not claim to regenerate every experimental reference feature from raw
SAbDab coordinates. The feature-table entry point can accept a separately
computed reference-compatible feature table without these approximations.

## Supported entities and deterministic selection

Supply one protein antigen chain, one heavy variable chain and optionally one
separate light variable chain. Chains must be disjoint. Multiple coordinate
models are selected by a one-based model index. Each submitted manifest record
is one interface. Extra chains are excluded before SASA and contact calculations.
No crystal symmetry expansion or automatic biological-assembly inference is
performed. Non-protein antigens, multi-chain antigens, full constant domains,
fused scFv chains and ambiguous chain IDs are not supported in this profile.

Canonical internal chains A/B/C preserve selected coordinates but renumber
supported CA-containing protein residues sequentially. Residue mappings preserve
original chain, residue number/insertion code and residue name in the run JSON.
Hydrogen/deuterium atoms are omitted; zero-occupancy atoms are removed. The
highest-occupancy alternative for each atom name is retained, with blank/A
preferred on a tie. MSE is classified as methionine. Other unrecognized residue
types and non-protein entities are excluded explicitly by this profile.

ANARCII is called with `number(...)` followed by `to_scheme("imgt")` on the
numberer. Query amino acids are checked against the original sequence position
by position. IMGT CDR ranges are 27-38, 56-65 and 105-117, including insertions
at those positions. All three CDRs must be annotated on each selected chain.
No exception is converted into a framework-only map. The selected heavy and
light roles must agree with ANARCII H and K/L classification.
On Windows, cached numbering runs in a separate worker process so that PyTorch
and PyMOL do not load conflicting OpenMP runtimes into the same process. The
mapper does not enable the unsafe duplicate-OpenMP workaround.

## Prediction feature conventions and reference differences

| Component | Structure projection profile | Limitation relative to experimental reference calculation |
|---|---|---|
| Contact events | Heavy-atom pairs within 4.5 A; close events within 3.5 A; clash events within 2.0 A | Counts are atom events unless a feature explicitly says residue pairs |
| Hydrogen-bond fields | Element/distance heuristic for N/O/S within 3.5 A, with acidic/basic atom exclusions and backbone/backbone exclusion | No explicit-H placement, donor-parent check or D-H-A angle; not strict hydrogen bonds |
| Other interaction fields | Frozen distance/residue/atom flags evaluated on the same 4.5-A candidate pool | Nominal aromatic/cation-pi radii above 4.5 A do not expand the candidate pool; no ring-orientation criterion |
| Geometry | Whole-interface CA points and one interface patch; means/medians of the global proxy coincide | Original reference includes aggregation over experimental local patches |
| Approach angle | Missing, followed by frozen reference median imputation | No approach-angle measurement is inferred |
| CDR geometry | CA contour, endpoint distance, extension, turning, PCA and protrusion proxies | Not a dynamics or conformational-energy calculation |
| Secondary structure | Prediction convention initializes measurable loops to coil fractions | No DSSP or `cmd.dss` calculation is performed by this profile |
| B-factor descriptors | Input CA B values and CDR-vs-framework standardized values | pLDDT-like prediction confidence is not experimental thermal motion |
| SASA/BSA | Selected protein subcomplex; PyMOL dot_solvent=1, dot_density=1; BSA=(S_ab+S_ag-S_complex)/2 | Depends on the stored PyMOL build/radii; hydrogens are not added |
| Water terms | Historical compatibility zeros | No water-bridge analysis; zero is not evidence of water absence |
| AA-pair contacts | Directed 5x5 property-class fractions, weighted by close atom-contact events; all-interface and CDR scopes | Not an unweighted residue-residue frequency matrix |
| AA-pair entropy | Existing prediction convention normalizes by occupied property bins | The reference generator used the fixed full matrix convention; this discrepancy is retained and disclosed |
| Sequence | Property fractions across selected antibody chains and their CDR/framework partitions | Use isolated variable-domain chains; not antigen sequence identity |

Property groups are basic R/K/H, acidic D/E, polar N/Q/S/T/C, hydrophobic
A/V/I/L/M/G/P, and aromatic F/Y/W. The same residue can meet a different
interaction-specific definition (for example, histidine in aromatic contact
flags). These conventions are frozen and are not redefined during projection.

## Auditability

The numerical extraction core is mechanically derived from the existing
corrected target-mapping workflow. It does not depend on workstation paths,
target names, plotting, native model metrics or unpublished model weights.
Internal `formal_*` input names are schema identifiers retained for compatibility;
they are not claims about the measurement type. Run JSON records make the
prediction-compatible profile and its approximations visible for every run.

Reference support is finite and target coverage is uneven. A catalog lookup
cannot decide whether an OOV reflects a valid unobserved interface, a measurement
shift, a QC problem, or a physically implausible pose. The code deliberately
does not turn OOV into a positive/negative binding prediction.
