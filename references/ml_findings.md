# ES 335 Project: ML Findings and Dataset Audit

**Project:** ML-Assisted PC-SAFT for Pure-Component VLE of Hydrocarbons  
**Status:** Historical development findings; final held-out evaluation completed.  
**Recorded:** 2 November 2026

## Finding 1: Molecular representation and learning algorithm interact

In three-fold cross-validation restricted to the original 155-molecule training pool, the combination of **Extra Trees and 156 input features** (28 physicochemical/topological descriptors plus 128 chirality-aware Morgan fingerprint bits) achieved lower mean absolute error (MAE) for all three predicted PC-SAFT parameters than Extra Trees using 28 descriptors alone:

| PC-SAFT target | Extra Trees: 28 features | Extra Trees: 156 features | MAE reduction |
|---|---:|---:|---:|
| Segment number, m | 0.40414 | 0.28474 | 29.5% |
| Segment diameter, sigma (Angstrom) | 0.08311 | 0.06848 | 17.6% |
| Dispersion energy, epsilon/k (K) | 8.91469 | 7.62831 | 14.4% |

By contrast, **Ridge regression with 156 features was worse than Ridge regression with 12 features for all three parameters**:

| PC-SAFT target | Ridge: 12 features | Ridge: 156 features |
|---|---:|---:|
| m | 0.34703 | 0.43646 |
| sigma (Angstrom) | 0.09243 | 0.10772 |
| epsilon/k (K) | 10.15147 | 10.26883 |

**Report-ready interpretation:** The results indicate that richer molecular representations can benefit a nonlinear ensemble such as Extra Trees, while the same higher-dimensional representation may impair a linear Ridge model. Increasing feature dimensionality does not guarantee better generalization; model choice and representation should be evaluated together. The strongest observed parameter-prediction configuration in this comparison was Extra Trees with 28 descriptors and 128-bit Morgan fingerprints.

**Scientific caution:** These are training-only three-fold cross-validation results after comparing multiple configurations, not final held-out test performance. The 156-feature set differs from the 12-feature set in *both* additional descriptors and fingerprints; fingerprint contribution is more directly assessed by comparing 28 versus 156 features for the *same* algorithm. Differences do not prove causal superiority across other datasets, and lower parameter MAE does not by itself imply lower saturation-pressure error.

## Finding 2: Structural descriptors reduce feature collisions

Among the 155 molecules in the frozen original training pool:

| Molecular representation | Molecules sharing feature vectors | Distinct feature vectors |
|---|---:|---:|
| Original 5 descriptors | 77 | 106 |
| Enhanced 12 descriptors | 11 | 149 |
| Richer 28 descriptors | 11 | 149 |
| Richer 28 + Morgan128 (156 features) | 3 | 153 |

**Interpretation:** The extra seven descriptors greatly improved the model's ability to distinguish constitutional isomers. Adding 16 more numerical descriptors did not reduce exact collisions further, but the chirality-aware Morgan fingerprint bits reduced collision-group membership from 11 to 3. Fewer collisions do not prove better predictive accuracy; the cross-validation comparison provides separate evidence about prediction performance.

## Finding 3: Dataset-expansion audit and scope decision

A read-only audit of the Esper2023 parameter file reported **1,842 total records** and **233 new unique candidate structures** outside the curated initial set of 194 molecules:

| Expansion category | Candidate count |
|---|---:|
| Saturated hydrocarbons with 21+ carbons | 21 |
| Unsaturated/aromatic hydrocarbons with 3-20 carbons | 210 |
| Unsaturated/aromatic hydrocarbons with 21+ carbons | 2 |

The audit did not report extra saturated C3-C20 candidates. Counts are *candidates*, not a verified training-set expansion. Additional excluded or repeated records included 1,235 flagged for extra EOS terms, 194 already in the initial dataset, 5 with fewer than three carbons, 1 isolated-hydrogen case, 16 multiple-fragment cases, and 158 non-carbon/hydrogen records (as reported by the audit script). Exclusion counts reflect that script's processing logic, not necessarily mutually exclusive chemical categories.

**Recommended scope:** Retain the original curated 194-molecule population until the final held-out evaluation is complete. Adding unsaturated/aromatic chemicals broadens the modeling question; adding C21+ saturated hydrocarbons changes the size regime. Any expansion would require chemistry and parameter-schema checks, duplicate checks, and a clearly documented evaluation protocol. Do not merge these 233 candidates automatically or contaminate the reserved 39-molecule test set.

## Existing EOS milestone

The initial development FeOs integration reported 48/48 comparable pure-component saturation-pressure calculations on 12 development-validation molecules (four temperatures each) using reference and ML-predicted parameters, with a roughly 11.15% mean of the 12 molecule-level mean absolute percentage errors. This is comparison with the **published-parameter PC-SAFT calculations**, not experimental vapor-pressure accuracy. The stronger models have not yet been shown to improve this VLE result.

## Provenance and next required checks

Outputs reported from the user's Windows PowerShell runs:

- `results/metrics/richer_model_comparison_cv.csv`: 3-fold CV for Ridge, Random Forest, and Extra Trees across 12/28/156 features.
- `results/metrics/enhanced_descriptor_cv.csv`: earlier five-/twelve-feature comparison.
- `results/metrics/expansion_candidate_audit.csv`: read-only candidate audit.
- `data/processed/hydrocarbons_richer.csv`: generated feature dataset (may be gitignored; regenerate using `src/build_richer_features.py`).
- `results/predictions/molecule_split.csv`: original frozen 155/39 split.
- `results/metrics/development_vle_summary.csv`: initial development VLE comparison.

Next, compare promising models via FeOs using identical development-validation molecules and temperature grids; freeze the chosen model; train on all 155 original training-pool molecules; evaluate the 39 held-out test molecules once; and only then, if useful, train a separate deployment model on all 194 molecules. Preserve failures and document uncertainty.

This note records findings for the final submission; it does **not** claim that the final test set or independent experimental validation has been completed.

## Final evaluation update - November 2026

The development-stage analyses above were performed before
the independent held-out evaluation.

The frozen final Ridge model used five molecular descriptors
and was trained on all 155 original training-pool molecules.

Final held-out results:
- Test molecules: 39
- Thermodynamic state points: 156
- Mean saturation-pressure APE: 20.5448%
- Mean saturated-liquid-density APE: 2.6637%
- Successful pressure comparisons: 156/156
- Successful density comparisons: 156/156

Reported errors measure agreement with published-parameter
PC-SAFT calculations, not independent experimental accuracy.

The final test results were not used for model selection.
