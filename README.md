# ML-Assisted PC-SAFT for Pure-Component VLE of Hydrocarbons

**ES 335: Machine Learning | IIT Gandhinagar**

[**Interactive research website**](https://naivvaghela-nv.github.io/es335-ml-pcsaft/) · [**GitHub source repository**](https://github.com/naivvaghela-nv/es335-ml-pcsaft)

## Overview

This project studies a structure-to-thermodynamics pipeline for pure hydrocarbons:

**SMILES → RDKit molecular descriptors → machine-learning model → PC-SAFT parameters → FeOs phase equilibrium → saturation pressure and saturated-liquid density**

The research question is whether molecular descriptors can predict the pure-component PC-SAFT parameters **segment number** \(m\), **segment diameter** \(\sigma\), and **dispersion energy** \(\epsilon/k\), and how errors in these estimated parameters affect downstream vapor–liquid equilibrium (VLE) predictions.

The goal is a transparent comparison of candidate ML methods, not a claim that ML surpasses physical property measurements.

## Results at a glance

The frozen final model is **Ridge regression (alpha = 1.0) with five RDKit descriptors**, using median imputation and standard scaling. It was selected on development-validation *saturation-pressure error*, before inspecting the held-out test set.

| Quantity | Result |
|---|---:|
| Curated molecules | 194 |
| Training and development molecules | 155 |
| Held-out test molecules | 39 |
| Final-test state points | 156 (4 temperatures per molecule) |
| Mean test saturation-pressure absolute percentage error | **20.54%** |
| Mean test saturated-liquid-density absolute percentage error | **2.66%** |
| Valid reference/predicted pressure comparisons | 156/156 |
| Valid reference/predicted density comparisons | 156/156 |

**Interpretation:** The pressure and density errors above are measured **against FeOs/PC-SAFT predictions using published fitted parameters**, *not independently measured experimental properties*. The 100% numerical success rate refers to solver convergence at the evaluated states, not to predictive accuracy.

## Data and scope

The starting parameter collection is **Esper2023**, distributed with the open-source [FeOs project](https://github.com/feos-org/feos). The study deliberately restricts its scope to **194 selected saturated, neutral, nonassociating C3–C20 hydrocarbons**. More complex molecular families, polar/associating parameterizations, and molecules outside this scope were not silently mixed into the dataset.

The molecule-wise split was fixed before final evaluation:

- **155 molecules** in the training/development pool.
- **39 molecules** held aside for one final assessment.
- Within the training pool, model development used **116 fitting / 39 development-validation molecules** for the downstream VLE comparison.

The frozen split should be preserved in `results/predictions/molecule_split.csv`.

## Feature engineering and models

The final model uses five molecular features: molecular weight, carbon count, ring count, rotatable bonds, and branch points. Research comparisons included **12**, **28**, and **156**-feature representations; the last includes 128 Morgan fingerprint bits. The experiments compared a mean-prediction baseline, **Ridge**, **Random Forest**, and **Extra Trees** models.

Model selection used development validation and the downstream thermodynamic objective rather than selecting whichever algorithm happened to have the smallest individual-parameter MAE.

| Development-validation model | Mean saturation-pressure APE |
|---|---:|
| Ridge + 5 features | **16.22%** |
| Ridge + 12 features | 17.96% |
| Extra Trees + 156 features | 25.91% |

**Main finding:** Richer molecular representations and lower PC-SAFT parameter prediction error do not guarantee lower saturation-pressure error. Local sensitivity calculations help explain why deviations in the three parameters may interact or compensate, but do not by themselves prove causality.

## Evaluation protocol and integrity

1. Curate and encode molecules and freeze molecule-wise train/test assignments.
2. Compare baselines and candidate model/feature configurations using the training pool.
3. Evaluate shortlisted models on the 39 development-validation molecules using reference and ML-predicted PC-SAFT parameters.
4. Select Ridge + 5 for the downstream pressure objective.
5. Fit the selected approach to all 155 original training molecules.
6. Run the frozen evaluation once on the 39 previously untouched test molecules and retain the resulting metrics and curves.

The four test temperatures for each molecule were set using fractions **0.65, 0.70, 0.75, and 0.80** of the **reference-PC-SAFT critical temperature**. This is an important qualification: the evaluation-temperature grid is not obtained solely from molecular structure.

The one-time final evaluator intentionally prevents overwriting existing frozen outputs. **Do not delete or overwrite the test results to rerun model selection on them.**

## Main limitations

- The principal reference is published-parameter PC-SAFT, not independent experimental measurements; consequently, this is **not an experimental accuracy benchmark**.
- The chemistry coverage is deliberately restricted to selected hydrocarbons, and the molecule-wise split is not a scaffold- or family-extrapolation test.
- The 39-molecule test set provides a finite-sample estimate, not a universal accuracy guarantee.
- Pressure errors are substantially larger than density errors; some individual molecules are difficult. For example, cycloheptane had an approximately **87%** mean pressure APE across its four tested conditions.
- Lower MAE on individual PC-SAFT parameters does not necessarily imply more accurate equilibrium properties.
- The final Ridge model's \(\sigma\) test-set **RMSE was slightly worse** than that of the mean-prediction baseline; the comparison should not be reduced to uniformly better performance on every metric.
- All 156 computed pairs converged, but convergence does not validate the physical model or experimental fidelity.

## Repository contents

The exact filenames can evolve during cleanup, but the following artifacts are central:

| Path | Purpose |
|---|---|
| `src/final_holdout_evaluation.py` | Frozen Ridge + 5 final evaluation, preflight and overwrite safeguard |
| `src/export_website_data.py` | Export saved experiments into the website's JSON data file |
| `src/compare_richer_models.py` | Descriptor/model comparison experiments |
| `src/compare_development_vle_models_all39.py` | Downstream development VLE evaluation |
| `results/predictions/molecule_split.csv` | Saved molecule-wise train/test assignments |
| `results/predictions/final_test_vle_curves.csv` | Frozen pressure/density comparisons |
| `results/metrics/final_test_parameter_metrics.csv` | Held-out parameter MAE/RMSE |
| `results/metrics/final_test_thermo_summary.csv` | Held-out VLE summary |
| `docs/` | Published static Carbon-style research dashboard |
| `docs/data/results.json` | Precomputed data read by the browser |
| `references/` | Methods, experiments, scientific findings, and disclosures |

## Local setup (Windows PowerShell)

The site itself is static, but reproducing the analysis requires a working scientific Python environment with RDKit, FeOs, `si_units`, NumPy, pandas, and scikit-learn.

```powershell
git clone https://github.com/naivvaghela-nv/es335-ml-pcsaft.git
cd es335-ml-pcsaft
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

If the upstream raw parameters are absent, create `data/raw/` and obtain `esper2023.json` from the FeOs upstream PC-SAFT parameters collection. Preserve the original record structure, molecule identifiers, and published parameter values. **A clean installation also requires the curated `data/processed/hydrocarbons_initial.csv`; consult the committed preprocessing scripts and saved data files rather than replacing it with arbitrary reconstructed labels.**

Inspect whether the final evaluator has its required inputs without changing frozen outputs:

```powershell
.\.venv\Scripts\python.exe src/final_holdout_evaluation.py --preflight
```

Regenerate the website's exported data using *existing experiment results* (this does not retrain ML or rerun FeOs):

```powershell
.\.venv\Scripts\python.exe src/export_website_data.py
```

Preview the published site locally:

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --directory docs
```

Open <http://127.0.0.1:8000/>. The dashboard reads saved JSON in the browser; it does **not** calculate PC-SAFT in JavaScript.

**Reproducibility note:** The listed scripts and results reflect the working project; independent reproduction from a completely fresh clone must still be verified against the files currently committed in this repository. Do not interpret these steps as a verified one-command end-to-end rebuild of the source parameter dataset.

## References and acknowledgments

- **FeOs**: <https://github.com/feos-org/feos> (PC-SAFT implementation and published-parameter collection).
- **Esper2023** PC-SAFT parameter records distributed by FeOs; see upstream file metadata for full bibliographic attribution.
- Related research and experiment rationale are documented in `references/`.

## AI assistance disclosure

AI assistance, including ChatGPT, was used in project planning, code development, debugging, evaluation support, documentation, and the interactive website. Numerical results reported here are from the executed scientific Python scripts and saved output files. AI-generated interpretations and implementations are not a substitute for checking the code and the underlying calculations. See `references/AI_USAGE.md` if present; ensure the description accurately represents the work performed.

---

**Project links:** [Live website](https://naivvaghela-nv.github.io/es335-ml-pcsaft/) · [GitHub repository](https://github.com/naivvaghela-nv/es335-ml-pcsaft)
