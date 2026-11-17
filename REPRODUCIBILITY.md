# Reproducibility guide: macOS, Ubuntu, and Arch Linux

**Project:** ML-Assisted PC-SAFT for Pure-Component VLE of Hydrocarbons  
**Course:** ES335, IIT Gandhinagar  
**Repository:** <https://github.com/naivvaghela-nv/es335-ml-pcsaft>  
**Website:** <https://naivvaghela-nv.github.io/es335-ml-pcsaft/>

> **Verification status:** This guide describes the repository's recorded pipeline and provides executable checks. A cross-OS clean-clone run has **not yet been confirmed**. Do not label macOS, Ubuntu, or Arch Linux as "tested" until the commands below pass on that system. The grader can reproduce the saved website or the scientific calculations only if the required files are actually committed or retrievable.

## 1. What counts as reproduction?

There are **three separate tasks**, with different dependencies:

| Level | What you reproduce | Command / evidence |
| --- | --- | --- |
| A. Published website | Serve the same static dashboard and its saved numerical outputs | `python -m http.server 8000 --directory docs` and visit `http://127.0.0.1:8000/` |
| B. Frozen-data audit | Verify the 194-molecule dataset, saved 155/39 split, reference source, SHA-256 hashes and 156 saved VLE comparisons | `python scripts/verify_reproducibility.py check --scope science` |
| C. Independent scientific recomputation | Fit the **already selected** Ridge-5 model on the original 155 training molecules and run it on the same 39 frozen test molecules in a disposable directory, then compare results | `python scripts/verify_reproducibility.py recompute` |

**Level C is the scientific reproducibility check.** Merely displaying a saved website is not equivalent to rerunning the scientific pipeline. A fresh rerun must use the same input records, split, descriptor columns, model, and version-compatible FeOs implementation.

The study compares predicted PC-SAFT results to *published-parameter PC-SAFT*, not independent laboratory measurements. The four state temperatures are 0.65, 0.70, 0.75 and 0.80 of each reference-PC-SAFT critical temperature.

## 2. OS-specific prerequisites

Use a **64-bit Intel/AMD Linux** or **Intel / Apple Silicon macOS** machine with internet access for the initial installation. Python 3.11 is used consistently to avoid Arch Linux's rolling system Python version. FeOs and `si-units` publish CPython ABI-compatible wheels for these macOS and x86-64 Linux platforms. Linux ARM64 is **not covered by these prebuilt-wheel instructions** and may require compilation. See the [FeOs distribution](https://pypi.org/project/feos/0.9.6/) and [si-units distribution](https://pypi.org/project/si-units/). 

### macOS (Intel or Apple Silicon)

Open Terminal. Install [Homebrew](https://brew.sh/) first if it is not installed; then:

```bash
brew install git uv
```

If you do not use Homebrew, install `git` through Xcode Command Line Tools (`xcode-select --install`) and install `uv` using [Astral's official instructions](https://docs.astral.sh/uv/getting-started/installation/).

### Ubuntu 22.04 / 24.04 (x86-64)

```bash
sudo apt update
sudo apt install -y git curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing `uv`, **open a new terminal** so its installation location is on your `PATH`, or follow the printed instructions from the installer.

### Arch Linux (x86-64)

```bash
sudo pacman -Syu --needed git curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing `uv`, open a new terminal. **Do not install project Python packages into Arch's system Python**; use the isolated environment below.

### Confirm tool availability on any OS

```bash
git --version
uv --version
uname -m
```

For Linux, `uname -m` should report `x86_64` for the prebuilt FeOs wheels used here. For macOS, `x86_64` or `arm64` is supported by the cited release. If an installer fails, fix that first rather than continuing with missing dependencies.

## 3. Clean checkout and pinned interpreter

Run on **all three operating systems**:

```bash
git clone https://github.com/naivvaghela-nv/es335-ml-pcsaft.git
cd es335-ml-pcsaft
git rev-parse HEAD
uv python install 3.11
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
python --version
```

**Record the Git commit hash** from `git rev-parse HEAD` with your test results. The same commit must be used when comparing platforms.

Install dependencies from the repository's **actual** requirement file:

```bash
test -f requirements.txt || { echo 'FAIL: requirements.txt is missing'; exit 1; }
uv pip install -r requirements.txt
python -m pip check
```

Expected: `pip check` reports that no broken requirements were found. Do not replace `requirements.txt` with a guessed collection of newer packages; model outputs can change when library or FeOs versions change.

Check the scientific imports and versions:

```bash
python -c "import feos, si_units, rdkit, numpy, pandas, sklearn, joblib; print('Scientific imports: OK')"
python -m pip freeze
```

The pip package corresponding to Python module `si_units` is named **`si-units`**. The correct RDKit pip package is **`rdkit`**. FeOs and RDKit document cross-platform pip installation; this guide deliberately uses the repository's requirement versions rather than overriding them. If a native wheel is unavailable, stop and document the environment-specific problem instead of claiming reproduction passed. [FeOs](https://github.com/feos-org/feos) · [RDKit](https://www.rdkit.org/docs/Install.html) · [uv](https://docs.astral.sh/uv/concepts/python-versions/)

## 4. Required inputs: the most important step

The evaluator uses **exactly these three input files**:

```text
data/raw/esper2023.json

data/processed/hydrocarbons_initial.csv

results/predictions/molecule_split.csv
```

It also uses `src/final_holdout_evaluation.py` and the saved `results/metrics/final_test_config.json` to validate the original data checksums. These are not optional.

From a **fresh clone**, before conducting any new calculations, run:

```bash
python scripts/verify_reproducibility.py check --scope science --no-imports
```

The command deliberately **fails and prints missing filenames** if the repo lacks these inputs. That is useful evidence: do not hide a missing dataset by constructing an approximate substitute.

### Recover only the upstream Esper2023 JSON

The published source is FeOs `parameters/pcsaft/esper2023.json` ([official parameter directory](https://github.com/feos-org/feos/tree/main/parameters/pcsaft)). If your project did not commit the raw file, and `final_test_config.json` exists, run:

```bash
python scripts/verify_reproducibility.py fetch-raw
```

This downloads the upstream JSON **only if its SHA-256 matches the hash saved by the final run**. If it prints `HASH MISMATCH`, upstream may have changed. Retrieve the exact historical file used for the original evaluation; do not edit your saved config to match new data.

### Recover the curated hydrocarbon CSV

**This file cannot reliably be recreated from the upstream JSON with a generic command.** Your original 194-molecule selection, canonicalization, filtering, and descriptor generation must be preserved.

Choose one verifiable route **before submitting**:

1. **Preferred when license and file size permit:** Commit `data/processed/hydrocarbons_initial.csv` and the fixed molecule split to the public repo with source attribution, so both exist in a clean clone.
2. Otherwise, commit the complete deterministic preprocessing script **plus the exact raw source version**, document its command, and verify that it regenerates a byte-identical CSV matching `source_dataset_sha256` in `final_test_config.json`.

If neither route works, the project is **not fully reproducible** even if the dependency installation works. Do not claim otherwise.

To see whether the files are tracked locally:

```bash
git ls-files -- data/processed/hydrocarbons_initial.csv
git ls-files -- results/predictions/molecule_split.csv
git ls-files -- results/metrics/final_test_config.json
git ls-files -- results/predictions/final_test_vle_curves.csv
```

Each command should print a path. If a file is ignored, investigate `git check-ignore -v <path>`; add it only after reviewing its contents and distribution permissions.

## 5. Run the audit

Once data and dependencies are available:

```bash
python scripts/verify_reproducibility.py check --scope science
python src/final_holdout_evaluation.py --preflight
```

The second command should print a result equivalent to:

```text
PRECHECK PASS: 155 train, 39 frozen held-out; all 39 reference records match
No model fitted and no final test calculations performed.
```

The standalone checker confirms the saved file hashes, 194 distinct molecules, 155/39 molecule-wise split, 39 held-out molecules, 156 VLE rows, 156/156 comparable pressure/density states, and summary errors near **20.54% pressure** and **2.66% density**. These figures refer to the original saved run and are checked rather than silently regenerated.

## 6. Independently recompute the frozen evaluation (safely)

**Never run `src/final_holdout_evaluation.py --run-final` inside the submitted project folder.** That evaluator contains an intentional guard against overwriting existing final-test results.

Instead run:

```bash
python scripts/verify_reproducibility.py recompute
```

The helper:

1. Validates the three original input hashes and the saved frozen outputs.
2. Creates a **temporary separate project directory** and copies only the frozen inputs and original evaluator into it.
3. Runs the original evaluator's `--preflight`, then its `--run-final` mode with the original Ridge-5 specification.
4. Verifies 39 molecules, 156 state comparisons, matching state IDs, and pressure/density MAPE within **0.5 percentage point** of the recorded results.
5. Leaves **all committed frozen outputs untouched**.

A successful run finishes with:

```text
PASS: independent isolated rerun matches the original 39-molecule evaluation
```

The 0.5 percentage-point tolerance is a *cross-environment numerical acceptance rule*, not a claim that results will always differ by that much. Exact equality is preferable. Compare all per-molecule details if meaningful differences remain.

If a grader needs output files for inspection, run in a **new, empty, separate directory**:

```bash
python scripts/verify_reproducibility.py recompute --keep-dir ../es335-reproduction-output
```

Do **not** use the frozen test outcomes to select a new model or descriptor count. This optional rerun verifies an already frozen model, not a new development experiment.

## 7. Reproduce the saved interactive website

First check the committed site artifacts:

```bash
python scripts/verify_reproducibility.py check --scope site
```

Then serve the website from the repository root:

```bash
python -m http.server 8000 --directory docs
```

Visit <http://127.0.0.1:8000/> and confirm that the site shows **39** held-out molecules, **20.54%** pressure error, **2.66%** density error, and **156/156** valid comparisons. Interact with molecule selection, pressure/density switching, the data table, and the methodology Q&A. Use Ctrl+C in the terminal to stop the server.

### Re-export website JSON from saved experiment outputs (separate optional test)

The static site only needs its committed `docs/data/results.json`. To **regenerate** that file, additional model-comparison data and `hydrocarbons_richer.csv` are required. Check them before running the exporter:

```bash
python scripts/verify_reproducibility.py check --scope exporter
python src/export_website_data.py
```

The exporter should produce `docs/data/results.json` from recorded results without fitting ML or calling FeOs. It may include generation timestamps; compare its scientific contents rather than demanding byte-identical JSON across runs.

## 8. Troubleshooting without falsifying results

| Symptom | Meaning | Correct response |
| --- | --- | --- |
| `requirements.txt is missing` | A clean installation cannot use the declared dependencies | Commit the actual tested requirements from the working environment |
| `No module named 'feos'` / `si_units` | Incomplete package installation or incompatible environment | Confirm the active `.venv`, `python -m pip check`, and the requirement package versions; `si_units` is distributed as `si-units` |
| `No module named 'rdkit'` | RDKit is missing or installed in another interpreter | Confirm `.venv` and install from the actual project requirements |
| Missing `hydrocarbons_initial.csv` | Exact curated data unavailable | Commit the original CSV (if permitted) or a verified deterministic build script; do **not** fabricate it |
| Missing `molecule_split.csv` | The frozen train/test boundary cannot be checked | Commit the original split file, not a newly generated random split |
| `SHA-256 mismatch` | File contents differ from the original evaluated input | Recover the exact original, preserve the original config, investigate differences |
| Import succeeds but FeOs API fails | Installed FeOs release may be incompatible with the recorded code | Use the actual tested FeOs version from original `requirements.txt`/environment evidence; do not silently change scientific code |
| 156 states but different MAPE | Numerical/software or input mismatch | Verify hashes and package versions; report the discrepancy and compare per-molecule curves |
| Website displays `Awaiting data` | JSON or JavaScript did not load | Use the HTTP server, check `docs/data/results.json`, inspect browser console and network requests |
| `--run-final` refuses to overwrite | Expected safeguard in the original evaluator | Use the isolated `recompute` command, never delete frozen source files |

## 9. Reproducibility evidence to record

Only fill this out **after actually testing**. An unchecked or failed row must stay marked as such.

| Platform | Commit hash | Python / FeOs / RDKit versions | Clean install | Input hashes | Preflight | Isolated rerun | Website |
| --- | --- | --- | --- | --- | --- | --- | --- |
| macOS (Intel / Apple Silicon) | Not tested | Not tested | Not tested | Not tested | Not tested | Not tested | Not tested |
| Ubuntu x86-64 | Not tested | Not tested | Not tested | Not tested | Not tested | Not tested | Not tested |
| Arch Linux x86-64 | Not tested | Not tested | Not tested | Not tested | Not tested | Not tested | Not tested |

Keep the command output or terminal transcript for every completed run. **One verified clean clone is more defensible than claiming three operating systems without running them.**

## 10. Reproducibility integrity

- The final model is `SimpleImputer(median) + StandardScaler + Ridge(alpha=1.0)` applied to five molecular descriptors (molecular weight, carbon count, ring count, rotatable bonds, branch points).
- The molecule-wise split was fixed before final evaluation. The evaluator checks it against the seeded split, `random_state=42`.
- The frozen model was chosen on development VLE performance; the 39 final molecules were not used to choose it.
- The published parameter records are used to score ML predictions and establish the reference critical-temperature grid; they are not ML training features.
- The exact result tables should be retained and attributed. The test is numerical validation of published-parameter PC-SAFT agreement, **not experimental property accuracy**.
- Do not publish private keys, credentials, or personal files. Confirm dataset redistribution rights and cite FeOs and the Esper2023 parameter publication.

---

### Source references

- [FeOs framework and Python installation](https://github.com/feos-org/feos)
- [FeOs published pure-component PC-SAFT parameter collection](https://github.com/feos-org/feos/tree/main/parameters/pcsaft)
- [FeOs 0.9.6 wheel platforms](https://pypi.org/project/feos/0.9.6/)
- [si-units distribution / wheel platforms](https://pypi.org/project/si-units/)
- [RDKit pip installation](https://www.rdkit.org/docs/Install.html)
- [Astral uv Python version management](https://docs.astral.sh/uv/concepts/python-versions/)
