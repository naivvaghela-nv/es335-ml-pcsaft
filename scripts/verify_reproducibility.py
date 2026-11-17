#!/usr/bin/env python3
"""Audit and reproduce the ES335 PC-SAFT results without changing frozen outputs.

Stdlib-only harness. Scientific packages are imported only for a science check or rerun.
Place this file at REPO_ROOT/scripts/verify_reproducibility.py.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
from importlib import metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW_URL = "https://raw.githubusercontent.com/feos-org/feos/main/parameters/pcsaft/esper2023.json"
INPUTS = {
    "curated table": "data/processed/hydrocarbons_initial.csv",
    "frozen molecule split": "results/predictions/molecule_split.csv",
    "published parameters": "data/raw/esper2023.json",
}
SCIENCE = {
    **INPUTS,
    "frozen evaluator": "src/final_holdout_evaluation.py",
    "evaluation configuration": "results/metrics/final_test_config.json",
    "final VLE curves": "results/predictions/final_test_vle_curves.csv",
    "final thermodynamic summary": "results/metrics/final_test_thermo_summary.csv",
    "final parameter metrics": "results/metrics/final_test_parameter_metrics.csv",
}
SITE = {
    "website index": "docs/index.html",
    "website data": "docs/data/results.json",
}
EXPORTER = {
    "website exporter": "src/export_website_data.py",
    "VLE curves": "results/predictions/final_test_vle_curves.csv",
    "per-molecule errors": "results/metrics/final_test_per_molecule.csv",
    "parameter metrics": "results/metrics/final_test_parameter_metrics.csv",
    "summary": "results/metrics/final_test_thermo_summary.csv",
    "final configuration": "results/metrics/final_test_config.json",
    "molecule split": "results/predictions/molecule_split.csv",
    "enhanced CV metrics": "results/metrics/enhanced_descriptor_cv.csv",
    "richer CV metrics": "results/metrics/richer_model_comparison_cv.csv",
    "overfitting metrics": "results/metrics/development_overfitting_check.csv",
    "rich descriptor table": "data/processed/hydrocarbons_richer.csv",
}
HASH_KEYS = {
    "source_dataset_sha256": INPUTS["curated table"],
    "frozen_split_sha256": INPUTS["frozen molecule split"],
    "raw_pcsaft_sha256": INPUTS["published parameters"],
}
MODULES = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
           "rdkit": "rdkit", "feos": "feos", "si_units": "si-units", "joblib": "joblib"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def rows_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            fail(f"Empty CSV or missing header: {path}")
        rows = list(reader)
    if not rows:
        fail(f"CSV has no records: {path}")
    return rows


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need_files(mapping):
    missing = [(label, path) for label, path in mapping.items() if not (ROOT / path).is_file()]
    if missing:
        message = "\n".join(f"  - {label}: {p}" for label, p in missing)
        fail("Required files are missing from this checkout:\n" + message +
             "\nSee REPRODUCIBILITY.md, 'Missing data / recovery'.")
    print(f"PASS: {len(mapping)} required files exist")


def check_imports():
    for module, distribution in MODULES.items():
        try:
            importlib.import_module(module)
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                version = "installed (version unavailable)"
        except Exception as exc:
            fail(f"Cannot import {module}: {exc}. Check requirements.txt and Python environment.")
        print(f"  {module}: {version}")
    print("PASS: scientific Python modules import")


def verify_split():
    curated = rows_csv(ROOT / INPUTS["curated table"])
    split = rows_csv(ROOT / INPUTS["frozen molecule split"])
    features = {"smiles", "m", "sigma", "epsilon_k", "molecular_weight", "carbon_count",
                "ring_count", "rotatable_bonds", "branch_points"}
    missing = features - set(curated[0])
    if missing:
        fail(f"Curated data missing required columns: {sorted(missing)}")
    for table, name in ((curated, "curated data"), (split, "split")):
        ids = [r.get("smiles", "").strip() for r in table]
        if any(not v for v in ids) or len(ids) != len(set(ids)):
            fail(f"Empty or duplicate SMILES in {name}")
    if len(curated) != 194 or len(split) != 194:
        fail(f"Unexpected molecule counts: curated={len(curated)}, split={len(split)}; expected 194")
    if {r["smiles"] for r in curated} != {r["smiles"] for r in split}:
        fail("Curated dataset and recorded split have different SMILES sets")
    label_counts = {label: sum(r.get("split") == label for r in split) for label in ("train", "test")}
    if label_counts != {"train": 155, "test": 39}:
        fail(f"Wrong frozen split: {label_counts}; expected 155 train / 39 test")
    print("PASS: exactly 194 curated molecules and frozen 155/39 split")


def verify_hashes():
    config_path = ROOT / SCIENCE["evaluation configuration"]
    config = read_json(config_path)
    for key, relative in HASH_KEYS.items():
        expected = config.get(key)
        if not isinstance(expected, str) or len(expected) != 64:
            fail(f"Frozen config lacks usable {key}; cannot establish provenance")
        actual = digest(ROOT / relative)
        if actual.lower() != expected.lower():
            fail(f"SHA-256 mismatch in {relative}\nexpected: {expected}\nactual:   {actual}\n"
                 "Do NOT silently overwrite the dataset or frozen results.")
        print(f"PASS: {relative} matches frozen SHA-256")
    if config.get("chosen_model") != "Ridge_5":
        fail("Frozen configuration does not specify Ridge_5")
    return config


def verify_saved_results():
    summary = rows_csv(ROOT / SCIENCE["final thermodynamic summary"])
    curves = rows_csv(ROOT / SCIENCE["final VLE curves"])
    if len(summary) != 1 or len(curves) != 156:
        fail("Expected one summary row and 156 final VLE state rows")
    unique_smiles = {r.get("smiles", "") for r in curves}
    if len(unique_smiles) != 39:
        fail(f"Expected 39 unique held-out molecules; found {len(unique_smiles)}")
    r = summary[0]
    for col, number in {"n_test_molecules": 39, "n_attempted": 156,
                        "n_comparable_pressure": 156, "n_comparable_density": 156}.items():
        if int(float(r.get(col, "nan"))) != number:
            fail(f"Saved summary {col} != {number}")
    for key, target in (("mean_pressure_ape_percent_on_comparable", 20.54),
                        ("mean_density_ape_percent_on_comparable", 2.66)):
        result = float(r.get(key, "nan"))
        if not math.isfinite(result) or abs(result - target) > 0.15:
            fail(f"Saved {key} is {result}; expected approximately {target}%")
    print("PASS: saved results show 39 molecules, 156/156 comparisons, ~20.54% / ~2.66% MAPE")


def check(scope: str, imports=True):
    if scope == "site":
        need_files(SITE)
        value = read_json(ROOT / SITE["website data"])
        if not isinstance(value, dict):
            fail("docs/data/results.json should be a JSON object")
        print("PASS: static website data is valid JSON")
        return
    if scope == "exporter":
        need_files(EXPORTER)
        print("PASS: required website-export inputs exist")
        return
    if scope in ("science", "all"):
        need_files(SCIENCE)
        verify_split()
        verify_hashes()
        verify_saved_results()
        if imports:
            check_imports()
    if scope == "all":
        check("site", imports=False)
    print(f"PASS: {scope} verification completed")


def fetch_raw():
    config_path = ROOT / SCIENCE["evaluation configuration"]
    if not config_path.exists():
        fail(f"Missing saved evaluation config: {config_path}; cannot verify dataset provenance")
    expected = read_json(config_path).get("raw_pcsaft_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        fail("Missing raw_pcsaft_sha256 in saved configuration. Cannot safely download raw file.")
    target = ROOT / INPUTS["published parameters"]
    if target.exists():
        actual = digest(target)
        if actual.lower() != expected.lower():
            fail(f"An existing {target} does not match frozen SHA-256. Do not overwrite it.")
        print("PASS: raw parameters already exist with exact matching SHA-256")
        return
    print(f"Downloading original published parameter file from {RAW_URL}")
    request = Request(RAW_URL, headers={"User-Agent": "ES335-reproducibility-check/1.0"})
    with urlopen(request, timeout=30) as response:
        data = response.read(50_000_001)
    if len(data) > 50_000_000:
        fail("Download unexpectedly exceeds 50 MB")
    if hashlib.sha256(data).hexdigest().lower() != expected.lower():
        fail("UPSTREAM DATASET HASH MISMATCH. Upstream file may have changed; obtain the exact "
             "source used in the frozen run. Nothing has been written.")
    records = json.loads(data.decode("utf-8-sig"))
    if not isinstance(records, list):
        fail("Downloaded Esper2023 data is not the expected JSON list")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"PASS: downloaded {len(records)} raw records; hash verified; saved {target}")


def reproduce(tolerance: float, keep_dir: str | None):
    check("science", imports=True)
    frozen_summary = rows_csv(ROOT / SCIENCE["final thermodynamic summary"])[0]
    frozen_curves = rows_csv(ROOT / SCIENCE["final VLE curves"])

    @contextlib.contextmanager
    def workspace():
        if keep_dir:
            temp_root = Path(keep_dir).expanduser().resolve()
            if temp_root == ROOT or ROOT in temp_root.parents and temp_root.name in ("src", "data", "results"):
                fail("Choose a separate empty output directory, not an existing project source/results directory")
            if temp_root.exists() and any(temp_root.iterdir()):
                fail(f"Refusing to overwrite nonempty reproduction directory: {temp_root}")
            temp_root.mkdir(parents=True, exist_ok=True)
            yield temp_root
        else:
            with tempfile.TemporaryDirectory(prefix="es335-rerun-") as temp:
                yield Path(temp)

    with workspace() as scratch:
        required = ["src/final_holdout_evaluation.py", *INPUTS.values()]
        for relative in required:
            dst = scratch / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, dst)
        evaluator = scratch / "src/final_holdout_evaluation.py"
        print("\nRerunning the same frozen evaluator in an isolated workspace:", scratch, flush=True)
        for flag in ("--preflight", "--run-final"):
            subprocess.run([sys.executable, str(evaluator), flag], cwd=scratch, check=True)
        rerun_summary = rows_csv(scratch / SCIENCE["final thermodynamic summary"])[0]
        rerun_curves = rows_csv(scratch / SCIENCE["final VLE curves"])
        for col in ("n_test_molecules", "n_attempted", "n_comparable_pressure", "n_comparable_density"):
            lhs, rhs = int(float(frozen_summary[col])), int(float(rerun_summary[col]))
            if lhs != rhs:
                fail(f"Reproduction mismatch: {col}: saved={lhs}, rerun={rhs}")
        for key in ("mean_pressure_ape_percent_on_comparable",
                    "mean_density_ape_percent_on_comparable"):
            expected = float(frozen_summary[key])
            actual = float(rerun_summary[key])
            delta = abs(actual - expected)
            print(f"{key}: recorded={expected:.6f}%, rerun={actual:.6f}%, difference={delta:.6f} percentage points")
            if not math.isfinite(delta) or delta > tolerance:
                fail(f"Reproduction mismatch: {key} differs by more than {tolerance} percentage points")
        if len(rerun_curves) != len(frozen_curves):
            fail("Reproduction mismatch: different number of VLE state rows")
        keycols = ("smiles", "temperature_fraction_of_ref_Tc")
        signature = lambda row: tuple(row[k] for k in keycols)
        if {signature(r) for r in rerun_curves} != {signature(r) for r in frozen_curves}:
            fail("Reproduced state identifiers or temperature fractions differ")
        print("PASS: independent isolated rerun matches the original 39-molecule evaluation")
        if keep_dir:
            print("Reproduction outputs retained at:", scratch)
        else:
            print("Temporary computed outputs were discarded; the original frozen outputs are unchanged")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    check_cmd = sub.add_parser("check", help="verify files, provenance, saved outputs and imports")
    check_cmd.add_argument("--scope", choices=("site", "science", "exporter", "all"), default="all")
    check_cmd.add_argument("--no-imports", action="store_true", help="skip optional installed package check")
    sub.add_parser("fetch-raw", help="download Esper2023 only if SHA-256 matches frozen config")
    rerun = sub.add_parser("recompute", help="rerun final frozen model in isolated temporary workspace")
    rerun.add_argument("--tolerance", type=float, default=0.5,
                       help="maximum difference in final pressure/density MAPE, percentage points (default 0.5)")
    rerun.add_argument("--keep-dir", type=str, help="retain new outputs in a separate empty directory")
    args = p.parse_args()
    try:
        if args.command == "check":
            check(args.scope, imports=not args.no_imports)
        elif args.command == "fetch-raw":
            fetch_raw()
        else:
            if not math.isfinite(args.tolerance) or args.tolerance < 0:
                fail("--tolerance must be finite and nonnegative")
            reproduce(args.tolerance, args.keep_dir)
    except (RuntimeError, FileNotFoundError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print("\nFAIL:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
