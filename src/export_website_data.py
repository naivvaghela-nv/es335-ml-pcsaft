"""Export *existing* frozen ES335 held-out CSV results to docs/data/results.json.

Run from repository root:
    .\\.venv\\Scripts\\python.exe src/export_website_data.py

This script does not fit models, call FeOs, modify the held-out CSVs, or create
synthetic measurements. It validates the held-out files before publishing.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions"
METRICS = ROOT / "results" / "metrics"
OUTPUT = ROOT / "docs" / "data" / "results.json"

FILES = {
    "curves": PRED / "final_test_vle_curves.csv",
    "per_molecule": METRICS / "final_test_per_molecule.csv",
    "parameter_metrics": METRICS / "final_test_parameter_metrics.csv",
    "thermo_summary": METRICS / "final_test_thermo_summary.csv",
    "config": METRICS / "final_test_config.json",
    "split": PRED / "molecule_split.csv",
    "enhanced_cv": METRICS / "enhanced_descriptor_cv.csv",
    "richer_cv": METRICS / "richer_model_comparison_cv.csv",
    "overfitting": METRICS / "development_overfitting_check.csv",
    "richer_features": ROOT / "data" / "processed" / "hydrocarbons_richer.csv",
}
OPTIONAL = {
    "development_vle": METRICS / "development_model_vle_comparison_all39.csv",
    "development_density": METRICS / "development_density_summary_all39.csv",
    "sensitivities": METRICS / "development_pressure_sensitivities.csv",
}
NUMERIC_COLUMNS = {
    "temperature_fraction_of_ref_Tc", "temperature_K", "reference_Tc_K",
    "pressure_reference_bar", "pressure_ml_bar", "pressure_ape_percent",
    "density_reference_kg_m3", "density_ml_kg_m3", "density_ape_percent",
    "reference_m", "reference_sigma", "reference_epsilon_k",
    "predicted_m", "predicted_sigma", "predicted_epsilon_k",
    "mean_pressure_ape_percent", "mean_density_ape_percent",
    "mae", "rmse", "cv_mae_mean", "cv_mae_std", "cv_rmse_mean", "cv_rmse_std",
    "train_mae", "validation_mae", "train_rmse", "validation_rmse", "mae_gap", "n_features",
    "mean_pressure_ape_percent_on_comparable",
    "mean_density_ape_percent_on_comparable", "median_molecule_pressure_ape_percent",
    "median_molecule_density_ape_percent", "macro_mean_molecule_ape_percent",
    "mean_ape_percent_on_comparable", "mean_density_ape_percent",
    "S_dlnP_dlnparameter", "n_attempted", "n_comparable_pressure",
    "n_comparable_density", "n_test_molecules", "n_reference_success",
    "n_predicted_success", "n_comparable", "n_molecules",
    "failed_pressure_comparisons", "failed_density_comparisons", "n_points",
    "n_test_molecules", "n_reference_success", "n_ml_success",
}
REQUIRED_CURVE_COLUMNS = {
    "name", "smiles", "temperature_K", "temperature_fraction_of_ref_Tc",
    "pressure_reference_bar", "pressure_ml_bar", "pressure_ape_percent",
    "density_reference_kg_m3", "density_ml_kg_m3", "density_ape_percent",
    "reference_status", "predicted_status", "reference_m", "reference_sigma",
    "reference_epsilon_k", "predicted_m", "predicted_sigma",
    "predicted_epsilon_k",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is empty: {path}")
        rows = []
        for row in reader:
            cleaned = {}
            for key, raw in row.items():
                if key is None:
                    raise ValueError(f"Malformed CSV row in {path}")
                raw = (raw or "").strip()
                if key in NUMERIC_COLUMNS:
                    if not raw:
                        cleaned[key] = None
                    else:
                        n = float(raw)
                        cleaned[key] = n if math.isfinite(n) else None
                else:
                    cleaned[key] = raw
            rows.append(cleaned)
    return rows


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, label: str, tolerance=0.0001):
    require(abs(actual - expected) < tolerance, f"{label} mismatch: {actual:.8f} vs {expected:.8f}")


def mean(nums):
    return sum(nums) / len(nums) if nums else None


# Descriptor names are the same as in build_enhanced_features.py and
# build_richer_features.py. Feature collisions are computed for the original
# 155-molecule training pool; no fitted property labels are used.
BASE_5 = ["molecular_weight", "carbon_count", "ring_count", "rotatable_bonds", "branch_points"]
BASE_12 = BASE_5 + ["BertzCT", "Chi0v", "Chi1v", "Chi2v", "Kappa1", "Kappa2", "BalabanJ"]
ADDED_16 = [
    "Chi3v", "Chi4v", "Kappa3", "MolLogP", "MolMR", "LabuteASA",
    "NumBridgeheadAtoms", "NumSpiroAtoms", "NumAliphaticRings",
    "NumAtomStereoCenters", "primary_carbons", "secondary_carbons",
    "tertiary_carbons", "quaternary_carbons", "five_member_rings", "six_member_rings",
]
FEATURE_SETS = {
    "Original_5": BASE_5,
    "Enhanced_12": BASE_12,
    "Richer_28": BASE_12 + ADDED_16,
    "Richer_28_Morgan128": BASE_12 + ADDED_16 + [f"morgan_{i:03d}" for i in range(128)],
}
CV_FIELDS = ("cv_mae_mean", "cv_mae_std", "cv_rmse_mean", "cv_rmse_std")


def model_experiments(split_by_smiles: dict[str, str]) -> dict:
    enriched = read_csv(FILES["enhanced_cv"])
    richer = read_csv(FILES["richer_cv"])
    overfit = read_csv(FILES["overfitting"])
    feature_table = read_csv(FILES["richer_features"])
    require(len(feature_table) == 194, "Richer dataset must contain 194 distinct molecules")
    feature_ids = [r["smiles"] for r in feature_table]
    require(len(set(feature_ids)) == 194 and set(feature_ids) == set(split_by_smiles),
            "Feature-table molecule identities do not match the frozen split")
    train_rows = [r for r in feature_table if split_by_smiles[r["smiles"]] == "train"]
    require(len(train_rows) == 155, "Collision computation must use the 155 training molecules only")
    collisions = []
    for label, cols in FEATURE_SETS.items():
        missing = set(cols) - set(feature_table[0])
        require(not missing, f"Feature table missing {label}: {sorted(missing)}")
        descriptors = [tuple(row[c] for c in cols) for row in train_rows]
        counts = Counter(descriptors)
        colliding = sum(counts[d] > 1 for d in descriptors)
        collisions.append({
            "features": label, "n_features": len(cols),
            "train_molecules": len(train_rows),
            "colliding_molecules": colliding,
            "unique_vectors": len(counts),
        })

    seen = {}
    allowed_models = {"Ridge", "Random_Forest", "Extra_Trees"}
    allowed_targets = {"m", "sigma", "epsilon_k"}
    for source, data in (("enhanced_descriptor_cv.csv", enriched),
                         ("richer_model_comparison_cv.csv", richer)):
        for r in data:
            label, model, target = r["features"], r["model"], r["target"]
            require(label in FEATURE_SETS and model in allowed_models and target in allowed_targets,
                    f"Unexpected cross-validation row from {source}: {label}/{model}/{target}")
            for field in CV_FIELDS:
                require(Number_valid(r.get(field)), f"Missing/invalid {field}: {source} {label}/{model}/{target}")
            require(r["cv_mae_mean"] >= 0 and r["cv_rmse_mean"] >= 0 and
                    r["cv_mae_std"] >= 0 and r["cv_rmse_std"] >= 0,
                    "Cross-validation error and standard deviations must not be negative")
            if r.get("n_features") is not None:
                require(int(r["n_features"]) == len(FEATURE_SETS[label]),
                        f"Feature count mismatch for {label} in {source}")
            record = {"features": label, "n_features": len(FEATURE_SETS[label]),
                      "model": model, "target": target}
            record.update({field: r[field] for field in CV_FIELDS})
            key = (label, model, target)
            if key in seen:
                for field in CV_FIELDS:
                    require(abs(seen[key][field] - record[field]) < 0.00001,
                            f"Inconsistent repeated CV result {key} ({field}); stop rather than publish")
            else:
                seen[key] = record
    require(len(seen) >= 30, f"Too few cross-validation records: {len(seen)}")
    require(len({k[0] for k in seen}) == 4, "CV data must cover all four feature representations")
    require(len({k[2] for k in seen}) == 3, "CV data must cover all three targets")
    require(len(overfit) == 9, "Expected 9 development overfitting comparison rows")
    require({(r["model"], r["target"]) for r in overfit} ==
            {(m, t) for m in ("Ridge_5", "Ridge_12", "ExtraTrees_156") for t in allowed_targets},
            "Development overfitting comparison missing model-target pairs")
    for r in overfit:
        for key in ("train_mae", "validation_mae", "train_rmse", "validation_rmse"):
            require(Number_valid(r.get(key)) and r[key] >= 0,
                    f"Missing/nonfinite {key} for {r['model']} / {r['target']}")
    return {
        "cv_models": sorted(seen.values(), key=lambda r:(r["n_features"], r["target"], r["model"])),
        "collisions": collisions,
        "overfitting": overfit,
        "sources": {
            "enhanced_descriptor_cv": sha256(FILES["enhanced_cv"]),
            "richer_model_comparison_cv": sha256(FILES["richer_cv"]),
            "development_overfitting_check": sha256(FILES["overfitting"]),
            "richer_feature_table": sha256(FILES["richer_features"]),
        },
        "note": "All CV values use the original training pool, with three fixed folds. Collisions use the same 155 training SMILES. No test targets were used to select models.",
    }


def Number_valid(n):
    return isinstance(n, (float, int)) and math.isfinite(n)


def main():
    for label, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required {label}: {path}")
    curves = read_csv(FILES["curves"])
    molecules = read_csv(FILES["per_molecule"])
    param_metrics = read_csv(FILES["parameter_metrics"])
    summary_rows = read_csv(FILES["thermo_summary"])
    with FILES["config"].open(encoding="utf-8-sig") as file:
        config = json.load(file)
    split = read_csv(FILES["split"])

    require(len(summary_rows) == 1, "Expected exactly one frozen held-out thermodynamic summary")
    summary = summary_rows[0]
    require(len(molecules) == 39 and len(curves) == 156, "Expected 39 molecules and 156 final-test curve rows")
    require(len({row["smiles"] for row in molecules}) == 39, "Duplicate molecule in per-molecule summary")
    require(REQUIRED_CURVE_COLUMNS <= set(curves[0]), "Curve CSV missing required columns")
    require(config.get("chosen_model") == "Ridge_5", "Unexpected final model; review evaluation protocol")
    require(config.get("training_molecules") == 155 and config.get("held_out_test_molecules") == 39,
            "Unexpected frozen train/test sizes")

    splits_by_smiles = {row["smiles"]: row["split"] for row in split}
    require(len(splits_by_smiles) == 194 and len(split) == 194, "Expected unique frozen split for 194 molecules")
    test_ids = {smiles for smiles, value in splits_by_smiles.items() if value == "test"}
    train_ids = {smiles for smiles, value in splits_by_smiles.items() if value == "train"}
    require(len(test_ids) == 39 and len(train_ids) == 155 and not (test_ids & train_ids),
            "Frozen split invalid")
    curve_ids = {row["smiles"] for row in curves}
    summary_ids = {row["smiles"] for row in molecules}
    require(curve_ids == test_ids == summary_ids, "Website results must contain only the frozen final-test molecules")
    require(len({(row["smiles"], row["temperature_fraction_of_ref_Tc"]) for row in curves}) == 156,
            "Repeated temperature points detected")

    grouped = {}
    for row in curves:
        grouped.setdefault(row["smiles"], []).append(row)
        require(row["reference_status"] == "ok" and row["predicted_status"] == "ok",
                f"Unexpected solver failure for {row['name']}; site exporter expects success; review manually")
        for key in ("pressure_reference_bar", "pressure_ml_bar",
                    "density_reference_kg_m3", "density_ml_kg_m3"):
            require(row[key] is not None and row[key] > 0,
                    f"Invalid property {key} for {row['name']}")
        p_err = 100 * abs(row["pressure_ml_bar"] - row["pressure_reference_bar"]) / row["pressure_reference_bar"]
        rho_err = 100 * abs(row["density_ml_kg_m3"] - row["density_reference_kg_m3"]) / row["density_reference_kg_m3"]
        close(row["pressure_ape_percent"], p_err, f"Pressure APE for {row['name']}", tolerance=0.00001)
        close(row["density_ape_percent"], rho_err, f"Density APE for {row['name']}", tolerance=0.00001)

    for mol in molecules:
        points = grouped[mol["smiles"]]
        require(len(points) == 4, f"Expected four temperature states for {mol['name']}")
        close(mol["mean_pressure_ape_percent"], mean([p["pressure_ape_percent"] for p in points]),
              f"Molecule pressure APE: {mol['name']}")
        close(mol["mean_density_ape_percent"], mean([p["density_ape_percent"] for p in points]),
              f"Molecule density APE: {mol['name']}")
    close(summary["mean_pressure_ape_percent_on_comparable"],
          mean([p["pressure_ape_percent"] for p in curves]), "Final mean pressure APE")
    close(summary["mean_density_ape_percent_on_comparable"],
          mean([p["density_ape_percent"] for p in curves]), "Final mean density APE")
    require(summary["n_comparable_pressure"] == 156 and summary["n_comparable_density"] == 156,
            "Final summary comparable counts do not equal observed successes")

    experiments = model_experiments(splits_by_smiles)
    # Descriptive dataset overview. Counts are computed from structure-derived
    # columns only; target labels never enter the profile.
    source_rows = read_csv(FILES["richer_features"])
    carbon_dist = Counter()
    ring_dist = Counter()
    for row in source_rows:
        try:
            carbon = int(float(row["carbon_count"]))
            rings = int(float(row["ring_count"]))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("Invalid carbon_count/ring_count in richer feature table") from exc
        require(3 <= carbon <= 20 and rings >= 0,
                f"Unexpected carbon/ring count for {row.get('smiles')}")
        carbon_dist[carbon] += 1
        ring_dist[rings] += 1
    require(sum(carbon_dist.values()) == 194 and sum(ring_dist.values()) == 194,
            "Descriptive dataset distribution does not cover 194 molecules")
    dataset_profile = {
        "total": 194,
        "carbon_count": [{"value": k, "count": v} for k, v in sorted(carbon_dist.items())],
        "ring_count": [{"value": k, "count": v} for k, v in sorted(ring_dist.items())],
        "note": "Entire 194-molecule curated collection; input descriptors only, no test target labels used in chart",
    }
    optional = {}
    for key, path in OPTIONAL.items():
        optional[key] = read_csv(path) if path.is_file() else []
        if not path.is_file():
            print(f"Optional development file absent: {path.name} (section will use explanatory text)")

    export = {
        "metadata": {
            "title": "ML-Assisted PC-SAFT for Pure-Component VLE of Hydrocarbons",
            "course": "ES 335 Machine Learning",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_csv_sha256": sha256(FILES["curves"]),
            "test_molecules": 39,
            "training_molecules": 155,
            "dataset_molecules": 194,
            "model": "Ridge(alpha=1.0), 5 RDKit descriptors",
            "interpretation": "Errors are against published-parameter PC-SAFT, not experimental measurements.",
            "temperature_basis": "T/Tc from the published reference parameterization",
        },
        "thermo_summary": summary,
        "parameter_metrics": param_metrics,
        "per_molecule": molecules,
        "curves": curves,
        "development": optional,
        "experiments": experiments,
        "dataset_profile": dataset_profile,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(export, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print("WEBSITE DATA EXPORT: PASS")
    print(f"  Input final-test curves: {len(curves)} rows")
    print(f"  Frozen held-out molecules: {len(molecules)}")
    print(f"  Training-CV model/feature/target results: {len(experiments['cv_models'])}")
    print("  Feature collisions (training molecules): " +
          ", ".join(f"{x['n_features']} features: {x['colliding_molecules']}" for x in experiments["collisions"]))
    print(f"  Pressure error: {summary['mean_pressure_ape_percent_on_comparable']:.4f}%")
    print(f"  Density error: {summary['mean_density_ape_percent_on_comparable']:.4f}%")
    print(f"  Wrote: {OUTPUT}")
    print("  No ML retraining or FeOs calculations performed.")
    print("  WARNING: docs/data/results.json contains molecular labels and parameter values; publishing the website makes these public.")


if __name__ == "__main__":
    main()
