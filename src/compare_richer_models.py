r"""Training-only cross-validation for richer PC-SAFT parameter predictors.

Run from repository root:
    .\.venv\Scripts\python.exe src/compare_richer_models.py

This file NEVER trains on or evaluates the 39-molecule held-out test set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/hydrocarbons_richer.csv"
SPLIT = ROOT / "results/predictions/molecule_split.csv"
OUT = ROOT / "results/metrics/richer_model_comparison_cv.csv"
CONFIG_OUT = ROOT / "results/metrics/richer_model_comparison_config.json"

BASE_12 = [
    "molecular_weight", "carbon_count", "ring_count",
    "rotatable_bonds", "branch_points", "BertzCT", "Chi0v",
    "Chi1v", "Chi2v", "Kappa1", "Kappa2", "BalabanJ",
]
ADDED_16 = [
    "Chi3v", "Chi4v", "Kappa3", "MolLogP", "MolMR", "LabuteASA",
    "NumBridgeheadAtoms", "NumSpiroAtoms", "NumAliphaticRings",
    "NumAtomStereoCenters", "primary_carbons", "secondary_carbons",
    "tertiary_carbons", "quaternary_carbons",
    "five_member_rings", "six_member_rings",
]
FINGERPRINT = [f"morgan_{i:03d}" for i in range(128)]
FEATURE_SETS = {
    "Enhanced_12": BASE_12,
    "Richer_28": BASE_12 + ADDED_16,
    "Richer_28_Morgan128": BASE_12 + ADDED_16 + FINGERPRINT,
}
TARGETS = ["m", "sigma", "epsilon_k"]


def make_models() -> dict:
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random_Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Extra_Trees": ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }


def main() -> None:
    df = pd.read_csv(DATA).sort_values("smiles").reset_index(drop=True)
    split = pd.read_csv(SPLIT, dtype={"smiles": str})
    if df["smiles"].isna().any() or df["smiles"].duplicated().any():
        raise ValueError("Duplicate/missing molecules in feature table")
    if split["smiles"].duplicated().any() or set(df.smiles) != set(split.smiles):
        raise ValueError("Frozen split mismatch")
    expected_labels = set(split["split"].unique())
    if expected_labels != {"train", "test"}:
        raise ValueError(f"Unrecognized frozen split labels: {expected_labels}")

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, shuffle=True
    )
    if set(df.iloc[train_idx].smiles) != set(split.loc[split.split == "train", "smiles"]):
        raise ValueError("Training split changed: stop")
    if set(df.iloc[test_idx].smiles) != set(split.loc[split.split == "test", "smiles"]):
        raise ValueError("Held-out split changed: stop")
    print(f"Train pool: {len(train_idx)}; untouched final test: {len(test_idx)}")

    required = set(TARGETS)
    for features in FEATURE_SETS.values():
        required.update(features)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not np.isfinite(df.iloc[train_idx][list(required)].astype(float).to_numpy()).all():
        raise ValueError("Invalid numeric values in training pool")

    folds = KFold(n_splits=3, shuffle=True, random_state=42)
    rows = []
    for label, features in FEATURE_SETS.items():
        X = df.iloc[train_idx][features]
        for target in TARGETS:
            y = df.iloc[train_idx][target]
            for name, model in make_models().items():
                scores = cross_validate(
                    model, X, y, cv=folds,
                    scoring={"MAE": "neg_mean_absolute_error",
                             "RMSE": "neg_root_mean_squared_error"},
                    n_jobs=1,
                    error_score="raise",
                )
                maes = -scores["test_MAE"]
                rmses = -scores["test_RMSE"]
                rows.append({
                    "features": label,
                    "model": name,
                    "target": target,
                    "n_features": len(features),
                    "cv_mae_mean": float(maes.mean()),
                    "cv_mae_std": float(maes.std()),
                    "cv_rmse_mean": float(rmses.mean()),
                    "cv_rmse_std": float(rmses.std()),
                })
                print(f"{label:23s} {name:14s} {target:10s} "
                      f"MAE {maes.mean():.5f} +/- {maes.std():.5f}")

    output = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)
    config = {
        "training_molecules": len(train_idx),
        "reserved_test_molecules": len(test_idx),
        "random_seed": 42,
        "n_cv_folds": 3,
        "feature_sets": FEATURE_SETS,
        "model_notes": "Ridge(alpha=1), RF/ExtraTrees(n_estimators=200, min_samples_leaf=2, random_state=42)",
        "note": "Do not select final model using held-out test metrics.",
    }
    CONFIG_OUT.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nBest configuration by parameter (training CV only):")
    for target in TARGETS:
        print("\n", target)
        sub = output[output.target == target].sort_values("cv_mae_mean")
        print(sub[["features", "model", "cv_mae_mean", "cv_mae_std"]].head(6).to_string(index=False))
    print("\nSaved:", OUT)
    print("Saved:", CONFIG_OUT)
    print("The final test set was not trained on or evaluated.")


if __name__ == "__main__":
    main()
