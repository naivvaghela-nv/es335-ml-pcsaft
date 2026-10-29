from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/hydrocarbons_initial.csv"
METRICS_DIR = ROOT / "results/metrics"
PREDICTIONS_DIR = ROOT / "results/predictions"
MODELS_DIR = ROOT / "results/models"

for directory in [METRICS_DIR, PREDICTIONS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "molecular_weight",
    "carbon_count",
    "ring_count",
    "rotatable_bonds",
    "branch_points",
]

TARGETS = ["m", "sigma", "epsilon_k"]

# Load molecular dataset
df = pd.read_csv(DATA_FILE)
df = df.sort_values("smiles").reset_index(drop=True)

required = ["smiles"] + FEATURES + TARGETS
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["smiles"].isna().any():
    raise ValueError("Missing molecular identifiers found.")

if df["smiles"].duplicated().any():
    raise ValueError("Duplicate SMILES found. Review dataset.")

if len(df) < 20:
    raise ValueError(
        f"Only {len(df)} molecules. Review dataset size "
        "before using this evaluation plan."
    )

for column in FEATURES + TARGETS:
    df[column] = pd.to_numeric(df[column], errors="raise")

if not np.isfinite(df[FEATURES + TARGETS].to_numpy()).all():
    raise ValueError("Missing or non-finite numeric values found.")

if (df[TARGETS] <= 0).any().any():
    raise ValueError("PC-SAFT target parameters must be positive.")

print("Total molecules:", len(df))
print("Input features:", FEATURES)
print("Predicted targets:", TARGETS)

# Fixed split for reproducibility
train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=42,
    shuffle=True,
)

split_labels = np.full(len(df), "test", dtype=object)
split_labels[train_idx] = "train"

split_df = pd.DataFrame({
    "smiles": df["smiles"],
    "split": split_labels,
})

split_df.to_csv(
    PREDICTIONS_DIR / "molecule_split.csv",
    index=False,
)

print("Training molecules:", len(train_idx))
print("Reserved test molecules:", len(test_idx))

X_train = df.iloc[train_idx][FEATURES]

cv = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

results = []

for target in TARGETS:
    y_train = df.iloc[train_idx][target]

    models = {
        "Mean_Baseline": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
    }

    for model_name, model in models.items():

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "MAE": "neg_mean_absolute_error",
                "RMSE": "neg_root_mean_squared_error",
            },
            n_jobs=1,
        )

        mae_values = -scores["test_MAE"]
        rmse_values = -scores["test_RMSE"]

        results.append({
            "target": target,
            "model": model_name,
            "cv_mae_mean": mae_values.mean(),
            "cv_mae_std": mae_values.std(),
            "cv_rmse_mean": rmse_values.mean(),
            "cv_rmse_std": rmse_values.std(),
        })

        # Train the model on all training molecules.
        # Final test data remains unused.
        model.fit(X_train, y_train)

        joblib.dump(
            model,
            MODELS_DIR / f"{model_name}_{target}.joblib",
        )

        print(
            f"{target:10s} | {model_name:15s} "
            f"| CV MAE = {mae_values.mean():.5f} "
            f"| CV RMSE = {rmse_values.mean():.5f}"
        )

metrics = pd.DataFrame(results)

metrics.to_csv(
    METRICS_DIR / "training_cv_metrics.csv",
    index=False,
)

metadata = {
    "random_state": 42,
    "test_fraction": 0.20,
    "cv_folds": 3,
    "features": FEATURES,
    "targets": TARGETS,
    "molecules_total": len(df),
    "molecules_train": len(train_idx),
    "molecules_test_reserved": len(test_idx),
    "ridge_alpha": 1.0,
}

with open(METRICS_DIR / "run_config.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nCross-validation results:")
print(metrics.to_string(index=False))

print("\nSaved training metrics and split.")
print("Final test set has NOT been evaluated.")
