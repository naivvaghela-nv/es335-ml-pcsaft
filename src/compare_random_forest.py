from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
)

ROOT = Path(__file__).resolve().parents[1]

data = pd.read_csv(
    ROOT / "data/processed/hydrocarbons_initial.csv"
)
data = data.sort_values("smiles").reset_index(drop=True)

features = [
    "molecular_weight",
    "carbon_count",
    "ring_count",
    "rotatable_bonds",
    "branch_points",
]

targets = ["m", "sigma", "epsilon_k"]

train_idx, test_idx = train_test_split(
    np.arange(len(data)),
    test_size=0.20,
    random_state=42,
    shuffle=True,
)

# Check against the previously saved train/test split.
saved_split = pd.read_csv(
    ROOT / "results/predictions/molecule_split.csv"
)

expected_train = set(data.iloc[train_idx]["smiles"])
saved_train = set(
    saved_split.loc[
        saved_split["split"] == "train", "smiles"
    ]
)

if expected_train != saved_train:
    raise RuntimeError("Train/test split mismatch!")

X_train = data.iloc[train_idx][features]

cv = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

rows = []

for target in targets:
    y_train = data.iloc[train_idx][target]

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

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

    mae = -scores["test_MAE"]
    rmse = -scores["test_RMSE"]

    rows.append({
        "target": target,
        "model": "Random_Forest",
        "cv_mae_mean": mae.mean(),
        "cv_mae_std": mae.std(),
        "cv_rmse_mean": rmse.mean(),
        "cv_rmse_std": rmse.std(),
    })

existing = pd.read_csv(
    ROOT / "results/metrics/training_cv_metrics.csv"
)

combined = pd.concat(
    [existing, pd.DataFrame(rows)],
    ignore_index=True,
)

combined.to_csv(
    ROOT / "results/metrics/model_comparison_cv.csv",
    index=False,
)

print("\nMODEL COMPARISON")
print(
    combined[
        ["target", "model", "cv_mae_mean", "cv_rmse_mean"]
    ].sort_values(["target", "cv_mae_mean"]).to_string(index=False)
)

print("\nFinal test set remains untouched.")
