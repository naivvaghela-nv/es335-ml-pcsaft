from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "data/processed/hydrocarbons_enhanced.csv"
)

SPLIT = (
    ROOT
    / "results/predictions/molecule_split.csv"
)

OUTPUT = (
    ROOT
    / "results/metrics/enhanced_descriptor_cv.csv"
)

df = pd.read_csv(DATA)
df = df.sort_values("smiles").reset_index(drop=True)

saved_split = pd.read_csv(SPLIT)

ORIGINAL = [
    "molecular_weight",
    "carbon_count",
    "ring_count",
    "rotatable_bonds",
    "branch_points",
]

ENHANCED = ORIGINAL + [
    "BertzCT",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "Kappa1",
    "Kappa2",
    "BalabanJ",
]

TARGETS = ["m", "sigma", "epsilon_k"]

if df["smiles"].duplicated().any():
    raise ValueError("Duplicate molecules detected.")

if saved_split["smiles"].duplicated().any():
    raise ValueError("Duplicate split identifiers detected.")

train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.20,
    random_state=42,
    shuffle=True,
)

expected_train = set(df.iloc[train_idx]["smiles"])
expected_test = set(df.iloc[test_idx]["smiles"])

saved_train = set(
    saved_split.loc[
        saved_split["split"] == "train", "smiles"
    ]
)

saved_test = set(
    saved_split.loc[
        saved_split["split"] == "test", "smiles"
    ]
)

if (
    expected_train != saved_train
    or expected_test != saved_test
):
    raise ValueError(
        "Train/test split mismatch. Stop the experiment."
    )

print("Training molecules:", len(train_idx))
print("Reserved test molecules:", len(test_idx))
print("Final test set will not be evaluated.")

cv = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

rows = []

for feature_label, features in [
    ("Original_5", ORIGINAL),
    ("Enhanced_12", ENHANCED),
]:
    X_train = df.iloc[train_idx][features]

    for target in TARGETS:
        y_train = df.iloc[train_idx][target]

        models = {
            "Ridge": make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Ridge(alpha=1.0),
            ),
            "Random_Forest": RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
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

            mae = -scores["test_MAE"]
            rmse = -scores["test_RMSE"]

            rows.append({
                "features": feature_label,
                "model": model_name,
                "target": target,
                "cv_mae_mean": float(mae.mean()),
                "cv_mae_std": float(mae.std()),
                "cv_rmse_mean": float(rmse.mean()),
                "cv_rmse_std": float(rmse.std()),
            })

            print(
                f"{feature_label:12s} "
                f"{model_name:14s} "
                f"{target:10s} "
                f"MAE={mae.mean():.5f} "
                f"RMSE={rmse.mean():.5f}"
            )

results = pd.DataFrame(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print("\nBest configurations by target:")
for target in TARGETS:
    subset = results[
        results["target"] == target
    ].sort_values("cv_mae_mean")

    print(f"\n{target}:")
    print(
        subset[
            ["features", "model", "cv_mae_mean", "cv_rmse_mean"]
        ].to_string(index=False)
    )

print("\nSaved:", OUTPUT)
print("The final test set remains untouched.")
