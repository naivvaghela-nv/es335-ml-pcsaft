from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from compare_development_vle_models_all39 import (
    load_data,
    build_model,
    FEATURE_SETS,
    TARGETS,
)

ROOT = Path(__file__).resolve().parents[1]

train, validation = load_data()

if len(train) != 116 or len(validation) != 39:
    raise RuntimeError("Unexpected development split sizes.")

if set(train.smiles) & set(validation.smiles):
    raise RuntimeError("Training/validation leakage detected.")

models = [
    "Ridge_5",
    "Ridge_12",
    "ExtraTrees_156",
]

rows = []

for model_name in models:
    features = FEATURE_SETS[model_name]

    for target in TARGETS:
        model = build_model(model_name)

        X_train = train[features]
        X_val = validation[features]

        y_train = train[target]
        y_val = validation[target]

        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        train_mae = mean_absolute_error(
            y_train, pred_train
        )
        val_mae = mean_absolute_error(
            y_val, pred_val
        )

        train_rmse = mean_squared_error(
            y_train, pred_train
        ) ** 0.5

        val_rmse = mean_squared_error(
            y_val, pred_val
        ) ** 0.5

        rows.append({
            "model": model_name,
            "target": target,
            "train_mae": train_mae,
            "validation_mae": val_mae,
            "train_rmse": train_rmse,
            "validation_rmse": val_rmse,
            "mae_gap": val_mae - train_mae,
        })

results = pd.DataFrame(rows)

output = (
    ROOT
    / "results/metrics/development_overfitting_check.csv"
)

output.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output, index=False)

print("\nTRAINING VS VALIDATION PARAMETER ERRORS")
print(results.to_string(index=False))

print("\nSaved:", output)
print("Reserved final test set remains untouched.")
