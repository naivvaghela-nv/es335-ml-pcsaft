r"""One-time final evaluation: frozen Ridge_5, 155 train / 39 held-out molecules.

From the repository root, on Windows PowerShell:
  .\.venv\Scripts\python.exe src/final_holdout_evaluation.py --preflight
  .\.venv\Scripts\python.exe src/final_holdout_evaluation.py --run-final

Do not use final held-out results for model or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import feos
import joblib
import numpy as np
import pandas as pd
import si_units as si
from rdkit import Chem
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/hydrocarbons_initial.csv"
SPLIT = ROOT / "results/predictions/molecule_split.csv"
RAW = ROOT / "data/raw/esper2023.json"
PRED = ROOT / "results/predictions"
METRICS = ROOT / "results/metrics"
MODELS = ROOT / "results/models"

FEATURES = [
    "molecular_weight", "carbon_count", "ring_count",
    "rotatable_bonds", "branch_points",
]
TARGETS = ["m", "sigma", "epsilon_k"]
FRACTIONS = (0.65, 0.70, 0.75, 0.80)
OUTPUTS = {
    "parameters": PRED / "final_test_parameters.csv",
    "vle": PRED / "final_test_vle_curves.csv",
    "parameter_metrics": METRICS / "final_test_parameter_metrics.csv",
    "thermo_summary": METRICS / "final_test_thermo_summary.csv",
    "per_molecule": METRICS / "final_test_per_molecule.csv",
    "config": METRICS / "final_test_config.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_data():
    for path in (DATA, SPLIT, RAW):
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")

    data = pd.read_csv(DATA, dtype={"smiles": str})
    frozen = pd.read_csv(SPLIT, dtype={"smiles": str})
    if any(df.smiles.isna().any() or df.smiles.duplicated().any()
           for df in (data, frozen)):
        raise ValueError("Missing or duplicate SMILES found")
    if set(data.smiles) != set(frozen.smiles):
        raise ValueError("Frozen split does not contain exactly the dataset molecules")
    if set(frozen["split"]) != {"train", "test"}:
        raise ValueError("Unexpected train/test split labels")
    for col in FEATURES + TARGETS:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
        data[col] = pd.to_numeric(data[col], errors="raise")
    if not np.isfinite(data[FEATURES + TARGETS].to_numpy(dtype=float)).all():
        raise ValueError("Data contain non-finite descriptor or parameter values")
    if (data[TARGETS] <= 0).any().any():
        raise ValueError("Nonpositive published PC-SAFT parameters")

    # Verify exact split selection used by src/train_baselines.py.
    sorted_data = data.sort_values("smiles").reset_index(drop=True)
    train_indices, test_indices = train_test_split(
        np.arange(len(sorted_data)), test_size=0.20,
        random_state=42, shuffle=True,
    )
    expected_train = set(sorted_data.iloc[train_indices].smiles)
    expected_test = set(sorted_data.iloc[test_indices].smiles)
    actual_train = set(frozen.loc[frozen["split"] == "train", "smiles"])
    actual_test = set(frozen.loc[frozen["split"] == "test", "smiles"])
    if expected_train != actual_train or expected_test != actual_test:
        raise ValueError("Saved split differs from the original frozen random_state=42 split")
    if len(data) != 194 or len(actual_train) != 155 or len(actual_test) != 39:
        raise ValueError("Expected 194 total, 155 train and 39 test molecules")

    joined = data.merge(frozen, on="smiles", validate="one_to_one")
    train = joined[joined["split"] == "train"].sort_values("smiles").copy()
    test = joined[joined["split"] == "test"].sort_values("smiles").copy()
    if set(train.smiles) & set(test.smiles):
        raise RuntimeError("Data leakage: overlapping molecule identifiers")
    return train, test


def reference_records():
    """Use the same hydrogen-safe reference matching as development evaluation."""
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    with RAW.open(encoding="utf-8") as stream:
        records = json.load(stream)
    lookup = {}
    for record in records:
        if any(k in record for k in ("mu", "q", "association_sites")):
            continue
        smiles = record.get("identifier", {}).get("smiles")
        if not isinstance(smiles, str) or not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles, parser)
        if mol is None:
            continue
        atoms = list(mol.GetAtoms())
        if any(a.GetAtomicNum() == 1 and a.GetDegree() == 0 for a in atoms):
            continue
        if len(Chem.GetMolFrags(mol)) != 1:
            continue
        if not any(a.GetAtomicNum() == 6 for a in atoms) or any(
            a.GetAtomicNum() not in (1, 6) for a in atoms
        ):
            continue
        canonical = Chem.MolToSmiles(
            Chem.RemoveHs(mol), canonical=True, isomericSmiles=True
        )
        lookup.setdefault(canonical, []).append(record)
    return lookup


def match_reference(row, lookup):
    values = {k: float(row[k]) for k in TARGETS}
    matches = [record for record in lookup.get(row["smiles"], [])
               if all(k in record and abs(float(record[k]) - values[k]) < 1e-4
                      for k in TARGETS)]
    if not matches:
        raise RuntimeError(
            f"No exact published-parameter record for {row['name']} ({row['smiles']}). "
            "Investigate source labels before final evaluation."
        )
    if len(matches) > 1:
        print(f"Reference note: {row['name']} has {len(matches)} matching records; using first")
    return matches[0]


def make_eos(record, parameters=None):
    updated = copy.deepcopy(record)
    if parameters is not None:
        updated.update({name: float(parameters[name]) for name in TARGETS})
    pure = feos.PureRecord.from_json_str(json.dumps(updated))
    return feos.EquationOfState.pcsaft(feos.Parameters.new_pure(pure))


def get_equilibrium(eos, temperature_k):
    try:
        eq = feos.PhaseEquilibrium.pure(
            eos, temperature_or_pressure=float(temperature_k) * si.KELVIN
        )
        pressure = float(eq.liquid.pressure() / si.BAR)
        density = float(eq.liquid.mass_density() / (si.KILOGRAM / si.METER**3))
        if not all(math.isfinite(v) and v > 0 for v in (pressure, density)):
            raise ValueError("Nonpositive or nonfinite saturation property")
        return pressure, density, "ok"
    except Exception as error:
        return math.nan, math.nan, f"{type(error).__name__}: {error}"


def pct_error(predicted, reference):
    if math.isfinite(predicted) and math.isfinite(reference) and reference > 0:
        return 100.0 * abs(predicted - reference) / reference
    return math.nan


def run_final(train, test, lookup):
    # Do not silently overwrite final held-out results.
    existing = [p for p in OUTPUTS.values() if p.exists()]
    if existing:
        raise RuntimeError(
            "Final results already exist. Do not rerun/tune after looking at test results. "
            f"Existing: {existing}"
        )
    print("FINAL EVALUATION: frozen Ridge_5 (alpha=1), 155 train / 39 test")
    print("This is the one-time held-out evaluation. No further model selection.")

    # Lookups use the reference labels from the held-out table only for scoring
    # and reference EOS construction. They are never supplied to the ML fit.
    reference_by_smiles = {
        row["smiles"]: match_reference(row, lookup) for _, row in test.iterrows()
    }
    predictions = test[["name", "cas", "smiles"]].copy()
    parameter_metrics = []
    trained_models = {}
    for target in TARGETS:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        model.fit(train[FEATURES], train[target])
        predicted = model.predict(test[FEATURES])
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(train[FEATURES], train[target])
        baseline_predicted = baseline.predict(test[FEATURES])
        predictions[f"reference_{target}"] = test[target].to_numpy(dtype=float)
        predictions[f"predicted_{target}"] = predicted
        predictions[f"dummy_{target}"] = baseline_predicted
        trained_models[target] = model
        for label, yhat in [("Ridge_5", predicted), ("Mean_Baseline", baseline_predicted)]:
            parameter_metrics.append({
                "model": label, "target": target,
                "n_test_molecules": len(test),
                "mae": float(mean_absolute_error(test[target], yhat)),
                "rmse": float(math.sqrt(mean_squared_error(test[target], yhat))),
            })

    # Always report missing/nonpositive predictions rather than clip or omit them.
    rows = []
    for _, molecule in predictions.iterrows():
        name, smiles = str(molecule["name"]), str(molecule["smiles"])
        source = reference_by_smiles[smiles]
        p = {target: float(molecule[f"predicted_{target}"]) for target in TARGETS}
        predicted_valid = all(math.isfinite(value) and value > 0 for value in p.values())
        ref_eos = None
        pred_eos = None
        ref_setup = "ok"
        pred_setup = "ok"
        try:
            ref_eos = make_eos(source)
            tc_ref = float(feos.State.critical_point(ref_eos).temperature / si.KELVIN)
            if not (math.isfinite(tc_ref) and tc_ref > 0):
                raise ValueError("Nonfinite/nonpositive reference critical temperature")
        except Exception as error:
            tc_ref = math.nan
            ref_setup = f"reference_setup_or_Tc_failure: {type(error).__name__}: {error}"
        if not predicted_valid:
            pred_setup = "nonpositive_or_nonfinite_predicted_parameters"
        else:
            try:
                pred_eos = make_eos(source, p)
            except Exception as error:
                pred_setup = f"predicted_EOS_setup_failure: {type(error).__name__}: {error}"

        for fraction in FRACTIONS:
            temperature = fraction * tc_ref if math.isfinite(tc_ref) else math.nan
            if ref_eos is not None and math.isfinite(temperature):
                pref, rhoref, sref = get_equilibrium(ref_eos, temperature)
            else:
                pref, rhoref, sref = math.nan, math.nan, ref_setup
            if pred_eos is not None and math.isfinite(temperature):
                pml, rhoml, sml = get_equilibrium(pred_eos, temperature)
            else:
                pml, rhoml = math.nan, math.nan
                sml = pred_setup if pred_setup != "ok" else "reference_Tc_unavailable"
            rows.append({
                "name": name, "cas": molecule["cas"], "smiles": smiles,
                "temperature_fraction_of_ref_Tc": fraction,
                "temperature_K": temperature, "reference_Tc_K": tc_ref,
                "pressure_reference_bar": pref, "pressure_ml_bar": pml,
                "pressure_ape_percent": pct_error(pml, pref),
                "density_reference_kg_m3": rhoref, "density_ml_kg_m3": rhoml,
                "density_ape_percent": pct_error(rhoml, rhoref),
                "reference_status": sref, "predicted_status": sml,
                **{f"reference_{k}": float(molecule[f"reference_{k}"]) for k in TARGETS},
                **{f"predicted_{k}": p[k] for k in TARGETS},
            })
        print(f"Evaluated {name}")

    curves = pd.DataFrame(rows)
    if len(predictions) != 39 or len(curves) != 156:
        raise AssertionError("Expected 39 final molecules and 156 temperature rows")
    by_molecule = []
    for (smiles, name), part in curves.groupby(["smiles", "name"], sort=True):
        by_molecule.append({
            "smiles": smiles, "name": name,
            "n_attempted": len(part),
            "n_comparable_pressure": int(part.pressure_ape_percent.notna().sum()),
            "n_comparable_density": int(part.density_ape_percent.notna().sum()),
            "mean_pressure_ape_percent": float(part.pressure_ape_percent.mean()),
            "mean_density_ape_percent": float(part.density_ape_percent.mean()),
        })
    molecule_results = pd.DataFrame(by_molecule)
    thermo_summary = pd.DataFrame([{
        "model": "Ridge_5", "n_test_molecules": len(test),
        "n_attempted": len(curves),
        "n_reference_success": int((curves.reference_status == "ok").sum()),
        "n_predicted_success": int((curves.predicted_status == "ok").sum()),
        "n_comparable_pressure": int(curves.pressure_ape_percent.notna().sum()),
        "n_comparable_density": int(curves.density_ape_percent.notna().sum()),
        "mean_pressure_ape_percent_on_comparable": float(curves.pressure_ape_percent.mean()),
        "median_molecule_pressure_ape_percent": float(molecule_results.mean_pressure_ape_percent.median()),
        "mean_density_ape_percent_on_comparable": float(curves.density_ape_percent.mean()),
        "median_molecule_density_ape_percent": float(molecule_results.mean_density_ape_percent.median()),
        "failed_pressure_comparisons": int(curves.pressure_ape_percent.isna().sum()),
        "failed_density_comparisons": int(curves.density_ape_percent.isna().sum()),
    }])
    config = {
        "phase": "FINAL HELD-OUT TEST: one-time evaluation; no further model selection",
        "training_molecules": len(train), "held_out_test_molecules": len(test),
        "chosen_model": "Ridge_5",
        "chosen_based_on": "Development validation pressure MAPE: Ridge_5 16.2203%; Ridge_12 17.9553%; ExtraTrees_156 25.9094%",
        "target_parameters": TARGETS, "features": FEATURES,
        "estimator": "SimpleImputer(median) + StandardScaler + Ridge(alpha=1.0)",
        "test_split_random_state": 42, "temperature_fractions_of_reference_Tc": list(FRACTIONS),
        "source_dataset_sha256": digest(DATA), "frozen_split_sha256": digest(SPLIT),
        "raw_pcsaft_sha256": digest(RAW),
        "interpretation": "Errors relative to published-parameter PC-SAFT, not independent experimental measurements; published reference Tc defines evaluation temperature grid.",
        "metric_caveat": "Pressure/density MAPE calculated only for comparable states; inspect failed counts and per-molecule table.",
    }

    # Only publish results after every molecule has been processed.
    PRED.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUTS["parameters"], index=False)
    curves.to_csv(OUTPUTS["vle"], index=False)
    pd.DataFrame(parameter_metrics).to_csv(OUTPUTS["parameter_metrics"], index=False)
    thermo_summary.to_csv(OUTPUTS["thermo_summary"], index=False)
    molecule_results.to_csv(OUTPUTS["per_molecule"], index=False)
    OUTPUTS["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    for name, model in trained_models.items():
        joblib.dump(model, MODELS / f"final_Ridge_5_{name}.joblib")
    print("\nFINAL HELD-OUT PARAMETER METRICS:")
    print(pd.DataFrame(parameter_metrics).to_string(index=False))
    print("\nFINAL HELD-OUT THERMODYNAMIC SUMMARY:")
    print(thermo_summary.to_string(index=False))
    print("\nSaved files:")
    for path in OUTPUTS.values():
        print(" -", path)
    print("The final held-out evaluation is complete. Do NOT tune the model to these results.")


def main():
    args = argparse.ArgumentParser(description=__doc__)
    mode = args.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="check inputs and reference matching; do not train or evaluate")
    mode.add_argument("--run-final", action="store_true", help="run the one-time held-out evaluation")
    parsed = args.parse_args()
    train, test = load_frozen_data()
    lookup = reference_records()
    # Check ALL final-test reference matching without predicting anything.
    for _, row in test.iterrows():
        match_reference(row, lookup)
    print(f"PRECHECK PASS: {len(train)} train, {len(test)} frozen held-out; all {len(test)} reference records match")
    print("Frozen model: Ridge(alpha=1.0), original 5 descriptors; reference-Tc grid: 0.65,0.70,0.75,0.80")
    if parsed.preflight:
        print("No model fitted and no final test calculations performed.")
    else:
        run_final(train, test, lookup)


if __name__ == "__main__":
    main()
