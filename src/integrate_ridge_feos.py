"""Training-only validation of Ridge -> PC-SAFT -> saturation pressure.

Run from the repo root using the Python executable in .venv.
This script does not train on or evaluate the reserved final test molecules.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import feos
import numpy as np
import pandas as pd
import si_units as si
from rdkit import Chem
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "molecular_weight", "carbon_count", "ring_count",
    "rotatable_bonds", "branch_points",
]
TARGETS = ["m", "sigma", "epsilon_k"]
TEMP_FRACTIONS = (0.65, 0.70, 0.75, 0.80)
N_MOLECULES = 12


def eos_from_record(record, values=None):
    """Clone a published FeOs record, optionally replacing its 3 PC-SAFT parameters."""
    updated = copy.deepcopy(record)
    if values is not None:
        updated.update({k: float(values[k]) for k in TARGETS})
    pure_record = feos.PureRecord.from_json_str(json.dumps(updated))
    return feos.EquationOfState.pcsaft(feos.Parameters.new_pure(pure_record))


def saturation_pressure_bar(eos, temperature_k):
    eq = feos.PhaseEquilibrium.pure(
        eos, temperature_or_pressure=float(temperature_k) * si.KELVIN
    )
    pressure_bar = float(eq.liquid.pressure() / si.BAR)
    if not math.isfinite(pressure_bar) or pressure_bar <= 0:
        raise ValueError(f"Nonphysical saturation pressure: {pressure_bar}")
    return pressure_bar


def main():
    data_path = ROOT / "data/processed/hydrocarbons_initial.csv"
    split_path = ROOT / "results/predictions/molecule_split.csv"
    raw_path = ROOT / "data/raw/esper2023.json"
    output_dir = ROOT / "results/predictions"
    metrics_dir = ROOT / "results/metrics"
    for path in (output_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    split = pd.read_csv(split_path, dtype={"smiles": str})
    if df.smiles.duplicated().any() or split.smiles.duplicated().any():
        raise ValueError("Duplicate SMILES in data or recorded split")
    if set(df.smiles) != set(split.smiles):
        raise ValueError("Dataset does not match frozen molecule split")
    df = df.merge(split, on="smiles", validate="one_to_one")
    train = df.loc[df.split == "train"].sort_values("smiles").reset_index(drop=True)
    reserved = (df.split == "test").sum()
    print(f"Train-pool molecules: {len(train)} | Untouched test: {reserved}")

    # Only split the training pool. Never use the final test set here.
    dev_train, dev_valid = train_test_split(
        train, test_size=0.25, random_state=17, shuffle=True
    )
    validation = dev_valid.sample(
        n=min(N_MOLECULES, len(dev_valid)), random_state=23
    ).sort_values("smiles").copy()

    for col in FEATURES + TARGETS:
        if not np.isfinite(train[col].astype(float).to_numpy()).all():
            raise ValueError(f"Missing or nonfinite column: {col}")

    predicted = pd.DataFrame(index=validation.index)
    for target in TARGETS:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        model.fit(dev_train[FEATURES], dev_train[target])
        predicted[target] = model.predict(validation[FEATURES])

    # Exact reference records: reproduce molecular weight and other metadata
    # from the published JSON, not RDKit-rounded values in the feature table.
    with raw_path.open(encoding="utf-8") as f:
        raw_records = json.load(f)
    record_lookup = {}
    for record in raw_records:
        if any(key in record for key in ("mu", "q", "association_sites")):
            continue
        smiles = record.get("identifier", {}).get("smiles")
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            record_lookup.setdefault(canonical, []).append(record)

    rows = []
    for idx, molecule in validation.iterrows():
        smiles = molecule["smiles"]
        name = str(molecule["name"])
        reference_values = {k: float(molecule[k]) for k in TARGETS}
        predicted_values = {k: float(predicted.loc[idx, k]) for k in TARGETS}
        matches = [
            rec for rec in record_lookup.get(smiles, [])
            if all(abs(float(rec[k]) - reference_values[k]) < 0.0001 for k in TARGETS)
        ]
        if not matches:
            raise RuntimeError(f"Could not match original fitted record: {name} ({smiles})")
        source_record = matches[0]

        predicted_ok = all(
            math.isfinite(value) and value > 0
            for value in predicted_values.values()
        )
        reference_eos = eos_from_record(source_record)
        try:
            tc = float(feos.State.critical_point(reference_eos).temperature / si.KELVIN)
            if not math.isfinite(tc) or tc <= 0:
                raise ValueError(f"Invalid critical temperature: {tc}")
            tc_error = ""
        except Exception as exc:
            tc, tc_error = math.nan, f"reference_Tc_failure: {type(exc).__name__}: {exc}"

        predicted_eos = None
        predicted_setup_error = ""
        if not predicted_ok:
            predicted_setup_error = "nonpositive_or_nonfinite_predicted_parameter"
        else:
            try:
                predicted_eos = eos_from_record(source_record, predicted_values)
            except Exception as exc:
                predicted_setup_error = f"predicted_EOS_failure: {type(exc).__name__}: {exc}"

        for fraction in TEMP_FRACTIONS:
            temperature = tc * fraction if math.isfinite(tc) else math.nan
            p_ref, p_ml = math.nan, math.nan
            reference_status = tc_error or "ok"
            predicted_status = predicted_setup_error or "ok"

            if math.isfinite(temperature):
                try:
                    p_ref = saturation_pressure_bar(reference_eos, temperature)
                except Exception as exc:
                    reference_status = f"reference_solver_failure: {type(exc).__name__}: {exc}"
                if predicted_eos is not None:
                    try:
                        p_ml = saturation_pressure_bar(predicted_eos, temperature)
                    except Exception as exc:
                        predicted_status = f"predicted_solver_failure: {type(exc).__name__}: {exc}"
            else:
                if not predicted_setup_error:
                    predicted_status = "not_attempted_no_reference_temperature"

            ape = 100 * abs(p_ml - p_ref) / p_ref if (
                math.isfinite(p_ref) and p_ref > 0 and math.isfinite(p_ml)
            ) else math.nan
            rows.append({
                "name": name, "smiles": smiles, "cas": molecule["cas"],
                "temperature_fraction_of_ref_Tc": fraction,
                "temperature_K": temperature,
                "reference_Tc_K": tc,
                "pressure_reference_bar": p_ref,
                "pressure_ml_bar": p_ml,
                "absolute_percentage_error": ape,
                "reference_status": reference_status,
                "predicted_status": predicted_status,
                **{f"reference_{k}": reference_values[k] for k in TARGETS},
                **{f"predicted_{k}": predicted_values[k] for k in TARGETS},
            })
        print(f"{name}: predicted {predicted_values}")

    results = pd.DataFrame(rows)
    curve_path = output_dir / "development_vle_curves.csv"
    results.to_csv(curve_path, index=False)
    summary = (
        results.groupby(["name", "smiles"], as_index=False)
        .agg(
            n_points=("temperature_fraction_of_ref_Tc", "size"),
            n_reference_success=("pressure_reference_bar", "count"),
            n_ml_success=("pressure_ml_bar", "count"),
            n_comparable=("absolute_percentage_error", "count"),
            mean_ape_percent=("absolute_percentage_error", "mean"),
        )
    )
    summary_path = metrics_dir / "development_vle_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nDevelopment VLE summary (not the final test):")
    print(summary.to_string(index=False))
    print(f"\nSaved: {curve_path}")
    print(f"Saved: {summary_path}")
    print(f"Comparison successes: {results.absolute_percentage_error.notna().sum()} / {len(results)}")
    print("Final held-out test molecules remain unused.")


if __name__ == "__main__":
    main()
