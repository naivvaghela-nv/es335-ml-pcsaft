r"""Compare baseline Ridge, enhanced Ridge, and rich-feature Extra Trees via FeOs.

For Windows PowerShell (run in project root):
    .\.venv\Scripts\python.exe src/compare_development_vle_models.py

The reserved 39-molecule final test set is neither fitted on nor evaluated.
The original development split and 12 validation molecules match
src/integrate_ridge_feos.py (random_state 17 and 23).

Outputs:
    results/predictions/development_model_vle_curves_all39.csv
    results/metrics/development_model_vle_comparison_all39.csv
    results/metrics/development_model_vle_config_all39.json
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "data/processed/hydrocarbons_initial.csv"
RICH = ROOT / "data/processed/hydrocarbons_richer.csv"
SPLIT = ROOT / "results/predictions/molecule_split.csv"
RAW = ROOT / "data/raw/esper2023.json"
CURVE_OUT = ROOT / "results/predictions/development_model_vle_curves_all39.csv"
SUMMARY_OUT = ROOT / "results/metrics/development_model_vle_comparison_all39.csv"
CONFIG_OUT = ROOT / "results/metrics/development_model_vle_config_all39.json"
PREVIOUS_CURVE = ROOT / "results/predictions/development_vle_curves.csv"

TARGETS = ["m", "sigma", "epsilon_k"]
BASE_5 = [
    "molecular_weight", "carbon_count", "ring_count",
    "rotatable_bonds", "branch_points",
]
BASE_12 = BASE_5 + [
    "BertzCT", "Chi0v", "Chi1v", "Chi2v", "Kappa1", "Kappa2", "BalabanJ",
]
ADDED_16 = [
    "Chi3v", "Chi4v", "Kappa3", "MolLogP", "MolMR", "LabuteASA",
    "NumBridgeheadAtoms", "NumSpiroAtoms", "NumAliphaticRings",
    "NumAtomStereoCenters", "primary_carbons", "secondary_carbons",
    "tertiary_carbons", "quaternary_carbons", "five_member_rings", "six_member_rings",
]
FINGERPRINT = [f"morgan_{i:03d}" for i in range(128)]
FEATURE_SETS = {
    "Ridge_5": BASE_5,
    "Ridge_12": BASE_12,
    "ExtraTrees_156": BASE_12 + ADDED_16 + FINGERPRINT,
}
TEMPERATURE_FRACTIONS = (0.65, 0.70, 0.75, 0.80)
N_MOLECULES = 39


def build_model(label):
    if label.startswith("Ridge_"):
        # Matches the original development Ridge pipeline.
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
        )
    if label == "ExtraTrees_156":
        # Matches src/compare_richer_models.py cross-validation configuration.
        return ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    raise ValueError(f"Unknown model: {label}")


def make_eos(record, predicted=None):
    record = copy.deepcopy(record)
    if predicted is not None:
        record.update({target: float(predicted[target]) for target in TARGETS})
    pure = feos.PureRecord.from_json_str(json.dumps(record))
    return feos.EquationOfState.pcsaft(feos.Parameters.new_pure(pure))


def pressure_bar(eos, temperature_k):
    equilibrium = feos.PhaseEquilibrium.pure(
        eos, temperature_or_pressure=float(temperature_k) * si.KELVIN
    )
    result = float(equilibrium.liquid.pressure() / si.BAR)
    if not (math.isfinite(result) and result > 0):
        raise ValueError(f"Nonpositive/nonfinite pressure: {result}")
    return result


def load_data():
    original = pd.read_csv(INITIAL)
    richer = pd.read_csv(RICH)
    split = pd.read_csv(SPLIT, dtype={"smiles": str})
    for label, table in [("initial", original), ("richer", richer), ("split", split)]:
        if table.smiles.isna().any() or table.smiles.duplicated().any():
            raise ValueError(f"Missing/duplicate SMILES in {label}")
    if not (set(original.smiles) == set(richer.smiles) == set(split.smiles)):
        raise ValueError("Initial, richer and frozen split contain different molecules")
    if set(split["split"].unique()) != {"train", "test"}:
        raise ValueError("Unexpected frozen split labels")

    # Published parameters in the richer dataset must match the curated original.
    for target in TARGETS:
        check = original[["smiles", target]].merge(
            richer[["smiles", target]], on="smiles", validate="one_to_one",
            suffixes=("_old", "_rich")
        )
        if not np.allclose(check[f"{target}_old"], check[f"{target}_rich"],
                           rtol=0, atol=1e-7):
            raise ValueError(f"Reference targets were changed in richer data: {target}")

    features_not_in_original = [
        feature for feature in FEATURE_SETS["ExtraTrees_156"]
        if feature not in original.columns
    ]
    data = original.merge(
        richer[["smiles"] + features_not_in_original],
        on="smiles", validate="one_to_one"
    ).merge(split, on="smiles", validate="one_to_one")

    train = data.loc[data["split"] == "train"].sort_values("smiles").reset_index(drop=True)
    reserved = data.loc[data["split"] == "test"]
    if len(train) != 155 or len(reserved) != 39:
        raise ValueError(f"Expected frozen 155/39 split, got {len(train)}/{len(reserved)}")
    needed = list(dict.fromkeys(FEATURE_SETS["ExtraTrees_156"] + TARGETS))
    if not np.isfinite(train[needed].astype(float).to_numpy()).all():
        raise ValueError("Training-pool data have nonfinite features or labels")

    # Must match integrate_ridge_feos.py EXACTLY: same sorted train, seeds, selection.
    dev_train, dev_valid = train_test_split(
        train, test_size=0.25, random_state=17, shuffle=True
    )
    validation = dev_valid.sample(
        n=min(N_MOLECULES, len(dev_valid)), random_state=23
    ).sort_values("smiles").copy()
    if set(dev_train.smiles) & set(validation.smiles):
        raise AssertionError("Development training/validation leakage")
    print(f"Training pool: {len(train)} | Reserved final test: {len(reserved)}")
    print(f"Development fit: {len(dev_train)} | Development validation: {len(dev_valid)}")
    print(f"Selected development molecules: {len(validation)}")
    return dev_train, validation


def load_reference_lookup():
    with RAW.open(encoding="utf-8") as stream:
        source_records = json.load(stream)

    # This avoids RDKit warnings from disconnected, isolated-H records.
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    lookup = {}
    counts = {"isolated_H": 0, "disconnected": 0, "non_CH": 0, "unparseable": 0}
    for record in source_records:
        if any(key in record for key in ("mu", "q", "association_sites")):
            continue
        smiles = record.get("identifier", {}).get("smiles")
        if not isinstance(smiles, str) or not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles, parser)
        if mol is None:
            counts["unparseable"] += 1
            continue
        atoms = list(mol.GetAtoms())
        if any(atom.GetAtomicNum() == 1 and atom.GetDegree() == 0 for atom in atoms):
            counts["isolated_H"] += 1
            continue
        if len(Chem.GetMolFrags(mol)) != 1:
            counts["disconnected"] += 1
            continue
        if (not any(atom.GetAtomicNum() == 6 for atom in atoms)
                or any(atom.GetAtomicNum() not in (1, 6) for atom in atoms)):
            counts["non_CH"] += 1
            continue
        canonical = Chem.MolToSmiles(
            Chem.RemoveHs(mol), canonical=True, isomericSmiles=True
        )
        lookup.setdefault(canonical, []).append(record)
    print("Reference-record scan exclusions:", counts)
    return lookup


def validate_old_baseline(new_curves):
    """Verify Ridge_5 reproduces the earlier development run if its CSV exists."""
    if not PREVIOUS_CURVE.exists():
        print("Baseline check: previous development CSV absent, skipped.")
        return
    old = pd.read_csv(PREVIOUS_CURVE)
    new = new_curves.loc[new_curves.model == "Ridge_5"]
    keys = ["smiles", "temperature_fraction_of_ref_Tc"]
    merged = old.merge(new, on=keys, validate="one_to_one", suffixes=("_old", "_new"))
    if len(merged) != len(old):
        raise RuntimeError("Original Ridge development molecules/temperature grid changed")
    for field in [
        "predicted_m", "predicted_sigma", "predicted_epsilon_k",
        "pressure_reference_bar", "pressure_ml_bar",
    ]:
        left = merged[f"{field}_old"].astype(float).to_numpy()
        right = merged[f"{field}_new"].astype(float).to_numpy()
        if not np.allclose(left, right, rtol=1e-5, atol=1e-7, equal_nan=True):
            raise RuntimeError(f"Ridge_5 does not reproduce earlier run: {field}")
    print("Baseline check: PASS (Ridge_5 reproduces previous development results).")


def main():
    train, validation = load_data()
    reference_lookup = load_reference_lookup()
    model_predictions = {}
    for label, features in FEATURE_SETS.items():
        predictions = pd.DataFrame(index=validation.index)
        for target in TARGETS:
            model = build_model(label)
            model.fit(train[features], train[target])
            predictions[target] = model.predict(validation[features])
        model_predictions[label] = predictions
        print(f"Trained {label}: {len(features)} features, {len(train)} development fit molecules")

    rows = []
    for idx, molecule in validation.iterrows():
        smiles, name = molecule["smiles"], molecule["name"]
        expected = {key: float(molecule[key]) for key in TARGETS}
        matching_records = [
            rec for rec in reference_lookup.get(smiles, [])
            if all(abs(float(rec[key]) - expected[key]) < 1e-4 for key in TARGETS)
        ]
        if not matching_records:
            raise RuntimeError(
                f"No matching published reference record for {name} ({smiles}). "
                "Investigate before proceeding."
            )
        if len(matching_records) > 1:
            print(f"WARNING: {name} has {len(matching_records)} matching reference records; using first")
        source = matching_records[0]
        reference_eos = make_eos(source)
        try:
            tc = float(feos.State.critical_point(reference_eos).temperature / si.KELVIN)
            if not (math.isfinite(tc) and tc > 0):
                raise ValueError(f"Invalid reference critical temperature {tc}")
            critical_status = "ok"
        except Exception as exc:
            tc = math.nan
            critical_status = f"reference_Tc_failure: {type(exc).__name__}: {exc}"

        reference_points = {}
        for fraction in TEMPERATURE_FRACTIONS:
            temperature = tc * fraction if math.isfinite(tc) else math.nan
            if not math.isfinite(temperature):
                reference_points[fraction] = (temperature, math.nan, critical_status)
            else:
                try:
                    pressure = pressure_bar(reference_eos, temperature)
                    reference_points[fraction] = (temperature, pressure, "ok")
                except Exception as exc:
                    status = f"reference_solver_failure: {type(exc).__name__}: {exc}"
                    reference_points[fraction] = (temperature, math.nan, status)

        for label, predictions in model_predictions.items():
            p = {key: float(predictions.loc[idx, key]) for key in TARGETS}
            valid = all(math.isfinite(v) and v > 0 for v in p.values())
            predicted_eos = None
            setup_error = ""
            if not valid:
                setup_error = "nonpositive_or_nonfinite_predicted_parameter"
            else:
                try:
                    predicted_eos = make_eos(source, p)
                except Exception as exc:
                    setup_error = f"predicted_EOS_setup_failure: {type(exc).__name__}: {exc}"
            for fraction, (temperature, reference_pressure, reference_status) in reference_points.items():
                predicted_pressure = math.nan
                predicted_status = setup_error or "ok"
                if predicted_eos is None:
                    pass
                elif not math.isfinite(temperature):
                    predicted_status = "not_attempted_no_reference_temperature"
                else:
                    try:
                        predicted_pressure = pressure_bar(predicted_eos, temperature)
                    except Exception as exc:
                        predicted_status = f"predicted_solver_failure: {type(exc).__name__}: {exc}"
                ape = (100 * abs(predicted_pressure - reference_pressure) / reference_pressure
                       if math.isfinite(reference_pressure) and reference_pressure > 0
                       and math.isfinite(predicted_pressure) else math.nan)
                rows.append({
                    "model": label,
                    "name": name,
                    "smiles": smiles,
                    "cas": molecule["cas"],
                    "temperature_fraction_of_ref_Tc": fraction,
                    "temperature_K": temperature,
                    "reference_Tc_K": tc,
                    "pressure_reference_bar": reference_pressure,
                    "pressure_ml_bar": predicted_pressure,
                    "absolute_percentage_error": ape,
                    "reference_status": reference_status,
                    "predicted_status": predicted_status,
                    **{f"reference_{key}": expected[key] for key in TARGETS},
                    **{f"predicted_{key}": p[key] for key in TARGETS},
                })
        print(f"Evaluated: {name}")

    curves = pd.DataFrame(rows)
    validate_old_baseline(curves)
    CURVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(CURVE_OUT, index=False)

    summaries = []
    for label in FEATURE_SETS:
        part = curves.loc[curves.model == label]
        per_molecule = part.groupby("smiles")["absolute_percentage_error"].mean()
        summaries.append({
            "model": label,
            "n_molecules": int(part.smiles.nunique()),
            "n_attempted": len(part),
            "n_reference_success": int((part.reference_status == "ok").sum()),
            "n_predicted_success": int((part.predicted_status == "ok").sum()),
            "n_comparable": int(part.absolute_percentage_error.notna().sum()),
            "mean_ape_percent_on_comparable": float(part.absolute_percentage_error.mean()),
            "macro_mean_molecule_ape_percent": float(per_molecule.mean()),
            "median_molecule_ape_percent": float(per_molecule.median()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(SUMMARY_OUT, index=False)
    print("\nDEVELOPMENT comparison (NOT final test):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    config = {
        "source": "Esper2023 curated saturated hydrocarbons, 194 total",
        "frozen_training_pool": 155,
        "frozen_final_test_pool_untouched": 39,
        "development_train_seed": 17,
        "development_validation_sample_seed": 23,
        "development_validation_molecules": N_MOLECULES,
        "temperature_fractions_of_reference_Tc": list(TEMPERATURE_FRACTIONS),
        "models": {
            "Ridge_5": "SimpleImputer(median) + StandardScaler + Ridge(alpha=1)",
            "Ridge_12": "SimpleImputer(median) + StandardScaler + Ridge(alpha=1)",
            "ExtraTrees_156": "ExtraTrees(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1)",
        },
        "features": FEATURE_SETS,
        "limitations": [
            "Development-validation comparison only, NOT an independent final test",
            "Reference pressure means published-parameter PC-SAFT, not experimental truth",
            "Mean percentage errors are conditional on successful calculations; inspect failure counts",
        ],
    }
    CONFIG_OUT.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\nFiles written:", CURVE_OUT, SUMMARY_OUT, CONFIG_OUT, sep="\n")
    print("The frozen 39-molecule final test set was not evaluated.")


if __name__ == "__main__":
    main()


