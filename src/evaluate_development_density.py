r"""Compare saturated-liquid density using frozen development parameter predictions.

Run from repository root:
    .\.venv\Scripts\python.exe src/evaluate_development_density.py

Reads the 39-molecule development comparison only; never evaluates final test.
"""
from __future__ import annotations

import math
from pathlib import Path

import feos
import pandas as pd
import si_units as si

from compare_development_vle_models_all39 import (
    ROOT, TARGETS, load_data, load_reference_lookup, make_eos,
)

SOURCE = ROOT / "results/predictions/development_model_vle_curves_all39.csv"
OUTPUT = ROOT / "results/predictions/development_density_curves_all39.csv"
SUMMARY = ROOT / "results/metrics/development_density_summary_all39.csv"
MODELS = ("Ridge_5", "Ridge_12", "ExtraTrees_156")
FRACTIONS = (0.65, 0.70, 0.75, 0.80)


def state_properties(eos, temp_k):
    eq = feos.PhaseEquilibrium.pure(
        eos, temperature_or_pressure=float(temp_k) * si.KELVIN
    )
    pressure = float(eq.liquid.pressure() / si.BAR)
    density = float(eq.liquid.mass_density() / (si.KILOGRAM / si.METER**3))
    if not all(math.isfinite(v) and v > 0 for v in (pressure, density)):
        raise ValueError("Invalid pressure or saturated-liquid density")
    return pressure, density


def calc(eos, temp):
    try:
        pressure, density = state_properties(eos, temp)
        return {"pressure": pressure, "density": density, "status": "ok"}
    except Exception as exc:
        return {
            "pressure": math.nan, "density": math.nan,
            "status": f"{type(exc).__name__}: {exc}",
        }


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Run the all39 VLE comparison first: {SOURCE}")
    data = pd.read_csv(SOURCE)
    _, dev_validation = load_data()
    expected_smiles = set(dev_validation.smiles)
    if set(data.smiles) != expected_smiles or len(expected_smiles) != 39:
        raise RuntimeError("CSV does not match frozen 39-molecule development validation")
    if set(data.model) != set(MODELS) or len(data) != 39 * 4 * 3:
        raise RuntimeError("Expected three models with 156 rows each")
    keys = ["model", "smiles", "temperature_fraction_of_ref_Tc"]
    if data.duplicated(keys).any():
        raise RuntimeError("Duplicate model/molecule/temperature rows")
    if set(round(float(x), 5) for x in data.temperature_fraction_of_ref_Tc) != set(FRACTIONS):
        raise RuntimeError("Unexpected temperature grid")
    if ((data.reference_status != "ok") | (data.predicted_status != "ok")).any():
        raise RuntimeError("Previous run had failures: inspect before density analysis")

    lookup = load_reference_lookup()
    output_rows = []
    max_pressure_disagreement = 0.0
    for smiles, molecule_rows in data.groupby("smiles", sort=True):
        molecule = molecule_rows.iloc[0]
        target = {k: float(molecule[f"reference_{k}"]) for k in TARGETS}
        records = [
            r for r in lookup.get(smiles, [])
            if all(abs(float(r[k]) - target[k]) < 1e-4 for k in TARGETS)
        ]
        if len(records) != 1:
            raise RuntimeError(f"Expected one matched reference record for {smiles}, got {len(records)}")
        reference_eos = make_eos(records[0])
        ref_by_temp = {}
        for fraction, group in molecule_rows.groupby("temperature_fraction_of_ref_Tc"):
            temp = float(group.iloc[0].temperature_K)
            if not all(abs(float(t) - temp) < 1e-7 for t in group.temperature_K):
                raise RuntimeError(f"Inconsistent grid for {smiles}")
            ref_by_temp[float(fraction)] = calc(reference_eos, temp)

        for model_name in MODELS:
            model_rows = molecule_rows.loc[molecule_rows.model == model_name]
            first = model_rows.iloc[0]
            predicted = {k: float(first[f"predicted_{k}"]) for k in TARGETS}
            if any(
                (model_rows[f"predicted_{k}"] - predicted[k]).abs().max() > 1e-7
                for k in TARGETS
            ):
                raise RuntimeError(f"Model parameters change across temperatures: {smiles}")
            if not all(math.isfinite(v) and v > 0 for v in predicted.values()):
                predicted_eos = None
                setup_status = "invalid_predicted_parameters"
            else:
                try:
                    predicted_eos = make_eos(records[0], predicted)
                    setup_status = "ok"
                except Exception as exc:
                    predicted_eos = None
                    setup_status = f"setup_{type(exc).__name__}: {exc}"

            for _, row in model_rows.iterrows():
                fraction = float(row.temperature_fraction_of_ref_Tc)
                ref = ref_by_temp[fraction]
                pred = calc(predicted_eos, float(row.temperature_K)) if predicted_eos else {
                    "pressure": math.nan, "density": math.nan, "status": setup_status
                }
                for label, recomputed, saved in (
                    ("reference", ref["pressure"], float(row.pressure_reference_bar)),
                    ("ML", pred["pressure"], float(row.pressure_ml_bar)),
                ):
                    if math.isfinite(recomputed) and math.isfinite(saved):
                        relative = abs(recomputed - saved) / max(abs(saved), 1e-12)
                        max_pressure_disagreement = max(max_pressure_disagreement, relative)
                        if relative > 1e-4:
                            raise RuntimeError(
                                f"{label} pressure differs from prior CSV for {smiles}, "
                                f"{model_name}, T={row.temperature_K}: {relative:.3g}"
                            )
                density_ape = (
                    100.0 * abs(pred["density"] - ref["density"]) / ref["density"]
                    if ref["status"] == "ok" and pred["status"] == "ok" else math.nan
                )
                output_rows.append({
                    "model": model_name, "name": row["name"], "smiles": smiles,
                    "temperature_fraction_of_ref_Tc": fraction,
                    "temperature_K": float(row.temperature_K),
                    "pressure_reference_bar": ref["pressure"],
                    "pressure_ml_bar": pred["pressure"],
                    "pressure_ape_percent": float(row.absolute_percentage_error),
                    "density_reference_kg_m3": ref["density"],
                    "density_ml_kg_m3": pred["density"],
                    "density_ape_percent": density_ape,
                    "reference_status": ref["status"],
                    "predicted_status": pred["status"],
                })
        print(f"Density evaluated: {molecule['name']}")

    curves = pd.DataFrame(output_rows)
    summary = []
    for model_name, section in curves.groupby("model", sort=False):
        by_molecule = section.groupby("smiles")["density_ape_percent"].mean()
        summary.append({
            "model": model_name,
            "n_attempted": len(section),
            "n_comparable_density": int(section.density_ape_percent.notna().sum()),
            "n_reference_success": int((section.reference_status == "ok").sum()),
            "n_predicted_success": int((section.predicted_status == "ok").sum()),
            "mean_density_ape_percent": section.density_ape_percent.mean(),
            "median_molecule_density_ape_percent": by_molecule.median(),
            "mean_pressure_ape_percent": section.pressure_ape_percent.mean(),
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(OUTPUT, index=False)
    table = pd.DataFrame(summary)
    table.to_csv(SUMMARY, index=False)
    print("\nDEVELOPMENT density and pressure comparison (NOT final test):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Max relative pressure disagreement with saved curves: {max_pressure_disagreement:.3g}")
    print("Saved:", OUTPUT, SUMMARY, sep="\n")
    print("The 39 final-test molecules remain untouched.")


if __name__ == "__main__":
    main()

