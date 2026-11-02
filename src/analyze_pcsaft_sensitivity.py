"""Finite-difference PC-SAFT pressure sensitivity on development molecules only.

Uses published reference parameters, one temperature (0.75 reference Tc),
and a symmetric 1% perturbation per parameter. Never reads final test results.
"""
from __future__ import annotations

import math

import feos
import pandas as pd
import si_units as si

from compare_development_vle_models_all39 import (
    ROOT, TARGETS, load_data, load_reference_lookup, make_eos,
)

SOURCE = ROOT / "results/predictions/development_model_vle_curves_all39.csv"
SENS_OUT = ROOT / "results/metrics/development_pressure_sensitivities.csv"
CONTRIB_OUT = ROOT / "results/metrics/development_pressure_error_decomposition.csv"
NAMES = (
    "cyclopropane", "cyclobutane", "4-methyltridecane",
    "decane", "1,1,3-trimethylcyclohexane",
)
FRACTION = 0.75
DELTA = 0.01


def pressure(eos, temp):
    eq = feos.PhaseEquilibrium.pure(eos, temperature_or_pressure=temp * si.KELVIN)
    value = float(eq.liquid.pressure() / si.BAR)
    if not (math.isfinite(value) and value > 0):
        raise ValueError("Invalid saturation pressure")
    return value


def main():
    data = pd.read_csv(SOURCE)
    _, validation = load_data()
    if len(validation) != 39 or set(data.smiles) != set(validation.smiles):
        raise RuntimeError("Expected only the frozen 39-molecule development set")
    if data.duplicated(["model", "smiles", "temperature_fraction_of_ref_Tc"]).any():
        raise RuntimeError("Duplicate development pressure records")
    lookup = load_reference_lookup()
    sensitivities = []
    contributions = []
    for name in NAMES:
        subset = data.loc[
            (data.name == name)
            & (abs(data.temperature_fraction_of_ref_Tc - FRACTION) < 1e-8)
        ]
        if len(subset) != 3 or set(subset.model) != {
            "Ridge_5", "Ridge_12", "ExtraTrees_156"
        }:
            raise RuntimeError(f"Expected 3 model rows for {name}, found {len(subset)}")
        row = subset.iloc[0]
        smiles = str(row.smiles)
        reference = {p: float(row[f"reference_{p}"]) for p in TARGETS}
        matches = [
            r for r in lookup.get(smiles, [])
            if all(abs(float(r[p]) - reference[p]) < 1e-4 for p in TARGETS)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Reference record mismatch for {name}")
        record = matches[0]
        temp = float(row.temperature_K)
        p0 = pressure(make_eos(record), temp)
        if abs(p0 - float(row.pressure_reference_bar)) / p0 > 1e-4:
            raise RuntimeError(f"Reference saturation pressure mismatch for {name}")
        local_sens = {}
        for parameter in TARGETS:
            plus = reference.copy()
            minus = reference.copy()
            plus[parameter] *= 1 + DELTA
            minus[parameter] *= 1 - DELTA
            status = "ok"
            p_plus = math.nan
            p_minus = math.nan
            sens = math.nan
            try:
                p_plus = pressure(make_eos(record, plus), temp)
                p_minus = pressure(make_eos(record, minus), temp)
                sens = math.log(p_plus / p_minus) / math.log((1 + DELTA) / (1 - DELTA))
            except Exception as exc:
                status = f"{type(exc).__name__}: {exc}"
            local_sens[parameter] = sens
            sensitivities.append({
                "name": name, "smiles": smiles, "temperature_K": temp,
                "parameter": parameter, "delta_fraction": DELTA,
                "S_dlnP_dlnparameter": sens,
                "pressure_reference_bar": p0,
                "pressure_plus_bar": p_plus,
                "pressure_minus_bar": p_minus,
                "status": status,
            })

        for _, model_row in subset.iterrows():
            pred = {p: float(model_row[f"predicted_{p}"]) for p in TARGETS}
            actual = math.log(float(model_row.pressure_ml_bar) / p0)
            components = {
                p: local_sens[p] * math.log(pred[p] / reference[p])
                if math.isfinite(local_sens[p]) and pred[p] > 0 else math.nan
                for p in TARGETS
            }
            approx = sum(components.values()) if all(math.isfinite(v) for v in components.values()) else math.nan
            contributions.append({
                "name": name, "model": model_row.model,
                "temperature_K": temp,
                "actual_ln_pressure_ratio": actual,
                "first_order_ln_pressure_ratio": approx,
                "linearization_residual": actual - approx,
                **{f"contribution_{p}": components[p] for p in TARGETS},
            })
        print(f"Sensitivity evaluated: {name}")

    SENS_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sensitivities).to_csv(SENS_OUT, index=False)
    pd.DataFrame(contributions).to_csv(CONTRIB_OUT, index=False)
    print("\nSigned local sensitivities d ln(Psat) / d ln(parameter):")
    print(pd.DataFrame(sensitivities).pivot(index="name", columns="parameter", values="S_dlnP_dlnparameter").round(3).to_string())
    print("\nOutputs:", SENS_OUT, CONTRIB_OUT, sep="\n")
    print("First-order contributions are explanatory approximations, not causal proof.")
    print("Final test molecules remain untouched.")


if __name__ == "__main__":
    main()
