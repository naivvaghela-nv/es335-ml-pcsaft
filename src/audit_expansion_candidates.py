r"""Audit possible additional hydrocarbons in Esper2023; NEVER modifies existing data.

From PowerShell repository root:
    .\.venv\Scripts\python.exe src/audit_expansion_candidates.py
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/esper2023.json"
EXISTING = ROOT / "data/processed/hydrocarbons_initial.csv"
OUT = ROOT / "results/metrics/expansion_candidate_audit.csv"
TARGETS = ("m", "sigma", "epsilon_k")


def main() -> None:
    with RAW.open(encoding="utf-8") as f:
        records = json.load(f)
    original = pd.read_csv(EXISTING, dtype={"smiles": str})
    original_smiles = set(original["smiles"])

    params = Chem.SmilesParserParams()
    params.removeHs = False
    skip = Counter()
    candidates = {}

    for record in records:
        if any(key in record for key in ("mu", "q", "association_sites")):
            skip["additional_EOS_terms"] += 1
            continue
        smiles = record.get("identifier", {}).get("smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            skip["no_SMILES"] += 1
            continue
        mol = Chem.MolFromSmiles(smiles, params)
        if mol is None:
            skip["invalid_SMILES"] += 1
            continue
        atoms = list(mol.GetAtoms())
        if any(a.GetAtomicNum() == 1 and a.GetDegree() == 0 for a in atoms):
            skip["isolated_hydrogen"] += 1
            continue
        if len(Chem.GetMolFrags(mol)) != 1:
            skip["multiple_fragments"] += 1
            continue
        if any(a.GetAtomicNum() not in (1, 6) for a in atoms) or not any(
            a.GetAtomicNum() == 6 for a in atoms
        ):
            skip["not_CH_only"] += 1
            continue
        if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in atoms):
            skip["charged_or_radical"] += 1
            continue
        try:
            vals = [float(record[k]) for k in TARGETS]
        except (ValueError, TypeError, KeyError):
            skip["bad_parameters"] += 1
            continue
        if any(not math.isfinite(v) or v <= 0 for v in vals):
            skip["bad_parameters"] += 1
            continue
        try:
            normal = Chem.RemoveHs(mol)
            canonical = Chem.MolToSmiles(normal, canonical=True, isomericSmiles=True)
        except Exception:
            skip["normalization_failed"] += 1
            continue
        if canonical in original_smiles:
            skip["already_in_initial_dataset"] += 1
            continue
        carbon_count = sum(a.GetAtomicNum() == 6 for a in atoms)
        if carbon_count < 3:
            skip["fewer_than_3_carbons"] += 1
            continue
        saturated = all(
            bond.GetBondType() == Chem.BondType.SINGLE
            for bond in normal.GetBonds()
        )
        if saturated and carbon_count <= 20:
            family = "Saturated C3-C20 missed by initial filter"
        elif saturated:
            family = "Saturated C21+"
        elif carbon_count <= 20:
            family = "Unsaturated/aromatic C3-C20"
        else:
            family = "Unsaturated/aromatic C21+"
        if canonical in candidates:
            skip["duplicate_candidate_structure"] += 1
            # Duplicate fitted parameter records are not assumed equivalent.
            continue
        candidates[canonical] = {
            "family": family,
            "name": record["identifier"].get("name", ""),
            "cas": record["identifier"].get("cas", ""),
            "smiles": canonical,
            "carbon_count": carbon_count,
            "ring_count": normal.GetRingInfo().NumRings(),
            **{k: v for k, v in zip(TARGETS, vals)},
        }
    rows = list(candidates.values())
    frame = pd.DataFrame(rows, columns=[
        "family", "name", "cas", "smiles", "carbon_count", "ring_count",
        "m", "sigma", "epsilon_k",
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["family", "carbon_count", "smiles"]).to_csv(OUT, index=False)
    print("Esper2023 records:", len(records))
    print("Current curated molecules:", len(original))
    print("New unique candidate structures:", len(frame))
    print("\nPotential expansion groups:")
    print(frame.groupby("family", dropna=False).size().to_string() if len(frame) else "None")
    print("\nExcluded/repeated record counts:")
    for key, value in sorted(skip.items()):
        print(f"  {key}: {value}")
    print("\nAudit file:", OUT)
    print("\nNothing was added to the existing dataset or train/test split.")
    print("Candidate labels and compatibility still require manual review.")


if __name__ == "__main__":
    main()
