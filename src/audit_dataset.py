import json
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

source = Path("data/raw/esper2023.json")
output = Path("data/processed/hydrocarbons_initial.csv")

with source.open("r", encoding="utf-8") as file:
    records = json.load(file)

rows = []
seen_smiles = set()
duplicate_count = 0

for record in records:
    # Only use records without extra interaction terms.
    if any(
        key in record
        for key in ("mu", "q", "association_sites")
    ):
        continue

    identifier = record.get("identifier") or {}
    smiles = identifier.get("smiles")

    if not smiles:
        continue

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue

    # One neutral molecule, not multiple fragments.
    if len(Chem.GetMolFrags(mol)) != 1:
        continue

    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        continue

    # Carbon and hydrogen only.
    if any(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms()):
        continue

    # Saturated compounds only: no double, triple,
    # or aromatic bonds.
    if any(
        bond.GetBondType() != Chem.BondType.SINGLE
        for bond in mol.GetBonds()
    ):
        continue

    carbon_count = sum(
        atom.GetAtomicNum() == 6
        for atom in mol.GetAtoms()
    )

    if not (3 <= carbon_count <= 20):
        continue

    try:
        m = float(record["m"])
        sigma = float(record["sigma"])
        epsilon_k = float(record["epsilon_k"])
    except (KeyError, TypeError, ValueError):
        continue

    if not all(
        math.isfinite(x) and x > 0
        for x in (m, sigma, epsilon_k)
    ):
        continue

    canonical = Chem.MolToSmiles(mol, canonical=True)

    if canonical in seen_smiles:
        duplicate_count += 1
        continue

    seen_smiles.add(canonical)

    rows.append({
        "name": identifier.get("name"),
        "cas": identifier.get("cas"),
        "smiles": canonical,
        "carbon_count": carbon_count,
        "molecular_weight": Descriptors.MolWt(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "branch_points": sum(
            atom.GetDegree() >= 3
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 6
        ),
        "m": m,
        "sigma": sigma,
        "epsilon_k": epsilon_k,
    })

df = pd.DataFrame(rows)

print("Original parameter records:", len(records))
print("Eligible unique hydrocarbons:", len(df))
print("Repeated molecular structures skipped:", duplicate_count)

if df.empty:
    raise RuntimeError(
        "No eligible hydrocarbons found. Check filtering rules."
    )

print("\nDistribution by carbon count:")
print(df.groupby("carbon_count").size().to_string())

print("\nSample molecules:")
print(
    df[["name", "smiles", "m", "sigma", "epsilon_k"]]
    .head(10)
    .to_string(index=False)
)

output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False)

print("\nSaved initial dataset to:", output)
