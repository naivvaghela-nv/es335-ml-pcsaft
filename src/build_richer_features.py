r"""Expand hydrocarbon molecular descriptors without touching the frozen test split.

Run from repository root:
    .\.venv\Scripts\python.exe src/build_richer_features.py

The output is a deterministic feature table, not a trained model.
No reference fitted PC-SAFT parameter is used to generate any feature.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/hydrocarbons_enhanced.csv"
SPLIT_PATH = ROOT / "results/predictions/molecule_split.csv"
OUTPUT = ROOT / "data/processed/hydrocarbons_richer.csv"

BASE_12 = [
    "molecular_weight", "carbon_count", "ring_count",
    "rotatable_bonds", "branch_points", "BertzCT", "Chi0v",
    "Chi1v", "Chi2v", "Kappa1", "Kappa2", "BalabanJ",
]

# These features are computed solely from the molecular graph/SMILES.
ADDED_16 = [
    "Chi3v", "Chi4v", "Kappa3", "MolLogP", "MolMR", "LabuteASA",
    "NumBridgeheadAtoms", "NumSpiroAtoms", "NumAliphaticRings",
    "NumAtomStereoCenters", "primary_carbons", "secondary_carbons",
    "tertiary_carbons", "quaternary_carbons",
    "five_member_rings", "six_member_rings",
]

N_BITS = 128
FP_COLUMNS = [f"morgan_{i:03d}" for i in range(N_BITS)]
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=N_BITS, includeChirality=True
)


def graph_features(mol: Chem.Mol) -> dict[str, float]:
    degree_counts = {degree: 0 for degree in (1, 2, 3, 4)}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            carbon_neighbors = sum(
                nbr.GetAtomicNum() == 6 for nbr in atom.GetNeighbors()
            )
            if carbon_neighbors in degree_counts:
                degree_counts[carbon_neighbors] += 1

    rings = [len(ring) for ring in Chem.GetSymmSSSR(mol)]
    result = {
        "Chi3v": Descriptors.Chi3v(mol),
        "Chi4v": Descriptors.Chi4v(mol),
        "Kappa3": Descriptors.Kappa3(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "MolMR": Descriptors.MolMR(mol),
        "LabuteASA": Descriptors.LabuteASA(mol),
        "NumBridgeheadAtoms": Descriptors.NumBridgeheadAtoms(mol),
        "NumSpiroAtoms": Descriptors.NumSpiroAtoms(mol),
        "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
        "NumAtomStereoCenters": Descriptors.NumAtomStereoCenters(mol),
        "primary_carbons": degree_counts[1],
        "secondary_carbons": degree_counts[2],
        "tertiary_carbons": degree_counts[3],
        "quaternary_carbons": degree_counts[4],
        "five_member_rings": rings.count(5),
        "six_member_rings": rings.count(6),
    }
    assert set(result) == set(ADDED_16)
    return result


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. First run src/build_enhanced_features.py"
        )
    df = pd.read_csv(INPUT)
    split = pd.read_csv(SPLIT_PATH, dtype={"smiles": str})

    if df["smiles"].isna().any() or df["smiles"].duplicated().any():
        raise ValueError("Missing or repeated SMILES in enhanced feature dataset")
    if split["smiles"].isna().any() or split["smiles"].duplicated().any():
        raise ValueError("Missing or repeated SMILES in frozen split")
    if set(df["smiles"]) != set(split["smiles"]):
        raise ValueError("Enhanced dataset does not match the frozen train/test split")
    if len(df) != 194 or len(split) != 194:
        raise ValueError("Expected 194 original molecules; stop and investigate")

    missing = [col for col in BASE_12 if col not in df.columns]
    if missing:
        raise ValueError(f"Missing base feature columns: {missing}")

    added_rows, fp_rows = [], []
    for idx, smi in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Unparseable SMILES in curated dataset at row {idx}: {smi}")
        if any(atom.GetAtomicNum() != 6 for atom in mol.GetAtoms()):
            raise ValueError(f"Unexpected non-carbon heavy atom in curated dataset: {smi}")
        added_rows.append(graph_features(mol))
        bit_array = np.zeros(N_BITS, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(FP_GENERATOR.GetFingerprint(mol), bit_array)
        fp_rows.append(bit_array)

    extended = pd.concat(
        [df.reset_index(drop=True),
         pd.DataFrame(added_rows),
         pd.DataFrame(np.vstack(fp_rows), columns=FP_COLUMNS)],
        axis=1,
    )
    FEATURES_28 = BASE_12 + ADDED_16
    all_features = FEATURES_28 + FP_COLUMNS
    numeric = extended[all_features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite values in descriptor features")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    extended.to_csv(OUTPUT, index=False)

    # Collision analysis uses training SMILES only. No fitted labels are read.
    train_smiles = set(split.loc[split["split"] == "train", "smiles"])
    train = extended[extended["smiles"].isin(train_smiles)]
    if len(train) != 155:
        raise ValueError(f"Expected 155 train-pool molecules, found {len(train)}")

    print("Data molecules:", len(extended))
    print("Frozen training pool:", len(train))
    print("Final test molecules: 39 (not evaluated)")
    print("Output:", OUTPUT)
    print("Descriptor counts: base=12, richer=28, richer+Morgan=156")
    for label, columns in [
        ("Base 12", BASE_12),
        ("Richer 28", FEATURES_28),
        ("Richer 28 + chirality-aware Morgan128", all_features),
    ]:
        n_colliding = int(train.duplicated(subset=columns, keep=False).sum())
        n_unique = int(train.drop_duplicates(subset=columns).shape[0])
        print(f"{label}: {n_colliding} colliding molecules; {n_unique} distinct vectors")

    print("\nNo targets, train/test assignments, or original CSV files were changed.")


if __name__ == "__main__":
    main()
