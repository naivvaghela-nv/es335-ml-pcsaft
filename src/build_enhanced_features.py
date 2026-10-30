from pathlib import Path
import math

import pandas as pd
from rdkit import Chem
from rdkit.Chem import GraphDescriptors

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/hydrocarbons_initial.csv"
OUTPUT = ROOT / "data/processed/hydrocarbons_enhanced.csv"
SPLIT = ROOT / "results/predictions/molecule_split.csv"

ORIGINAL_FEATURES = [
    "molecular_weight",
    "carbon_count",
    "ring_count",
    "rotatable_bonds",
    "branch_points",
]

NEW_DESCRIPTORS = {
    "BertzCT": GraphDescriptors.BertzCT,
    "Chi0v": GraphDescriptors.Chi0v,
    "Chi1v": GraphDescriptors.Chi1v,
    "Chi2v": GraphDescriptors.Chi2v,
    "Kappa1": GraphDescriptors.Kappa1,
    "Kappa2": GraphDescriptors.Kappa2,
    "BalabanJ": GraphDescriptors.BalabanJ,
}

df = pd.read_csv(INPUT)
split = pd.read_csv(SPLIT)

if df["smiles"].duplicated().any():
    raise ValueError("Duplicate SMILES found in dataset.")

if split["smiles"].duplicated().any():
    raise ValueError("Duplicate SMILES found in split.")

if set(df["smiles"]) != set(split["smiles"]):
    raise ValueError("Saved train/test split does not match dataset.")

print("Original molecules:", len(df))

# Generate molecular descriptors only from SMILES.
for name in NEW_DESCRIPTORS:
    df[name] = float("nan")

for idx, smiles in df["smiles"].items():
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES at row {idx}: {smiles}"
        )

    for name, function in NEW_DESCRIPTORS.items():
        value = float(function(mol))

        if not math.isfinite(value):
            raise ValueError(
                f"Invalid {name} for molecule {smiles}"
            )

        df.at[idx, name] = value

ALL_FEATURES = (
    ORIGINAL_FEATURES + list(NEW_DESCRIPTORS)
)

# Verify data integrity.
if len(df) != len(split):
    raise ValueError("Unexpected molecule count change.")

if df[ALL_FEATURES].isna().any().any():
    raise ValueError("Missing descriptor values found.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nEnhanced dataset saved:")
print(OUTPUT)

print("\nTotal features:", len(ALL_FEATURES))
print("New descriptors:", list(NEW_DESCRIPTORS))

# Compare collisions on training molecules only.
train_smiles = set(
    split.loc[split["split"] == "train", "smiles"]
)

train = df[df["smiles"].isin(train_smiles)].copy()

def show_collisions(columns, label):
    count = train.duplicated(
        subset=columns,
        keep=False
    ).sum()

    unique = len(
        train.drop_duplicates(subset=columns)
    )

    print(f"\n{label}")
    print("Molecules in collision groups:", count)
    print("Unique feature vectors:", unique)

show_collisions(
    ORIGINAL_FEATURES,
    "Original 5 descriptors"
)

show_collisions(
    ALL_FEATURES,
    "Enhanced 12 descriptors"
)

print("\nSample enhanced features:")
print(
    df[
        ["name", "BertzCT", "Chi0v", "Kappa1", "BalabanJ"]
    ].head(8).to_string(index=False)
)

print("\nOriginal dataset was not modified.")
print("Saved train/test split was not modified.")
