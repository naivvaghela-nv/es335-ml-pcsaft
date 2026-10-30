import pandas as pd

df = pd.read_csv("data/processed/hydrocarbons_initial.csv")
split = pd.read_csv("results/predictions/molecule_split.csv")

df = df.merge(split, on="smiles", validate="one_to_one")

# Inspect training molecules only.
train = df[df["split"] == "train"].copy()

features = [
    "molecular_weight",
    "carbon_count",
    "ring_count",
    "rotatable_bonds",
    "branch_points",
]

duplicates = train[
    train.duplicated(subset=features, keep=False)
].sort_values(features)

print("Training molecules:", len(train))
print("Molecules sharing feature vectors:", len(duplicates))
print(
    "Unique feature combinations:",
    len(train.drop_duplicates(subset=features))
)

print("\nExamples of indistinguishable molecules:")
print(
    duplicates[
        ["name"] + features + ["m", "sigma", "epsilon_k"]
    ].head(20).to_string(index=False)
)
