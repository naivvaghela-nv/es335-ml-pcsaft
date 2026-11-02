# Frozen Final Model Decision

Project: ML-Assisted PC-SAFT for Pure-Component VLE of Hydrocarbons

Primary model: Ridge regression

Features:
- molecular_weight
- carbon_count
- ring_count
- rotatable_bonds
- branch_points

Preprocessing:
- SimpleImputer(strategy="median")
- StandardScaler()

Regressor:
- Ridge(alpha=1.0)

Data:
- 194 curated saturated hydrocarbons
- 155 frozen training molecules
- 39 reserved final-test molecules

Selection criterion:
Mean saturation-pressure absolute percentage error
relative to published-parameter PC-SAFT on the
39-molecule development-validation subset.

Development results:
- Ridge 5: 16.2203%
- Ridge 12: 17.9553%
- Extra Trees 156: 25.9094%

Ridge 5 was selected for the primary vapor-pressure
objective based on development performance and simplicity.

Secondary evaluation:
- PC-SAFT parameter MAE and RMSE
- Saturated-liquid density error
- Solver success and failure counts

The final test set must not be used for further
model selection or hyperparameter tuning.
