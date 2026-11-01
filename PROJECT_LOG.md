# Project Development Log

## 2026-10-20 - Project Initialization

Completed:
- Initialized Git repository.
- Created project folder structure.
- Added project documentation.
- Defined initial ML and thermodynamic scope.

## Next Tasks

1. Obtain the published PC-SAFT parameter dataset.
2. Verify the number of compatible hydrocarbons.
3. Test FeOs using known substance parameters.
4. Generate RDKit descriptors.
5. Train the baseline model.

## Current Status

No experimental results available yet.

## Milestone 2 - Initial ML Baselines

- Established a fixed molecule-wise train/test split.
- Compared mean prediction with Ridge regression.
- Used three-fold cross-validation on training molecules.
- Reserved the final test set for later evaluation.
- Saved model configuration and validation metrics.

Next:
- Review baseline errors.
- Finalize model selection.
- Connect predicted parameters to FeOs.

## November 2, 2026 - Development VLE Model Comparison

Three ML configurations were compared on the same 12
development-validation molecules at four temperatures each.

Results:
- Ridge + 5 descriptors: mean pressure APE 11.1526%
- Ridge + 12 descriptors: mean pressure APE 20.4380%
- Extra Trees + 156 features: mean pressure APE 25.6552%

All models achieved 48/48 comparable saturation-pressure
calculations. The original Ridge baseline reproduction passed.

Key finding:
Better PC-SAFT parameter prediction in training-only
cross-validation did not translate into lower saturation-
pressure error on this development-validation subset.

Possible explanations include parameter sensitivity,
interactions between parameter errors, and differences
in molecular generalization. These mechanisms have not
yet been independently established.

Next:
Inspect per-molecule errors and perform broader development
validation before selecting the final model.

The frozen final test set remains unused.
