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
