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

## November 2, 2026 - Overfitting and Model Selection

Development-training molecules: 116
Development-validation molecules: 39
Reserved final-test molecules: 39 (untouched)

Training vs validation parameter MAE ratios:

Ridge 5:
- m: 1.13x
- sigma: 1.10x
- epsilon/k: 0.95x

Extra Trees 156:
- m: 4.37x
- sigma: 3.00x
- epsilon/k: 3.31x

Interpretation:
Extra Trees shows substantial training-validation gaps,
consistent with overfitting or high model variance.

Despite this, Extra Trees achieves lower validation
parameter MAE than Ridge 5 for all three targets.

Downstream VLE mean APE on 39 development molecules:
- Ridge 5: 16.2203%
- Ridge 12: 17.9553%
- Extra Trees 156: 25.9094%

All three configurations completed 156/156 pressure
comparisons.

Model-selection decision:
Ridge 5 is selected as the primary final VLE model
because it produced the lowest development pressure
error and uses a simpler molecular representation.

Extra Trees 156 remains a comparison model demonstrating
that lower parameter MAE does not guarantee lower VLE
pressure error.

Possible explanations include PC-SAFT sensitivity and
interactions between parameter errors. These mechanisms
have not been isolated experimentally.

Next: refit on 155 training molecules and evaluate the
reserved 39 final-test molecules once.

## November 2, 2026 - Scientific Evaluation

Development data:
- 116 fitting molecules
- 39 validation molecules
- 39 final-test molecules untouched

Saturation-pressure mean absolute percentage error:
- Ridge 5: 16.2203%
- Ridge 12: 17.9553%
- Extra Trees 156: 25.9094%

Saturated-liquid-density mean absolute percentage error:
- Ridge 5: 2.5689%
- Ridge 12: 1.7890%
- Extra Trees 156: 3.8550%

All 156 density and pressure comparisons per model
were successful.

Recomputed pressures agreed with saved values,
with maximum relative discrepancy 1.29e-10.

Scientific interpretation:
Ridge 5 performed best for saturation pressure,
while Ridge 12 performed best for liquid density.
The optimal ML parameter predictor is not necessarily
the optimal downstream thermodynamic predictor.

Local logarithmic pressure sensitivities were
evaluated for five development molecules.

Results:
- Segment number m: about -4.92 to -5.73
- Segment diameter sigma: approximately -3.00
- Dispersion energy epsilon/k: about -6.93 to -10.39

These results suggest that pressure is highly sensitive
to combinations of molecular parameter errors.
Parameter error cancellation or reinforcement is a
possible explanation, pending decomposition analysis.

All thermodynamic accuracy comparisons above are
relative to published-parameter PC-SAFT calculations,
not independent experimental measurements.
