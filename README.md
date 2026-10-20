# ML-Assisted PC-SAFT for Pure-Component VLE of Hydrocarbons

**Course:** ES 335 - Machine Learning

**Institution:** IIT Gandhinagar

## Research Question

Can molecular descriptors estimate PC-SAFT parameters
for simple hydrocarbons, and how do prediction errors
affect pure-component saturation-pressure calculations?

## Project Objectives

1. Collect published PC-SAFT parameter data.
2. Select compatible simple hydrocarbons.
3. Generate molecular descriptors using RDKit.
4. Implement a mean-prediction baseline.
5. Train and evaluate Ridge regression.
6. Compare predicted and published PC-SAFT parameters.
7. Calculate pure-component saturation pressure using FeOs.
8. Build an interactive results website using GitHub Pages.

## Planned ML Methods

- DummyRegressor (baseline)
- Ridge Regression
- Random Forest (optional extension)

## Thermodynamic Model

PC-SAFT Equation of State

Predicted parameters:
- Segment number (m)
- Segment diameter (sigma)
- Dispersion energy (epsilon/k)

Main thermodynamic output:
- Saturation pressure as a function of temperature

## Initial Scope

Simple non-associating hydrocarbons.

Mixtures, complex associating compounds and neural
networks are outside the initial scope.

## Data Sources

Primary candidate:
Esper et al. (2023), PCP-SAFT parameter collection.

https://github.com/feos-org/feos

The eligible molecule count has not yet been verified.

## Repository Structure

- data/ : Raw, processed and external data
- notebooks/ : Exploratory analysis and experiments
- src/ : Reusable Python code
- tests/ : Data and software checks
- results/ : Predictions, metrics and figures
- docs/ : GitHub Pages website
- references/ : Literature notes

## Project Status

Initialization stage. No trained models or validated
thermodynamic results are available yet.

## Reproducibility

Instructions, package versions and experimental
settings will be documented as implementation proceeds.

## Academic Integrity

External datasets, software, literature and
AI-assisted work will be appropriately disclosed.
