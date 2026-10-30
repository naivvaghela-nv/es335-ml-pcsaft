"""Plot one successful training-only validation VLE comparison."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
input_file = ROOT / 'results/predictions/development_vle_curves.csv'
output_file = ROOT / 'results/figures/development_vle_example.png'

df = pd.read_csv(input_file)
valid = df.dropna(subset=['temperature_K', 'pressure_reference_bar', 'pressure_ml_bar'])
counts = valid.groupby('smiles').size()
eligible = counts[counts >= 2]
if eligible.empty:
    raise RuntimeError('No molecule has two comparable pressure points; inspect failure statuses first.')

smiles = eligible.index[0]
group = valid.loc[valid['smiles'] == smiles].sort_values('temperature_K')
name = str(group.iloc[0]['name'])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(group['temperature_K'], group['pressure_reference_bar'], 'o-', label='Published parameters')
ax.plot(group['temperature_K'], group['pressure_ml_bar'], 's--', label='Ridge-predicted parameters')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Saturation pressure (bar)')
ax.set_title(f'{name}: development validation')
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
output_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_file, dpi=160)
plt.close(fig)
print('Saved:', output_file)
print('Molecule:', name)
