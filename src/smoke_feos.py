import feos
import si_units as si

# Read published parameters for pure hexane.
parameters = feos.Parameters.from_json(
    ["hexane"],
    pure_path="data/raw/esper2023.json"
)

# Build the PC-SAFT equation of state.
eos = feos.EquationOfState.pcsaft(parameters)

# Calculate vapor-liquid equilibrium at 300 K.
vle = feos.PhaseEquilibrium.pure(
    eos,
    temperature_or_pressure=300.0 * si.KELVIN
)

# Both phases have the same equilibrium pressure.
pressure_bar = vle.liquid.pressure() / si.BAR

print("Molecule: hexane")
print("Temperature: 300 K")
print(f"Saturation pressure: {pressure_bar:.5f} bar")
print("PC-SAFT equilibrium calculation successful.")
