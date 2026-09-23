# Finite-p solvers

Kadanoff–Baym/Schwinger–Dyson calculations for one- and two-replica correlators.
`contour.py` defines contour metadata and weights; `newton.py` and `components.py`
provide solver components; `chi.py`, `eta.py`, and `replica.py` solve the bath and
eta saddles. `action.py` evaluates the action, `sweeps.py` performs continuation,
and `diagnostics.py` checks numerical outputs.

The standalone entry point is `code/scripts/generate_figure4_data.py`; follow
[data/README.md](../../../data/README.md) for inputs, provenance, and the command.
See the scientific supplement, Secs. S2 and S5, for definitions.
