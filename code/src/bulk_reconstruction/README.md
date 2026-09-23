# Bulk reconstruction

`reconstruction.py` builds bulk kernels and Gaussian gates from the boundary
anticommutator and supports wavepacket propagation. `evaporation_circuit.py`
provides auxiliary evaporation-circuit routines; the manuscript's fixed figure
workflow uses `reconstruction.py` directly.

Use the bundled [bulk-reconstruction skill](../../../skills/bulk-reconstruction/SKILL.md)
for input normalization, component ordering, diagnostics, and general-flavor use.
For the paper's fixed presets, use the [figure guide](../../figure-reproduction/README.md).
