---
name: bulk-reconstruction
description: Reconstruct Gaussian bulk gates from a fermionic boundary two-point function after validating its anticommutator, conventions, and symmetries. Use for arbitrary-flavor HKLL circuit reconstruction and optional gate-angle or wavepacket plots, including a user-defined flavor projection.
---

# Bulk reconstruction

Convert boundary correlators into the anticommutator Gram matrix, validate it,
and call this project's general-flavor reconstruction engine. Make only the
optional plots requested. Do not assume an evaporation protocol, two flavors,
time-translation invariance, or left/right boundary-copy symmetry.

## Establish the input convention first

Read [input-contract.md](references/input-contract.md) before converting data.
Determine the operator basis, definition and normalization of the two-point
function, time grid, flavor order, and array layout. Infer these from supplied
code/metadata when possible; ask about conventions that remain ambiguous. A
symmetry test cannot identify the physical meaning of an unlabeled matrix.

Use `N` for the number of input operator components per time sample, not a
microscopic large-N parameter. The engine expects time-major ordering
`i = time_index * N + flavor_index`.

For Hermitian generators with `{chi_a(t),chi_b(t)} = delta_ab`:

- `W_ij = <chi_i chi_j>` gives `A = W + W.T = 2 Re W`.
- `Ggreater = -i W` gives `A = i(Ggreater + Ggreater.T) = -2 Im Ggreater`.
- The project's **normalized** `CorM = 2 W` gives `A = Re CorM`.

These identities require the exchanged-time/flavor symmetry and are not
interchangeable normalizations. For complex annihilation operators, require the
anticommutator or **both** greater and lesser functions; taking the real part
alone generally loses information.

Check finiteness, shape, exchange symmetry, equal-time normalization, and positive
semidefiniteness before reconstruction. Report residuals. Symmetrization may
remove roundoff only after validation; do not silently repair wrong signs,
missing blocks, noncanonical normalization or negative eigenvalues. Rank loss
can end the reconstructed geometry: do not add a ridge or invent invalid gates.

## Reconstruct and verify

Use [scripts/bulk_reconstruct.py](scripts/bulk_reconstruct.py). It imports
`code/src/bulk_reconstruction/reconstruction.py` from a checkout of this project,
not the paper's two-flavor plotting wrappers. It discovers the checkout from the
script/current directory, or accepts `--repo-root /path/to/checkout` when installed
elsewhere. Python 3.10+, NumPy, SciPy and Matplotlib are needed. The skill requires
a checkout; it does not bundle a second engine or automatically download code.

```sh
python /path/to/skills/bulk-reconstruction/scripts/bulk_reconstruct.py reconstruct \
  --input input.npz --kind majorana-greater --flavors 3 --output run/
```

`input.npz` contains `times` and `two_point`; see the input contract for layouts
and explicit `--kind` choices. Outputs include gates, masks, the accepted
anticommutator and diagnostics. Use `--save-kernels` to save the kernels too.
The engine always allocates kernels internally: check the memory estimate before
choosing a large grid or flavor count.

Inspect `diagnostics.json`: conversion residuals, eigenvalues, invalid diamonds,
unitarity, whitening errors and tolerances. The helper applies a stricter gate
tolerance than the legacy engine and propagates invalid gates to containing
diamonds. Explain empty/truncated geometry. Finite-grid or ill-conditioned
truncation alone does not establish a physical end-of-the-world brane.

## Optional subskills

Read only the relevant guide:

- [Gate-angle color map](references/gate-angle-map.md): `gate-map`. For `N > 1`,
  retain the existing plotted block norm `||U[:N,:N]||_F/N`. Here “angle” is the
  project's name for that strength map; do not change it into an inverse
  trigonometric angle or a reflection probability.
- [Wavepacket propagation amplitude](references/wavepacket-amplitude.md):
  `wavepacket --quantity amplitude` (or explicitly `intensity`). Specify an
  N-component initial vector, injection indices and mover direction.
- [Wavepacket Z / projection component](references/wavepacket-projection.md):
  `wavepacket --quantity projection --projection P.npy`. Require a Hermitian
  `N x N` matrix in the reconstructed flavor basis. Evaluate `psi^H P psi`,
  including off-diagonal entries. Never silently alternate signs across flavors.

The circuit evolves `[R_in; L_in] -> [L_out; R_out]`. Its L/R labels denote
movers, not input flavors. Retain complex amplitudes in saved data; distinguish
amplitude, intensity and signed projection. Do not divide projected density by
local intensity unless the user requests conditional polarization.

## Handoff

State the input convention, flavor order, normalization, grid, diagnostics and
truncation. For plots, identify the block-norm definition or observable matrix, initial
state, mover and injection point. Save numerical data alongside plots and inspect
their orientation, labels and masked regions. Keep generated runs outside the
skill folder. Do not rewrite manuscript figures or change model parameters merely
to exercise this skill.

All supporting scripts live inside this skill and import the existing engine
read-only. Do not edit existing reconstruction, plotting, or notebook code when
using or maintaining this skill. For its numerical contract checks, run
`python /path/to/skills/bulk-reconstruction/scripts/test_bulk_reconstruct.py`.
