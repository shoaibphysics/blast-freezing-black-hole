# Optional subskill: wavepacket Z / projection component

Follow [wavepacket-amplitude.md](wavepacket-amplitude.md) for propagation. Require
a finite Hermitian `N x N` matrix P in the same reconstructed flavor frame as
the amplitudes; do not infer it from N.

```sh
python /path/to/skills/bulk-reconstruction/scripts/bulk_reconstruct.py wavepacket \
  --reconstruction run/reconstruction.npz --initial-state psi0.npy \
  --start-u 0 --start-v 1 --mover R --quantity projection \
  --projection P.npy --output run/projection.png
```

Plot `q=psi^H P psi` on each R/L link, using full matrix multiplication or
`einsum`. `diag(P) dot abs(psi)**2` loses off-diagonal coherence. Check that the
imaginary residue is roundoff. Save P and its spectrum with propagated vectors.

“Projection” means the selected observable; it need not be idempotent:

- Two-flavor Z: `diag(1,-1)` gives `|psi_0|^2-|psi_1|^2`.
- Three-flavor contrast: `diag(1,0,-1)`.
- Population in normalized direction w: `P=w w^H`, a true projector.
- Off-diagonal Hermitian P measures coherence. Complex Hermitian entries are
  supported; a non-Hermitian matrix is rejected for this real-valued plot.

Default is **unnormalized projected density**, including local packet weight,
as in the existing Z plot. Conditional polarization `psi^H P psi/(psi^H psi)`
is a different requested task; mask zero weight and label normalization if used.

For a unit-norm source, bounds are
`[min(0,lambda_min(P)),max(0,lambda_max(P))]`, including zero at low weight.
Use a zero-centered diverging scale for mixed-sign spectra and a sequential
scale for positive projectors. Do not assume eigenvalues in `[-1,1]`. Identify P
in the handoff, and label this as Z only when that is the supplied observable.
