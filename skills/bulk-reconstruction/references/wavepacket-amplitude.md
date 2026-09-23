# Optional subskill: wavepacket propagation amplitude

Require a complex N-component initial vector (a `.npy` file for the helper),
injection **indices** `(u,v)` and mover `R` or `L`. Its norm must be one within
tolerance; do not silently normalize it. A physical `(t,z)` corresponds to
`times[u]=t-z`, `times[v]=t+z`; report any grid rounding explicitly.

```sh
python /path/to/skills/bulk-reconstruction/scripts/bulk_reconstruct.py wavepacket \
  --reconstruction run/reconstruction.npz --initial-state psi0.npy \
  --start-u 0 --start-v 1 --mover R --quantity amplitude \
  --output run/amplitude.png
```

The seed is an output link in the backend's reconstructed flavor frame, not a
physical boundary-flavor vector unless the identification has been established.
Bulk injection requires a valid gate. Boundary injection at `u=v` is R only;
the simulator handles L-to-R reflection. Mover labels are not flavor names.

Evolution runs forwards in `u+v`:
`[R(u,v-1);L(u-1,v)] --U(u,v)--> [L(u,v);R(u,v)]`.
An absent interior gate stops propagation; it does not impose a reflecting
end-of-geometry boundary. Explain truncation rather than claiming absorption or
inventing gates. Unreached links stay masked; known zero outputs stay zero.

Default color is amplitude magnitude `sqrt(psi^H psi)` summed in quadrature
over flavors. `--quantity intensity` plots `psi^H psi`. Neither is a complex
single-component amplitude. For phase/component requests, use saved complex
vectors and explicitly select Re, Im, magnitude or phase.

The helper draws outgoing links separately: R goes `(u,v)->(u,v+1)` and L goes
`(u,v)->(u+1,v)`. Physical times determine endpoints. Terminal links with no next
sample remain in the data but are omitted from the image; no grid extrapolation
is invented. The `.npz` alongside the image saves R/L vectors, intensities,
observable arrays, times, mover, initial vector and injection indices.

All retained gates are checked for unitarity before propagation. Test norm
conservation on a complete circuit cut including boundary/terminal links, not
by summing all spacetime links (which counts the same packet repeatedly).
For irregular grids, equal `u+v` is a topological layer, not equal physical time.
