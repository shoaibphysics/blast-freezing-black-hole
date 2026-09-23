# Optional subskill: gate-angle color map

Preserve the existing project definition exactly:

```
B = U[:N, :N]
color = np.linalg.norm(B, ord="fro") / N
```

The user calls this the gate “angle” map; its numerical quantity is the block
norm used by `bulk_norm_grid` and `plot_gate_color_map` in
`code/scripts/generate_paper_figures.py`. Do not replace it with an arcsine,
singular-channel angle, squared norm, or normalization by `sqrt(N)`.
For `[R_in;L_in] -> [L_out;R_out]`, B is the R-to-L reflection block. The color
scale is a gate strength, not an angle in radians or a reflection probability.

```sh
python /path/to/skills/bulk-reconstruction/scripts/bulk_reconstruct.py gate-map \
  --reconstruction run/reconstruction.npz --output run/gate-map.png
```

This same definition applies to any positive flavor count N, with no choice of
preferred flavors. For a unitary gate its range is `[0,1/sqrt(N)]`; do not
renormalize different N values to make their maximum one. Save both the raw
block norm and the displayed value. The helper uses the existing viridis color
map and data-driven color limits, with a colorbar labeled `||U_11||_F/N`.

Draw valid diamonds only in `(z,t)` coordinates, leaving missing geometry blank.
The helper builds half-step diamond cells from physical neighboring times,
including nonuniform grids, and exports indices, coordinates and norms alongside
the image. Protocol-specific lines and region labels require protocol metadata;
do not automatically add evaporation lines to an unrelated correlator.
