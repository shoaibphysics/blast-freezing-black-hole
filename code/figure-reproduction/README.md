# Figure reproduction and quick claim checks

All commands below run from this package's root with the interpreter from
[environment/README.md](../../environment/README.md). There are five main figures,
two supplementary diagrams, and no numbered tables. Historical asset filenames
are retained so that no numerical implementation or manuscript reference changes.

## Commands and output locations

```sh
.venv/bin/python code/figure-reproduction/reproduce_figures.py
.venv/bin/python code/figure-reproduction/check_claims.py
.venv/bin/python code/figure-reproduction/check_geometry.py
.venv/bin/python code/figure-reproduction/fetch_tex_support.py
.venv/bin/python code/figure-reproduction/build_paper.py
.venv/bin/python code/figure-reproduction/run_notebook.py
```

`reproduce_figures.py` is a grouped wrapper for all numerical panels of main
Figures 2–5; the table below lists every covered artifact. `build_paper.py`
groups the main document and both supplementary diagrams.

`fetch_tex_support.py` is a one-time network setup step only for rebuilding the
PDFs. It verifies and obtains the exact two publisher files; it generates no
scientific figure. See [paper/README.md](../../paper/README.md).

The common prefix `code/figure-reproduction/generated/` is ignored. Outputs are
in `figures/`, `claims/`, `geometry-check/`, `paper-build/paper/`, and `notebook/`
for the five reproduction/check commands. The setup download instead writes
the two ignored support files into `paper/`. The paper build refuses to replace an existing generated build;
move that generated folder aside before repeating. No wrapper updates supplied
paper PDFs, scientific sources, dataset files, or figure assets.

| Figure | Paper artifact | Script | Inputs | Generated output under the common prefix | Status | Notes |
|---|---|---|---|---|---|---|
| 1: setup | `paper/fig1.png` | `build_paper.py` embeds it | Supplied image | `paper-build/paper/manuscript.pdf` | manual-only | Final artwork supplied; no generator or confirmed editable source. |
| 2(a,b,d,e): numerical panels | `paper/figure2_bulk_geometry.pdf` | `reproduce_figures.py`; independent `check_geometry.py` | Analytic correlators, reconstruction engine, three `code/scripts/figures/` images | `figures/figure2_bulk_geometry.pdf`; `geometry-check/geometry-audit.json` | reproduced | Normal command reuses the saved formation map; independent replot checks it. Caption/observable discrepancies below remain documented. |
| 2(c,f): Penrose diagrams | Same composite | `reproduce_figures.py` assembles panels | `code/scripts/figures/fig3b.png`, `fig3d.png` | `figures/figure2_bulk_geometry.pdf` | manual-only | Original artwork confirmed by the author; assembled from supplied images, not computationally derived. |
| 3: operator size | `paper/fig4_eta_size_right_heatmap.png`, `paper/fig4_right_mover_fixed_v_cut.png` | `reproduce_figures.py`, `check_claims.py` | Analytic response and bulk kernels | `figures/fig4_eta_size_right_heatmap.png`, `figures/fig4_right_mover_fixed_v_cut.png`; `claims/size-v8.csv` | reproduced | Uses the preserved size cutoff 100 and fixed-v cut at 8. |
| 4: finite-p MI and entropy | `paper/figure4_finite_p.pdf` | `reproduce_figures.py`, `check_claims.py` | All 50 bundled snapshots for entropy; final-time 25 for MI | `figures/figure4_finite_p.pdf`; `claims/numeric-audit.json`, `claims/cached-entropy.csv` | reproduced | Cached reproduction; the independent fresh sweep has small numerical drift below. |
| 5: two-point function | `paper/fig2_two_point_heatmap.png` | `reproduce_figures.py` | Analytic correlators | `figures/fig2_two_point_heatmap.png` | reproduced | Uses the size/two-point preset, not the geometry preset. |
| S1, S2: contour diagrams | TikZ in `paper/supplementary.tex` | `build_paper.py` | Bundled TeX sources and TeX installation | `paper-build/paper/supplementary.pdf` | reproduced | Document compilation reproduces these source-defined diagrams. |

Statuses refer to the named artifact and check, not independent proof of all
physical interpretations. The [validation report](../../supplementary/validation-report.md)
records pre-staging and package-check evidence separately. Figures from saved data
are not mislabeled as new solver results.

## Quick claim checks

Run `check_claims.py` for the first four checks. It writes `claims/numeric-audit.json`
and diagnostic CSV/NPZ files. This inspects the full bundled dataset, not a selected
favorable snapshot. `check_geometry.py` supplies the fifth check.

| Reader question and paper anchor | Direct staged evidence | Expected observation | Evidence level and limit |
|---|---|---|---|
| How close does MI get to its maximum? Main Fig. 4(a), Eq. `eq:renyi2_MI`, Supplement S5 | `code/scripts/generate_paper_figures.py:compute_renyi2_mi_heatmaps`, `code/src/application/bulk_mi.py`, final-time 25 snapshot | Maximum about 1.383748074; `2 log(2)` about 1.386294361; ratio about 0.998163 (99.8163%); 1055 finite cells | Recomputed from cached tensors, first right-moving component, kernel-norm cutoff 50. One final-time map does not establish every parameter region or an MI time sweep. |
| What does the entropy curve show? Main Fig. 4(b), Supplement S5 | `compute_renyi2_action_curve`, `code/src/finite_p/action.py`, all 50 snapshots | Pre-evaporation sampled entropy density is zero; later sampled minima at t=14 and 22 are about 0.002668086 and 0.008122848; maximum about 0.182606816 | Cached action audit. Sampled nonzero minima do not rule out a zero between samples or establish exact vanishing in an analytic limit. |
| Can the cross-time rank claim be checked? Main solvable-limit discussion, Eq. `eq:regionII`, Supplement S3 | `code/src/large_p/evap_syk.py`; both analytic presets | Numerical rank 4 at relative singular-value threshold 1e-10; fifth singular value around 1e-15 | Direct numerical check of analytic-grid correlators, not a substitute for the factorization argument. |
| What are concrete size anchors? Main Fig. 3, Supplement S4 | `compute_size_operator_DeltaNeta_grid`, chi counterpart, reconstruction kernels | At u=0, v=8, measurement time 25: eta about 505.164494 and chi about -18.270403 | Raw numerical anchors. Display cutoff is 100; large values near ill-conditioning are not newly certified physical extremes. |
| What does the geometry color represent? Main Fig. 2 | `check_geometry.py`; `bulk_norm_grid`, `wavepacket_observable_grids` | Gate color is `||U11||_F/2`; late-injection polarization minimum about -0.154891; direct formation PNG matches supplied map | Numerical observable check. Signed local polarization is not integrated left-to-right transmission. |

The claim checker also verifies 51/51 checksums, 50/50 finite payloads, all
150 saved convergence flags, beta-grid consistency, and antisymmetry. Its S6 disk
identity check samples the stated formula; it is not a proof of the full theory.

## Parameters and implementation map

- Geometry: p=16, dt=0.5, mu=0.1, evaporation time 15, final time 80.
- Size/two-point: p=16, dt=0.5, mu=0.5, evaporation time 6, final time 25; fixed v=8.
- Finite-p: p=16, dt=0.5, mu=0.5, evaporation time 6, beta=5.735546586881491.
- Analytic `formation_beta=10` corresponds to physical beta_eta=20 at J=0.5.
  It is not the finite-p contour temperature.

Supplement S1 defines the model; S2 maps to `code/src/finite_p/contour.py`,
`chi.py`, `eta.py`, and `components.py`; S3 to `code/src/large_p/analytic.py`,
`analytic_sources.py`, `evap_syk.py`, and `formation_sources.py`. For S4 and
main Figure 3, the canonical size responses are
`compute_size_operator_DeltaNeta_grid` and `compute_size_operator_DeltaNchi_grid`
in `code/src/large_p/evap_syk.py`; `code/scripts/generate_paper_figures.py`
combines them with the reconstruction kernels. The separate
`code/src/application/infalling_operator_size.py` provides auxiliary
circuit-based routines, not the manuscript's figure entry point. S5 maps to the
replica/action/sweep modules and `bulk_mi.py`; S6 to the large-p conventions and
analytic-bound derivations. The paper is authoritative for formal derivations;
implemented checks cover specified identities only.

## Known limitations retained with the paper

The authors retain the manuscript wording. Figure 2's caption calls a gate-block
norm a reflection probability, and its discussion includes a literal no-color-change
statement despite negative plotted polarization values. The manuscript's exact
entropy-vanishing language is broader than the finite-p sampled evidence. These
are documented interpretation differences, not silently corrected calculations.
They do not by themselves measure boundary transmission or refute the separately
stated supplementary correlator bound.

The pre-staging fresh 50-time sweep converged all 150 saddles. All input tensors
passed rtol=1e-7, atol=1e-8 against the bundled data, but applying the same tight
comparison to derived grids gave `runs-but-differs`: maximum Iq difference
1.22e-7 and kernel-norm difference 3.33e-4 (the largest kernel difference is beyond
the plotting cutoff). Finite masks did not change, and the Figure 4 PNGs were
pixel-identical. No tolerance was changed to hide a failed comparison.
NumPy determinant warnings occurred despite finite results; an independent LU
cross-check agreed to roughly 1e-14. Platform changes may alter rounding.

Manual drawings are author-confirmed original artwork; separate editable drawing
sources are not included.
The full analytical claims are not independently proved by these scripts.

## Runtime and fresh solves

Cached checks, geometry regeneration, figure generation, and notebook execution
range from seconds to a few minutes on the tested CPU. A new full finite-p sweep
took about 20 minutes before staging. See [data/README.md](../../data/README.md)
for the command and provenance. Its `--tf-max 1` mode is a solver smoke test only.
Use a separate generated dataset; never overwrite published snapshots.
