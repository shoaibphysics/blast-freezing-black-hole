# Data for the Nature manuscript

The published numerical input is one [finite-p run](numerics/eta_two_replica/runs/mu_0p50_p_16_dt_0p50_tev_6p00_tmax_25p00_20260625_011517/README.md):
50 PyTorch snapshots at final times 0.5, 1.0, ..., 25.0, approximately 112 MiB.
The run's `two_point_functions/manifest.csv` indexes the snapshots. Its
`SHA256SUMS` covers all 50 snapshots and that manifest. All inputs are bundled;
no external dataset or private directory is needed.

Figure 4(b) uses all 50 snapshots for the entropy curve. Figure 4(a) uses the
final-time 25 snapshot for bulk mutual information. Figures 2, 3, and 5 use
analytic correlators computed by the maintained code; they need no additional
saved runs. Figure 1 uses supplied artwork in `paper/`; the Penrose panel inputs
are in `code/scripts/figures/`.

The historical run-directory suffix is retained for path compatibility. The
actual saved-data regeneration date, solver settings, tensor schema, and source
hashes are recorded in the run README. They describe dataset provenance, not
an additional dataset or a dependency on development Git history.

## Inspect and check the bundled data

After [environment setup](../environment/README.md), from the repository root:

```sh
.venv/bin/python code/figure-reproduction/check_claims.py
```

This verifies 51 checksums, inspects all 50 payloads and convergence flags, and
recomputes the plotted entropy and mutual information. It also checks the analytic
cross-time rank and selected size values. Exact expected signatures and evidence
limits are listed in [Quick claim checks](../code/figure-reproduction/README.md#quick-claim-checks).

## Reproduce figures or generate a fresh dataset

```sh
.venv/bin/python code/figure-reproduction/reproduce_figures.py
```

The [maintained notebook](../code/notebooks/paper_figures.ipynb) uses the same code.
A full new numerical sweep takes about 20 minutes on the tested CPU:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python code/scripts/generate_figure4_data.py
BH_DATA_ROOT="$PWD/data/generated" .venv/bin/python code/figure-reproduction/reproduce_figures.py
```

New runs go under `data/generated/`, are ignored by Git, and do not replace the
published snapshots. The loader honors an explicit `BH_DATA_ROOT` and does not
fall back to other datasets. No optional p=4 experimental dataset is included.
The Git ignore file allowlists only the approved data files, including the 50
snapshots, manifest, checksums, and READMEs.
