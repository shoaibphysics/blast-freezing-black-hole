# Finite-p data for manuscript Figure 4

This directory contains the corrected, regenerated dataset at the established
publication path `mu_0p50_p_16_dt_0p50_tev_6p00_tmax_25p00_20260625_011517`.
The timestamp suffix is retained for path compatibility. The actual regeneration
date is recorded below. `SHA256SUMS` covers the current snapshots and manifest.

## Parameters recorded in the snapshots

| Parameter | Value |
| --- | --- |
| Interaction scale cal_J | 0.5 |
| Chi coupling mu | 0.5 |
| Interaction order p | 16 |
| Real-time step dt | 0.5 |
| Euclidean points per segment N_beta | 3 |
| Euclidean step dt_euclidean | 0.4779622155734576 |
| Evaporation time t_ev | 6.0 |
| Evaporation parameter a_param | 1.0 |
| Inverse temperature beta | 5.735546586881491 |
| Final times t_f | 0.5 through 25.0, step 0.5 |

Beta is `large_p.analytic.beta_from_mu(cal_J=0.5, mu=0.5)`. The two-replica
eta problem uses this same contour beta. These are the finite-p parameters,
not the independent analytic formation settings used in other figures.

## Contents and use

`two_point_functions/manifest.csv` indexes the 50 snapshots by `t_f` and relative
`file` name. Other columns record parameters, convergence flags, residual norms,
continuation flags, and `N_contour`. All three saddles are marked converged in
each stored row. Every snapshot is needed for the plotted entropy curve; the
last snapshot is reused for the mutual-information heatmap.

Each PyTorch `.pt` dictionary contains:

- `G_chi`, `G_disc`, `G_twisted`: complex128 Green-function matrices for the chi
  bath, disconnected eta saddle, and twisted eta saddle.
- `w`: complex contour quadrature weights.
- `contour`: beta, t_f, dt, N_beta, N_t, N_fold, N_contour, and dt_euclidean.
- `parameters`: the physical parameters listed above.
- `chi_info`, `disc_info`, `twisted_info`: saved convergence and Newton/GMRES
  diagnostics; `used_*_continuation` records initial-guess continuation.

At t_f=25, N_beta=3, N_t=50, N_contour=106; G_chi is 212 x 212 and the
eta matrices are 424 x 424. The contour builder uses N_beta = round(beta/(4 dt))
and dt_euclidean = beta/(4 N_beta) on the imaginary branches. Consequently
beta_grid = 4 N_beta dt_euclidean = beta = 5.735546586881491, up to floating-point
precision. The real-time grid and the beta(mu) relation are unchanged.

Verify the current files from this directory:

```bash
shasum -a 256 -c SHA256SUMS
```

## Numerical generation

Run `code/scripts/generate_figure4_data.py` from the repository root, as explained
in [`data/README.md`](../../../../README.md). It calls
`finite_p.sweeps.solve_two_replica_eta_sweep`, which solves the chi
Schwinger-Dyson equation and then both eta saddles, embedding each preceding
converged solution onto the next contour as an initial guess. The entropy
plot evaluates `finite_p.action.eta_action_difference`; the mutual-information
plot uses the final-time tensors and `application.bulk_mi` reconstruction.

The new generator fixes CPU complex128 arithmetic, chi tolerance 1e-9, eta
tolerance 1e-8, at most 20 Newton iterations, GMRES maximum 220 with relative
tolerance 1e-4, and 18 line-search steps. Antisymmetry and preconditioning are
enabled, with an absolute-value guard of 2. It records these settings and
software versions in each new run. The regenerated published data use the settings
recorded below. The bundled snapshots and this provenance record are the published inputs;
no prior data backup or development repository is needed.

The `source_sha256` entries below identify the code at data-generation time.
They are provenance records, not an inventory of this curated package. Public
packaging updates comments, documentation, and notebook output paths without
changing scientific Python logic.

## Regeneration record

```json
{
  "parameters": {
    "cal_J": 0.5,
    "mu": 0.5,
    "p": 16,
    "dt": 0.5,
    "t_ev": 6.0,
    "a_param": 1.0,
    "beta": 5.735546586881491
  },
  "tfs": {
    "first": 0.5,
    "last": 25.0,
    "step": 0.5,
    "count": 50
  },
  "use_continuation": true,
  "chi_solve_kwargs": {
    "max_newton": 20,
    "gmres_maxiter": 220,
    "line_search_steps": 18,
    "gmres_rtol": 0.0001,
    "gmres_atol": 0.0,
    "max_abs_guard": 2.0,
    "enforce_antisymmetry": true,
    "use_preconditioner": true,
    "verbose": false,
    "device": "cpu",
    "tol": 1e-09
  },
  "replica_solve_kwargs": {
    "max_newton": 20,
    "gmres_maxiter": 220,
    "line_search_steps": 18,
    "gmres_rtol": 0.0001,
    "gmres_atol": 0.0,
    "max_abs_guard": 2.0,
    "enforce_antisymmetry": true,
    "use_preconditioner": true,
    "verbose": false,
    "device": "cpu",
    "tol": 1e-08
  },
  "versions": {
    "python": "3.12.14",
    "numpy": "2.5.3",
    "scipy": "1.18.1",
    "pandas": "3.0.5",
    "torch": "2.14.0"
  },
  "regeneration": {
    "completed_at_utc": "2026-09-16T07:51:33.495219+00:00",
    "generated_run_name": "mu_0p50_p_16_dt_0p50_tev_6p00_tmax_25p00_20260916_073206_027540",
    "generation_seconds": 1156.7688442919898,
    "dt_euclidean": 0.4779622155734576,
    "beta_grid": 5.735546586881491,
    "max_residuals": {
      "chi": 9.825179261652388e-10,
      "disc": 1.1786162530809455e-09,
      "twisted": 9.422125159856245e-09
    },
    "source_sha256": {
      "code/util/paths.py": "d98c9e055a90cb4d62ef1017b587aae9872ec8ae7edc29532ffd3b64c5a0f807",
      "code/util/output_utils.py": "72b3631f39f6c9bbce571f4ec768d5cc673622d8b7e03b0954867ac34da2168a",
      "code/util/plotting.py": "428e0a452475be90f458294e914035a5d3b5258248fd315901da087123d2f5bc",
      "code/util/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "code/tests/test_size_matching.py": "39653d43ea25faf935087670ae7ef02e7e13d7d66adb10695eca2ef7a4a316fb",
      "code/tests/test_paper_data.py": "a15a7505241150ec7e64876ea025e2173f22f0e8b1b119cf0d55d5a7ab4a8c2b",
      "code/tests/test_figure_migration.py": "baf17b40fbe929f578b250a9dc6902ce9b8c0fcd40e43598c16dde98d874b702",
      "code/scripts/generate_figure2.py": "328fd6f5bac398daba0a548ecdf158eec419af664976c329a53b656845d0d311",
      "code/scripts/generate_paper_figures.py": "7a8d6b9eaa8936ed084606711759226093d3ddd19c64bdd87466189c9eb74d9d",
      "code/scripts/generate_figure4.py": "68a59155e4288ce93b77a2a2a9b46024dbe2d7b33095bfc59126673f2ad0151a",
      "code/scripts/generate_figure4_data.py": "deb26e206b47feb36bcf6f5daa061a8afc0f52fb3a3b02de3c6bbe607f3710d2",
      "code/scripts/generate_v1p1_figure3.py": "cb964fb7ccf941d2aa1ac74b006e4dff9c8e030de6af878d3fed9aebf6ace465",
      "code/src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "code/src/large_p/formation_sources.py": "cda80616a14b98a2147d596cf2e0c738ef4569cedea7e0f8e9c389166dd15de4",
      "code/src/large_p/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "code/src/large_p/analytic_sources.py": "884b7cb17d9e455ee97dcc619df70df20e9404ad934f9473dde2bdd688e225a1",
      "code/src/large_p/largep_syk.py": "ddf65fe444a9f54f0ba908ed8c14c02e68a3acdaf3d0d40e81b7d6d3da200199",
      "code/src/large_p/evap_syk.py": "41e6f29018f8102d0528e6ba007a2119440c07792397920d1a77f34cfba14fe1",
      "code/src/large_p/analytic.py": "47597e2658379063b0ae5f06439f4867c8d65eadb995b4f04ce1d8b369397334",
      "code/src/finite_p/diagnostics.py": "0dc276d2122e5d18357aee23b7aaa92d83edb9f231cc911b5dde9db7e44994d5",
      "code/src/finite_p/newton.py": "3a65a22c69431f025d2f6635a65b63e7c8367947c8c706a962733db85c9ed0b8",
      "code/src/finite_p/replica.py": "aec53c5c35647832f38bd34f0add7d820f878f404206de42bb68172fdeb0700a",
      "code/src/finite_p/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "code/src/finite_p/action.py": "30a0a69d324e3924a288c51a55edbfb59d7c30768bc6e66f3e7c2fdf7890e7be",
      "code/src/finite_p/chi.py": "f69f2d93ffdbdf36b0ed34de7578278d38c26f92e28b486694b8403aab4ae7fb",
      "code/src/finite_p/io_utils.py": "a394045f6f63db04296a78f54cbb0b7c39b93d9c3175f0e93989fcf9fcb42d62",
      "code/src/finite_p/eta.py": "2e4b876aa16758c801316ccad92d6a2a824805891aeaf3222cbabf8a07c21bd4",
      "code/src/finite_p/sweeps.py": "b59df5438e0eb94066c521fd191fa58866fc756a91c7ca3c814ce1497423044b",
      "code/src/finite_p/components.py": "e9c1dc2df81c26de208206b092c8eda7375c44a6b96b4f4adef00055ef753c9e",
      "code/src/finite_p/contour.py": "9b4b0fdd01c7d0d4c55b84d34f428a89dc54b4e811f6874232ac4ed8345c5bdd",
      "code/src/bulk_reconstruction/reconstruction.py": "27fd9f341633d7d2db02378f93a2262bc15f800d3002183fcbaa7a4438cc0432",
      "code/src/bulk_reconstruction/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "code/src/bulk_reconstruction/evaporation_circuit.py": "053fdca37db6a209aaa7ed064276c85a16428ea3d5f8f8d5d5408e94408b6681",
      "code/src/application/bulk_mi.py": "f9ae34280f3051d69205a684c698a42867cc317724c466af2468f61512e2482d",
      "code/src/application/infalling_operator_size.py": "2599f94a10870ee3d79177b1fc664aa558333110b457a972f6dadecfe18d2b7f",
      "code/src/application/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    }
  }
}
```
