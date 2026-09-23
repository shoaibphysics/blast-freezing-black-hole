# Fresh paper-agent smoke test

Date: 2026-09-23 UTC. **Result: passed for bounded reader usability and cached
numerical checks.** A fresh reader session started in a separate isolated copy
of the publication package, read `AGENTS.md` first, and used only that package's
scientific files and guides. It did not consult development history or preparation
conversations. The existing tested Python 3.12.14 environment was reused; no new
installation, download, notebook execution, or full nonlinear sweep was performed.

The reader reviewed all 13 READMEs, AGENTS/CLAUDE, the bulk-reconstruction skill,
license scopes, reuse notes, relevant paper/supplement passages, and figure/output
entry points. It actually ran the following command from the isolated package root
with one OpenBLAS/OMP thread:

```sh
.venv/bin/python code/figure-reproduction/check_claims.py
```

The command exited 0 and wrote diagnostics only under
`code/figure-reproduction/generated/claims/`. This was a cached-data audit,
not a fresh solve, a proof of all physics claims, or a public release verification.

## Representative reader questions

**What is scientific ground truth?**
The Nature main manuscript and scientific supplement, their PDFs, and bundled
code/data. Root `supplementary/` reports are secondary APP context. The reader
correctly described the coupled-SYK evaporation model, stated large-N/large-p
limits, bulk reconstruction, operator size, mutual information, and second-Renyi
entropy; it distinguished analytic limits from finite-p numerical evidence.

**How close does Figure 4(a) get to its maximum?**
The reader used the actual cached checker, whose MI route is
`compute_renyi2_mi_heatmaps` and `application/bulk_mi.py`, anchored to main Figure
4(a), Eq. `eq:renyi2_MI`, and Supplement S5. It observed:

| Measurement | Observed result |
|---|---:|
| Maximum MI | 1.3837480740067698 nats |
| Theoretical maximum, 2 ln 2 | 1.3862943611198906 nats |
| Fraction of maximum | 0.9981632421046106 (99.81632421046106%) |
| Grid and finite/masked cells | 50 × 50; 1055 finite / 1445 masked |
| Dataset hashes | 51/51 matched |
| Finite payloads | 50/50 |
| Saved convergence flags | 150/150 true |
| Maximum beta-grid / antisymmetry discrepancy | 0 / 0 |

The reader identified the final-time-25 snapshot, first right-moving component,
and kernel-norm cutoff 50. One cached map does not establish an MI time sweep,
exact saturation, or all parameter regions.

**What does Figure 4(b)'s entropy show?**
All 50 samples were finite. Pre-evaporation samples were zero; the maximum entropy
density was 0.18260681636205683 at t=18. Later sampled minima were
0.002668085958073857 at t=14 and 0.008122848280091466 at t=22. The reader traced
this to `compute_renyi2_action_curve`, `finite_p/action.py`, and Supplement S5.
It correctly distinguished sampled minima from exact analytic zeros. Both
analytic-grid cross-time rank checks also returned 4 at threshold 1e-10.

**Which code implements Figure 3 and Supplement S4?**
The canonical size-response functions are `compute_size_operator_DeltaNeta_grid`
and `compute_size_operator_DeltaNchi_grid` in `code/src/large_p/evap_syk.py`.
The manuscript plotting driver combines these responses with the reconstruction
kernels. `application/infalling_operator_size.py` is an auxiliary circuit helper.
The refreshed guide and module READMEs make this distinction explicit.

**Are figures, setup, and output paths understandable?**
Yes. The reader correctly mapped historical filenames to all five current main
figures and the two supplementary TikZ diagrams. It distinguished the final
Figure 2 composite from its three inputs, and manual Figure 1/Penrose artwork
from computational panels. It identified the canonical figure wrapper, separate
publisher-support download, isolated PDF-build wrapper, and separate generated
locations. It understood that direct `paper/build.py` would replace PDFs in its
own directory and should not be used when preserving supplied assets.

**Are licenses and evidence limits clear?**
Yes. The reader found consistent CC BY 4.0 scope for paper/supplement/artwork,
MIT for code/skills/data/supporting documentation, and separate publisher terms.
It recognized the image exception under `code/` and build-helper exception under
`paper/`. It also recognized the retained norm/probability, polarization/color,
exact-entropy, manual-artwork, finite-grid, and prior fresh-solve limitations.

## Warnings and outcome

Existing docstring-escape warnings, NumPy `slogdet` warnings, and nonfatal font-cache
messages occurred. Reported cached results were finite and matched the documented
signatures. This reader did not repeat the earlier independent LU audit or full
figure/PDF comparisons. No source edits or new scientific claims were made.

No new blocking guidance contradiction was found. The preparation agent separately
verified source preservation, executable-AST equality, complete local links,
reference resolution, and the final file inventory. Full reproduction evidence
and release limits are summarized in [validation-report.md](validation-report.md).
