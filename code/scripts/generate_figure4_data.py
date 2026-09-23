#!/usr/bin/env python3
"""Re-solve the finite-p saddles used in manuscript Figure 4.

Writes a new, ignored run under data/generated; never overwrites published data.
See data/README.md for provenance and the distinction from the saved originals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
for path in (CODE_ROOT, CODE_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import scipy
import torch

from finite_p.diagnostics import solver_summary_dict
from finite_p.io_utils import slug_float
from finite_p.sweeps import solve_two_replica_eta_sweep
from large_p.analytic import beta_from_mu
from util.paths import REPO_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/generated",
                        help="Output root containing numerics/ (default: data/generated)")
    parser.add_argument("--tf-max", type=float, default=25.0,
                        help="Last final time, a positive multiple of 0.5 (default: 25)")
    args = parser.parse_args(argv)
    dt = 0.5
    if not np.isfinite(args.tf_max) or args.tf_max <= 0 or not np.isclose(
        args.tf_max / dt, round(args.tf_max / dt), rtol=0, atol=1e-10
    ):
        parser.error("--tf-max must be a positive multiple of 0.5")
    tfs = dt * np.arange(1, round(args.tf_max / dt) + 1)
    params = dict(cal_J=0.5, mu=0.5, p=16, dt=dt, t_ev=6.0, a_param=1.0,
                  beta=float(beta_from_mu(0.5, 0.5)))
    common = dict(max_newton=20, gmres_maxiter=220, line_search_steps=18,
                  gmres_rtol=1e-4, gmres_atol=0.0, max_abs_guard=2.0,
                  enforce_antisymmetry=True, use_preconditioner=True,
                  verbose=False, device="cpu")
    chi_kwargs = dict(common, tol=1e-9)
    replica_kwargs = dict(common, tol=1e-8)
    prefix = (f"mu_0p50_p_16_dt_0p50_tev_6p00_tmax_{slug_float(args.tf_max, 2)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run = args.data_root.expanduser().resolve() / "numerics/eta_two_replica/runs" / f"{prefix}_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    settings = dict(parameters=params, tfs=tfs.tolist(), use_continuation=True,
                    chi_solve_kwargs=chi_kwargs, replica_solve_kwargs=replica_kwargs,
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  scipy=scipy.__version__, pandas=pd.__version__, torch=torch.__version__))
    (run / "generation.json").write_text(json.dumps(settings, indent=2) + "\n")
    summary, solutions = solve_two_replica_eta_sweep(
        tfs, **params, use_continuation=True, stop_on_failure=True, verbose=True,
        chi_solve_kwargs=chi_kwargs, replica_solve_kwargs=replica_kwargs,
    )
    summary.to_csv(run / "solver_summary.csv", index=False)
    if len(solutions) != len(tfs) or not all(
        sol.chi_info.converged and sol.info_disc.converged and sol.info_twisted.converged
        for sol in solutions
    ):
        raise RuntimeError(f"Sweep did not converge completely; diagnostics in {run}")

    destination = run / "two_point_functions"
    destination.mkdir()
    rows = []
    for sol in solutions:
        filename = f"eta_two_replica_mu_0p50_p_16_dt_0p50_tev_6p00_tf_{slug_float(sol.t_f, 2)}.pt"
        infos = dict(chi_info=solver_summary_dict(sol.chi_info),
                     disc_info=solver_summary_dict(sol.info_disc),
                     twisted_info=solver_summary_dict(sol.info_twisted))
        continuation = {f"used_{name}_continuation": bool(getattr(sol, f"used_{name}_continuation"))
                        for name in ("chi", "disc", "twisted")}
        payload = {name: torch.as_tensor(getattr(sol, name), dtype=torch.complex128)
                   for name in ("G_chi", "G_disc", "G_twisted", "w")}
        payload.update(contour=sol.contour.as_dict(), parameters=params, **infos, **continuation)
        torch.save(payload, destination / filename)
        row = dict(t_f=sol.t_f, **params, **continuation, N_contour=sol.contour.N_contour, file=filename)
        for name in ("chi", "disc", "twisted"):
            row[f"{name}_converged"] = infos[f"{name}_info"]["converged"]
            row[f"{name}_residual_norm"] = infos[f"{name}_info"]["residual_norm"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(destination / "manifest.csv", index=False)
    files = sorted(destination.iterdir()) + [run / "generation.json", run / "solver_summary.csv"]
    (run / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(run)}\n" for path in files
    ))
    print(f"Saved {len(solutions)} converged snapshots to {run}")


if __name__ == "__main__":
    main()
