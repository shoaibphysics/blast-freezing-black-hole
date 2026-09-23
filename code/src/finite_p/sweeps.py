"""Sweep and continuation helpers.

This module manages repeated solves over a list of final times.  The individual
physics solvers live elsewhere, for example in chi.py.  The role of this module
is to organize the workflow:

    choose grid-aligned t_f values
    optionally embed the previous solution
    solve the next contour problem
    store every Green's function and convergence summary

For large final times, Newton iteration is much more stable when the solution
at a nearby earlier t_f is embedded onto the new contour and used as the
initial guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .chi import solve_chi
from .eta import solve_eta
from .components import extract_component_t0
from .contour import ContourMeta, embed_block_matrix, make_contour_weights, to_numpy
from .diagnostics import antisymmetry_error, solver_summary_dict


from .replica import solve_two_replica_eta_pair, pair_summary_row



@dataclass(frozen=True)
class SweepSolution:
    """One stored solution from a final-time sweep."""

    t_f: float
    G: np.ndarray
    w: np.ndarray
    info: Any
    used_continuation: bool

    @property
    def contour(self) -> ContourMeta:
        """Return the contour metadata associated with this solution."""

        return self.info.contour


def grid_index(value: float, dt: float, *, tol: float = 1e-10) -> int:
    """Return the integer n such that value = n dt.

    Raises ValueError if value is not aligned with the contour grid.
    """

    if dt <= 0:
        raise ValueError("dt must be positive")

    n = int(round(float(value) / float(dt)))
    snapped = n * float(dt)

    scale = max(1.0, abs(float(value)), abs(snapped))
    if abs(float(value) - snapped) > tol * scale:
        raise ValueError(
            f"value={value} is not aligned with dt={dt}; nearest grid value is {snapped}"
        )

    return n


def is_grid_aligned(value: float, dt: float, *, tol: float = 1e-10) -> bool:
    """Return True if value is an integer multiple of dt."""

    try:
        grid_index(value, dt, tol=tol)
    except ValueError:
        return False

    return True


def snap_to_grid(value: float, dt: float, *, ndigits: int = 12) -> float:
    """Return the nearest grid-aligned value n dt."""

    if dt <= 0:
        raise ValueError("dt must be positive")

    n = int(round(float(value) / float(dt)))
    return float(np.round(n * float(dt), ndigits))


def validate_tfs(
    tfs,
    *,
    dt: float,
    require_increasing: bool = True,
    min_delta_N_t: int = 1,
    tol: float = 1e-10,
) -> pd.DataFrame:
    """Validate a list of final times against the contour grid.

    The contour uses N_t = round(t_f / dt), so the safest sweep values obey

        t_f = N_t dt.

    This function checks that condition and returns a table with the grid
    integer N_t and the step sizes between consecutive final times.
    """

    tfs = np.asarray(list(tfs), dtype=float)

    if tfs.ndim != 1:
        raise ValueError("tfs must be one-dimensional")
    if tfs.size == 0:
        raise ValueError("tfs must be nonempty")

    N_t = np.array([grid_index(t_f, dt, tol=tol) for t_f in tfs], dtype=int)

    if np.any(N_t < 1):
        raise ValueError("all final times must have N_t >= 1")

    if len(np.unique(N_t)) != len(N_t):
        raise ValueError("duplicate contour sizes found after grid alignment")

    delta_N_t = np.empty_like(N_t, dtype=float)
    delta_t_f = np.empty_like(tfs, dtype=float)

    delta_N_t[0] = np.nan
    delta_t_f[0] = np.nan

    if len(N_t) > 1:
        delta_N_t[1:] = np.diff(N_t)
        delta_t_f[1:] = np.diff(tfs)

    if require_increasing:
        if np.any(np.diff(N_t) <= 0):
            raise ValueError("tfs must be strictly increasing for continuation")

        if min_delta_N_t is not None and min_delta_N_t > 0:
            if np.any(np.diff(N_t) < min_delta_N_t):
                raise ValueError(
                    f"successive tfs must differ by at least {min_delta_N_t} grid step(s)"
                )

    return pd.DataFrame(
        {
            "t_f": N_t * float(dt),
            "N_t": N_t,
            "delta_t_f": delta_t_f,
            "delta_N_t": delta_N_t,
        }
    )


def make_grid_aligned_tfs(
    *,
    dt: float,
    dense_start: float,
    dense_stop: float,
    dense_step: float | None = None,
    pre=(),
    tail=(),
    sort: bool = True,
    tol: float = 1e-10,
) -> np.ndarray:
    """Build a grid-aligned list of final times.

    The dense part is constructed using integer contour grid steps.  Additional
    sparse pre-points and tail-points are allowed, but they must also be
    aligned with dt.
    """

    if dense_step is None:
        dense_step = dt

    n_start = grid_index(dense_start, dt, tol=tol)
    n_stop = grid_index(dense_stop, dt, tol=tol)
    n_step = grid_index(dense_step, dt, tol=tol)

    if n_step <= 0:
        raise ValueError("dense_step must be positive")
    if n_stop < n_start:
        raise ValueError("dense_stop must be >= dense_start")

    dense_N = np.arange(n_start, n_stop + 1, n_step, dtype=int)

    pre_N = np.array([grid_index(x, dt, tol=tol) for x in pre], dtype=int)
    tail_N = np.array([grid_index(x, dt, tol=tol) for x in tail], dtype=int)

    all_N = np.r_[pre_N, dense_N, tail_N]

    if sort:
        all_N = np.unique(all_N)
    else:
        if len(np.unique(all_N)) != len(all_N):
            raise ValueError("duplicate t_f values found")

    return np.round(all_N * float(dt), 12)


def default_chi_tfs(
    *,
    dt: float,
    t_ev: float = 1.0,
    t_max: float = 10.0,
) -> np.ndarray:
    """Return a simple grid-aligned t_f list for chi continuation tests."""

    return make_grid_aligned_tfs(
        dt=dt,
        dense_start=t_ev,
        dense_stop=min(4.0, t_max),
        dense_step=max(dt, 0.5),
        pre=(0.6, 0.8),
        tail=tuple(x for x in (5.0, 6.0, 8.0, t_max) if x > min(4.0, t_max)),
    )


def fine_post_evap_tfs(
    *,
    dt: float,
    t_ev: float = 1.0,
    t_stop: float = 4.0,
    tail: tuple[float, ...] = (5.0, 6.0, 8.0, 10.0),
    pre: tuple[float, ...] = (0.6, 0.8),
) -> np.ndarray:
    """Return a fine post-evaporation grid with spacing equal to dt."""

    return make_grid_aligned_tfs(
        dt=dt,
        dense_start=t_ev,
        dense_stop=t_stop,
        dense_step=dt,
        pre=pre,
        tail=tail,
    )


def embedded_initial_guess(
    previous: SweepSolution | None,
    *,
    beta: float,
    t_f: float,
    dt: float,
    num_blocks: int,
):
    """Embed the previous solution onto the contour for a new t_f.

    Returns None if there is no previous solution.
    """

    if previous is None:
        return None

    _, new_meta = make_contour_weights(beta=beta, t_f=t_f, dt=dt)

    embedded = embed_block_matrix(
        previous.G,
        old_meta=previous.contour,
        new_meta=new_meta,
        num_blocks=num_blocks,
    )

    return to_numpy(embedded).astype(np.complex128, copy=False)


def solution_summary_row(
    solution: SweepSolution,
    *,
    include_antisymmetry: bool = True,
) -> dict:
    """Return one dataframe row summarizing a sweep solution."""

    row = solver_summary_dict(solution.info)
    row["t_f"] = float(solution.t_f)
    row["used_continuation"] = bool(solution.used_continuation)

    if include_antisymmetry:
        row["antisymmetry_error"] = antisymmetry_error(solution.G)

    return row


def solutions_to_summary(
    solutions: list[SweepSolution],
    *,
    include_antisymmetry: bool = True,
) -> pd.DataFrame:
    """Build a summary dataframe from stored sweep solutions."""

    rows = [
        solution_summary_row(
            solution,
            include_antisymmetry=include_antisymmetry,
        )
        for solution in solutions
    ]

    return pd.DataFrame(rows)


def solve_chi_sweep(
    tfs,
    *,
    beta: float,
    dt: float,
    cal_J: float,
    mu: float,
    p: int,
    use_continuation: bool = True,
    stop_on_failure: bool = True,
    validate_grid: bool = True,
    include_antisymmetry: bool = True,
    verbose: bool = False,
    **solve_kwargs,
) -> tuple[pd.DataFrame, list[SweepSolution]]:
    """Solve the single-replica chi problem over several final times.

    Parameters
    ----------
    tfs:
        Iterable of final times.  The order is respected.  For continuation,
        pass a strictly increasing list of grid-aligned final times.

    beta, dt, cal_J, mu, p:
        Parameters passed to solve_chi.

    use_continuation:
        If True, the previous converged solution is embedded onto the next
        contour and used as initial_G.

    stop_on_failure:
        If True, stop the sweep when a solve fails to converge.  If False,
        continue, but the next run will not use the failed solution for
        continuation.

    validate_grid:
        If True, require all t_f values to be integer multiples of dt.

    include_antisymmetry:
        If True, include max(abs(G + G.T)) in the summary dataframe.

    verbose:
        If True, print progress and pass verbose=True to individual solves
        unless solve_kwargs contains an explicit verbose value.

    solve_kwargs:
        Additional keyword arguments passed directly to solve_chi, such as
        max_newton, gmres_maxiter, line_search_steps, tol, and max_abs_guard.

    Returns
    -------
    summary:
        A dataframe with one row per attempted final time.

    solutions:
        A list of SweepSolution objects.  Each object stores the full Green's
        function at that t_f.
    """

    tfs = np.asarray(list(tfs), dtype=float)

    if validate_grid:
        validation = validate_tfs(
            tfs,
            dt=dt,
            require_increasing=use_continuation,
        )
        tfs = validation["t_f"].to_numpy(dtype=float)

    rows: list[dict] = []
    solutions: list[SweepSolution] = []

    previous: SweepSolution | None = None

    for t_f in tfs:
        if use_continuation and previous is not None and previous.info.converged:
            initial_G = embedded_initial_guess(
                previous,
                beta=beta,
                t_f=float(t_f),
                dt=dt,
                num_blocks=2,
            )
            used_continuation = True
        else:
            initial_G = None
            used_continuation = False

        if verbose:
            print(
                f"Solving chi: t_f={t_f:g}, "
                f"continuation={used_continuation}"
            )

        solver_verbose = solve_kwargs.get("verbose", verbose)
        call_kwargs = dict(solve_kwargs)
        call_kwargs.pop("verbose", None)

        G, w, info = solve_chi(
            beta=beta,
            t_f=float(t_f),
            dt=dt,
            cal_J=cal_J,
            mu=mu,
            p=p,
            initial_G=initial_G,
            verbose=solver_verbose,
            **call_kwargs,
        )

        solution = SweepSolution(
            t_f=float(t_f),
            G=G,
            w=w,
            info=info,
            used_continuation=used_continuation,
        )

        rows.append(
            solution_summary_row(
                solution,
                include_antisymmetry=include_antisymmetry,
            )
        )
        solutions.append(solution)

        if verbose:
            print(
                f"  status={info.status}, "
                f"residual={info.residual_norm}, "
                f"converged={info.converged}"
            )

        if info.converged:
            previous = solution
        elif stop_on_failure:
            break
        else:
            previous = None

    summary = pd.DataFrame(rows)
    return summary, solutions


def collect_component_at_separation(
    solutions: list[SweepSolution],
    *,
    component: str = "RR",
    segment: str = "real",
    separation: float | None = None,
    index: int = -1,
) -> pd.DataFrame:
    """Collect one extracted component value as a function of t_f.

    If separation is None, the value at ``index`` is used.  For example,
    ``index=-1`` gives the last available point on the extracted branch.

    If separation is given, the nearest available extracted point is used.
    """

    rows = []

    for solution in solutions:
        x_vals, vals = extract_component_t0(
            solution.G,
            solution.info,
            component=component,
            segment=segment,
        )

        if separation is None:
            chosen_index = int(index)
        else:
            chosen_index = int(np.argmin(np.abs(x_vals - separation)))

        value = vals[chosen_index]
        x_value = x_vals[chosen_index]

        rows.append(
            {
                "t_f": float(solution.t_f),
                "component": str(component).upper(),
                "segment": str(segment),
                "x": float(x_value),
                "value": value,
                "value_real": float(np.real(value)),
                "value_imag": float(np.imag(value)),
                "value_abs": float(np.abs(value)),
                "converged": bool(solution.info.converged),
                "used_continuation": bool(solution.used_continuation),
            }
        )

    return pd.DataFrame(rows)



#### Added after eta ####

@dataclass(frozen=True)
class EtaSweepSolution:
    """Stored solution for one single-replica eta solve in a t_f sweep."""

    t_f: float
    G_chi: np.ndarray
    G_eta: np.ndarray
    w: np.ndarray
    chi_info: Any
    eta_info: Any
    used_chi_continuation: bool
    used_eta_continuation: bool

    @property
    def contour(self) -> ContourMeta:
        return self.eta_info.contour


def eta_sweep_summary_row(
    solution: EtaSweepSolution,
    *,
    include_antisymmetry: bool = True,
) -> dict[str, Any]:
    """Convert one eta sweep solution into a compact dataframe row."""

    row = {
        "t_f": float(solution.t_f),
        "chi_status": solution.chi_info.status,
        "chi_converged": bool(solution.chi_info.converged),
        "chi_residual_norm": float(solution.chi_info.residual_norm),
        "chi_newton_iterations": int(solution.chi_info.newton.iterations),
        "eta_status": solution.eta_info.status,
        "eta_converged": bool(solution.eta_info.converged),
        "eta_residual_norm": float(solution.eta_info.residual_norm),
        "eta_newton_iterations": int(solution.eta_info.newton.iterations),
        "used_chi_continuation": bool(solution.used_chi_continuation),
        "used_eta_continuation": bool(solution.used_eta_continuation),
    }

    row.update(solution.contour.as_dict())

    if include_antisymmetry:
        row["chi_antisymmetry_error"] = antisymmetry_error(solution.G_chi)
        row["eta_antisymmetry_error"] = antisymmetry_error(solution.G_eta)

    return row


def eta_solutions_to_summary(
    solutions: Sequence[EtaSweepSolution],
    *,
    include_antisymmetry: bool = True,
) -> pd.DataFrame:
    """Build a dataframe summary from eta sweep solutions."""

    rows = [
        eta_sweep_summary_row(
            sol,
            include_antisymmetry=include_antisymmetry,
        )
        for sol in solutions
    ]
    return pd.DataFrame(rows)


def solve_eta_sweep(
    tfs: Sequence[float],
    *,
    beta: float,
    dt: float,
    cal_J: float,
    mu: float,
    p: int,
    t_ev: float,
    a_param: float,
    eta_bilinear_nu: float = 0.0,
    use_continuation: bool = True,
    stop_on_failure: bool = True,
    validate_grid: bool = True,
    include_antisymmetry: bool = True,
    verbose: bool = False,
    chi_solve_kwargs: dict[str, Any] | None = None,
    eta_solve_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[EtaSweepSolution]]:
    """Sweep the single-replica eta solve over final times.

    For each final time, this first solves the chi bath and then solves the eta
    probe using that chi solution as the fixed post-evaporation source.  When
    ``use_continuation`` is true, the previous converged chi and eta solutions
    are embedded onto the new contour and used as initial guesses.
    """

    tfs_arr = np.asarray(list(tfs), dtype=float)

    if validate_grid:
        validation = validate_tfs(
            tfs_arr,
            dt=dt,
            require_increasing=use_continuation,
        )
        tfs_arr = validation["t_f"].to_numpy(dtype=float)

    chi_kwargs = {} if chi_solve_kwargs is None else dict(chi_solve_kwargs)
    eta_kwargs = {} if eta_solve_kwargs is None else dict(eta_solve_kwargs)

    if "initial_G" in chi_kwargs:
        raise ValueError("solve_eta_sweep manages chi initial_G internally")
    if "initial_G" in eta_kwargs:
        raise ValueError("solve_eta_sweep manages eta initial_G internally")

    chi_verbose = chi_kwargs.pop("verbose", verbose)
    eta_verbose = eta_kwargs.pop("verbose", verbose)

    rows: list[dict[str, Any]] = []
    solutions: list[EtaSweepSolution] = []

    previous_chi: SweepSolution | None = None
    previous_eta: SweepSolution | None = None

    for t_f in tfs_arr:
        if verbose:
            step = len(rows) + 1
            print(
                f"\n=== eta sweep {step}/{len(tfs_arr)}: "
                f"t_f={float(t_f):g}, t_ev={t_ev:g}, dt={dt:g}, "
                f"mu={mu:g}, p={p} ===",
                flush=True,
            )

        


        chi_initial = None
        used_chi_continuation = False

        if use_continuation and previous_chi is not None and previous_chi.info.converged:
            chi_initial = embedded_initial_guess(
                previous_chi,
                beta=beta,
                t_f=float(t_f),
                dt=dt,
                num_blocks=2,
            )
            used_chi_continuation = True

        G_chi, w_chi, chi_info = solve_chi(
            beta=beta,
            t_f=float(t_f),
            dt=dt,
            cal_J=cal_J,
            mu=mu,
            p=p,
            initial_G=chi_initial,
            verbose=chi_verbose,
            **chi_kwargs,
        )

        chi_solution = SweepSolution(
            t_f=float(t_f),
            G=G_chi,
            w=w_chi,
            info=chi_info,
            used_continuation=used_chi_continuation,
        )

        if not chi_info.converged:
            row = {
                "t_f": float(t_f),
                "chi_status": chi_info.status,
                "chi_converged": False,
                "chi_residual_norm": float(chi_info.residual_norm),
                "chi_newton_iterations": int(chi_info.newton.iterations),
                "eta_status": "not_run",
                "eta_converged": False,
                "eta_residual_norm": np.nan,
                "eta_newton_iterations": 0,
                "used_chi_continuation": used_chi_continuation,
                "used_eta_continuation": False,
            }
            row.update(chi_info.contour.as_dict())

            if include_antisymmetry:
                row["chi_antisymmetry_error"] = antisymmetry_error(G_chi)
                row["eta_antisymmetry_error"] = np.nan

            rows.append(row)

            previous_chi = None
            previous_eta = None

            if stop_on_failure:
                break

            continue

        eta_initial = None
        used_eta_continuation = False

        if use_continuation and previous_eta is not None and previous_eta.info.converged:
            eta_initial = embedded_initial_guess(
                previous_eta,
                beta=beta,
                t_f=float(t_f),
                dt=dt,
                num_blocks=2,
            )
            used_eta_continuation = True

        G_eta, w_eta, eta_info = solve_eta(
            G_chi=G_chi,
            beta=beta,
            t_f=float(t_f),
            dt=dt,
            cal_J=cal_J,
            mu=mu,
            p=p,
            t_ev=t_ev,
            a_param=a_param,
            eta_bilinear_nu=eta_bilinear_nu,
            initial_G=eta_initial,
            verbose=eta_verbose,
            **eta_kwargs,
        )

        eta_base_solution = SweepSolution(
            t_f=float(t_f),
            G=G_eta,
            w=w_eta,
            info=eta_info,
            used_continuation=used_eta_continuation,
        )

        eta_solution = EtaSweepSolution(
            t_f=float(t_f),
            G_chi=G_chi,
            G_eta=G_eta,
            w=w_eta,
            chi_info=chi_info,
            eta_info=eta_info,
            used_chi_continuation=used_chi_continuation,
            used_eta_continuation=used_eta_continuation,
        )

        solutions.append(eta_solution)
        rows.append(
            eta_sweep_summary_row(
                eta_solution,
                include_antisymmetry=include_antisymmetry,
            )
        )

        if chi_info.converged and eta_info.converged:
            previous_chi = chi_solution
            previous_eta = eta_base_solution
        else:
            previous_chi = chi_solution if chi_info.converged else None
            previous_eta = None

            if stop_on_failure:
                break

    return pd.DataFrame(rows), solutions





#### Added after two-replica eta ####

@dataclass(frozen=True)
class TwoReplicaEtaSweepSolution:
    """Stored solution for one two-replica eta solve in a t_f sweep."""

    t_f: float
    G_chi: np.ndarray
    G_disc: np.ndarray
    G_twisted: np.ndarray
    w: np.ndarray
    chi_info: Any
    info_disc: Any
    info_twisted: Any
    pair: dict
    used_chi_continuation: bool
    used_disc_continuation: bool
    used_twisted_continuation: bool

    @property
    def contour(self) -> ContourMeta:
        return self.info_twisted.contour


def two_replica_eta_sweep_summary_row(
    solution: TwoReplicaEtaSweepSolution,
    *,
    t_b: float | None = None,
    include_antisymmetry: bool = True,
) -> dict[str, Any]:
    """Convert one two-replica eta sweep solution into a dataframe row."""

    row = {
        "t_f": float(solution.t_f),
        "chi_status": solution.chi_info.status,
        "chi_converged": bool(solution.chi_info.converged),
        "chi_residual_norm": float(solution.chi_info.residual_norm),
        "chi_newton_iterations": int(solution.chi_info.newton.iterations),
        "used_chi_continuation": bool(solution.used_chi_continuation),
        "used_disc_continuation": bool(solution.used_disc_continuation),
        "used_twisted_continuation": bool(solution.used_twisted_continuation),
    }

    pair_row = pair_summary_row(solution.pair, t_b=t_b)
    for key, value in pair_row.items():
        if key not in row:
            row[key] = value

    if include_antisymmetry:
        row["chi_antisymmetry_error"] = antisymmetry_error(solution.G_chi)
        row["disc_antisymmetry_error"] = antisymmetry_error(solution.G_disc)
        row["twisted_antisymmetry_error"] = antisymmetry_error(solution.G_twisted)

    return row


def two_replica_eta_solutions_to_summary(
    solutions: Sequence[TwoReplicaEtaSweepSolution],
    *,
    t_b: float | None = None,
    include_antisymmetry: bool = True,
) -> pd.DataFrame:
    """Build a dataframe summary from two-replica eta sweep solutions."""

    rows = [
        two_replica_eta_sweep_summary_row(
            sol,
            t_b=t_b,
            include_antisymmetry=include_antisymmetry,
        )
        for sol in solutions
    ]
    return pd.DataFrame(rows)


def solve_two_replica_eta_sweep(
    tfs: Sequence[float],
    *,
    beta: float,
    dt: float,
    cal_J: float,
    mu: float,
    p: int,
    t_ev: float,
    a_param: float,
    t_b: float | None = None,
    use_continuation: bool = True,
    stop_on_failure: bool = True,
    validate_grid: bool = True,
    include_antisymmetry: bool = True,
    verbose: bool = False,
    chi_solve_kwargs: dict[str, Any] | None = None,
    replica_solve_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[TwoReplicaEtaSweepSolution]]:
    """Sweep the two-replica eta solve over final times.

    For each final time this does three solves:

        1. the one-replica chi bath,
        2. the disconnected two-replica eta saddle,
        3. the chi-twisted two-replica eta saddle.

    If use_continuation is True, the previous converged chi, disconnected eta,
    and twisted eta solutions are embedded onto the new contour and used as
    initial guesses.
    """

    tfs_arr = np.asarray(list(tfs), dtype=float)

    if validate_grid:
        validation = validate_tfs(
            tfs_arr,
            dt=dt,
            require_increasing=use_continuation,
        )
        tfs_arr = validation["t_f"].to_numpy(dtype=float)

    chi_kwargs = {} if chi_solve_kwargs is None else dict(chi_solve_kwargs)
    replica_kwargs = {} if replica_solve_kwargs is None else dict(replica_solve_kwargs)

    if "initial_G" in chi_kwargs:
        raise ValueError("solve_two_replica_eta_sweep manages chi initial_G internally")

    forbidden_replica_keys = {
        "initial_G_disc",
        "initial_G_twisted",
        "mode",
        "return_problem",
    }
    bad_keys = forbidden_replica_keys.intersection(replica_kwargs)
    if bad_keys:
        raise ValueError(
            "solve_two_replica_eta_sweep manages these replica kwargs internally: "
            + ", ".join(sorted(bad_keys))
        )

    chi_verbose = chi_kwargs.pop("verbose", verbose)
    replica_verbose = replica_kwargs.pop("verbose", verbose)

    rows: list[dict[str, Any]] = []
    solutions: list[TwoReplicaEtaSweepSolution] = []

    previous_chi: SweepSolution | None = None
    previous_disc: SweepSolution | None = None
    previous_twisted: SweepSolution | None = None

    for t_f in tfs_arr:
        if verbose:
            step = len(rows) + 1
            print(
                f"\n=== two-replica eta sweep {step}/{len(tfs_arr)}: "
                f"t_f={float(t_f):g}, t_ev={t_ev:g}, dt={dt:g}, "
                f"mu={mu:g}, p={p} ===",
                flush=True,
            )

        chi_initial = None
        used_chi_continuation = False

        if use_continuation and previous_chi is not None and previous_chi.info.converged:
            chi_initial = embedded_initial_guess(
                previous_chi,
                beta=beta,
                t_f=float(t_f),
                dt=dt,
                num_blocks=2,
            )
            used_chi_continuation = True

        G_chi, w_chi, chi_info = solve_chi(
            beta=beta,
            t_f=float(t_f),
            dt=dt,
            cal_J=cal_J,
            mu=mu,
            p=p,
            initial_G=chi_initial,
            verbose=chi_verbose,
            **chi_kwargs,
        )

        chi_solution = SweepSolution(
            t_f=float(t_f),
            G=G_chi,
            w=w_chi,
            info=chi_info,
            used_continuation=used_chi_continuation,
        )

        if not chi_info.converged:
            row = {
                "t_f": float(t_f),
                "chi_status": chi_info.status,
                "chi_converged": False,
                "chi_residual_norm": float(chi_info.residual_norm),
                "chi_newton_iterations": int(chi_info.newton.iterations),
                "disc_status": "not_run",
                "disc_converged": False,
                "disc_residual_norm": np.nan,
                "twisted_status": "not_run",
                "twisted_converged": False,
                "twisted_residual_norm": np.nan,
                "used_chi_continuation": used_chi_continuation,
                "used_disc_continuation": False,
                "used_twisted_continuation": False,
            }
            row.update(chi_info.contour.as_dict())

            if include_antisymmetry:
                row["chi_antisymmetry_error"] = antisymmetry_error(G_chi)
                row["disc_antisymmetry_error"] = np.nan
                row["twisted_antisymmetry_error"] = np.nan

            rows.append(row)

            previous_chi = None
            previous_disc = None
            previous_twisted = None

            if stop_on_failure:
                break

            continue

        disc_initial = None
        twisted_initial = None
        used_disc_continuation = False
        used_twisted_continuation = False

        if use_continuation and previous_disc is not None and previous_disc.info.converged:
            disc_initial = embedded_initial_guess(
                previous_disc,
                beta=beta,
                t_f=float(t_f),
                dt=dt,
                num_blocks=4,
            )
            used_disc_continuation = True

        if use_continuation and previous_twisted is not None and previous_twisted.info.converged:
            twisted_initial = embedded_initial_guess(
                previous_twisted,
                beta=beta,
                t_f=float(t_f),
                dt=dt,
                num_blocks=4,
            )
            used_twisted_continuation = True

        pair = solve_two_replica_eta_pair(
            G_chi=G_chi,
            beta=beta,
            t_f=float(t_f),
            dt=dt,
            cal_J=cal_J,
            mu=mu,
            p=p,
            t_ev=t_ev,
            a_param=a_param,
            initial_G_disc=disc_initial,
            initial_G_twisted=twisted_initial,
            verbose=replica_verbose,
            **replica_kwargs,
        )

        disc_solution = SweepSolution(
            t_f=float(t_f),
            G=pair["G_disc"],
            w=pair["w_disc"],
            info=pair["info_disc"],
            used_continuation=used_disc_continuation,
        )

        twisted_solution = SweepSolution(
            t_f=float(t_f),
            G=pair["G_twisted"],
            w=pair["w_twisted"],
            info=pair["info_twisted"],
            used_continuation=used_twisted_continuation,
        )

        sweep_solution = TwoReplicaEtaSweepSolution(
            t_f=float(t_f),
            G_chi=G_chi,
            G_disc=pair["G_disc"],
            G_twisted=pair["G_twisted"],
            w=pair["w_twisted"],
            chi_info=chi_info,
            info_disc=pair["info_disc"],
            info_twisted=pair["info_twisted"],
            pair=pair,
            used_chi_continuation=used_chi_continuation,
            used_disc_continuation=used_disc_continuation,
            used_twisted_continuation=used_twisted_continuation,
        )

        solutions.append(sweep_solution)
        rows.append(
            two_replica_eta_sweep_summary_row(
                sweep_solution,
                t_b=t_b,
                include_antisymmetry=include_antisymmetry,
            )
        )

        all_converged = (
            chi_info.converged
            and pair["info_disc"].converged
            and pair["info_twisted"].converged
        )

        if all_converged:
            previous_chi = chi_solution
            previous_disc = disc_solution
            previous_twisted = twisted_solution
        else:
            previous_chi = chi_solution if chi_info.converged else None
            previous_disc = disc_solution if pair["info_disc"].converged else None
            previous_twisted = (
                twisted_solution if pair["info_twisted"].converged else None
            )

            if stop_on_failure:
                break

    return pd.DataFrame(rows), solutions