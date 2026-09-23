"""Generic Newton-Krylov solver.

This module contains only the numerical nonlinear-solver machinery.  It knows
nothing about chi, eta, contours, replicas, or actions.

The problem solved here is

    R(X) = 0,

where X can be a vector, matrix, or higher-rank NumPy array.  The physics
modules provide the residual R(X) and the Jacobian-vector product J[X] dX.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

# SciPy renamed the relative-tolerance keyword from tol to rtol.
_GMRES_RTOL_KEY = "rtol" if "rtol" in signature(gmres).parameters else "tol"


Array = np.ndarray
ResidualFn = Callable[[Array], Array]
JvpFn = Callable[[Array, Array], Array]
PreconditionerFn = Callable[[Array], Array]
ProjectorFn = Callable[[Array], Array]


@dataclass(frozen=True)
class NewtonInfo:
    """Convergence information returned by the Newton-Krylov solver."""

    status: str
    converged: bool
    iterations: int
    residual_norm: float | None
    residual_history: tuple[float, ...]
    gmres_info: tuple[int, ...]
    gmres_matvecs: tuple[int, ...]
    step_history: tuple[float, ...]
    failure_stage: str | None = None
    failure_message: str | None = None


def max_norm(x: Array) -> float:
    """Return max absolute value of an array."""

    x = np.asarray(x)

    if x.size == 0:
        return 0.0

    return float(np.max(np.abs(x)))


def finite_message(x: Array) -> str:
    """Summarize finite, NaN, and infinity content of an array."""

    x = np.asarray(x)
    finite = np.isfinite(x)

    nan_count = int(np.isnan(x).sum())
    inf_count = int(np.isinf(x).sum())

    if np.any(finite):
        finite_max = float(np.max(np.abs(x)[finite]))
    else:
        finite_max = float("nan")

    return (
        f"finite={bool(np.all(finite))}, "
        f"nan={nan_count}, "
        f"inf={inf_count}, "
        f"finite max|.|={finite_max:.3e}"
    )


def newton_krylov(
    x0,
    residual: ResidualFn,
    jvp: JvpFn,
    *,
    preconditioner: PreconditionerFn | None = None,
    projector: ProjectorFn | None = None,
    tol: float = 1e-8,
    max_newton: int = 12,
    gmres_rtol: float = 1e-4,
    gmres_atol: float = 0.0,
    gmres_maxiter: int = 60,
    max_abs_guard: float | None = 2.0,
    line_search_steps: int = 14,
    verbose: bool = False,
) -> tuple[Array, NewtonInfo]:
    """Solve R(X)=0 by Newton-Krylov.

    Parameters
    ----------
    x0:
        Initial guess.  It is converted to a complex NumPy array.

    residual:
        Function returning R(X), with the same shape as X.

    jvp:
        Function returning J[X] dX, where J[X] is the Jacobian of the residual
        at X.  The full Jacobian is never formed.

    preconditioner:
        Optional function approximating the inverse of a useful linear part.
        If supplied, GMRES solves the left-preconditioned system

            M^{-1} J[X] dX = - M^{-1} R(X).

    projector:
        Optional projection applied to the initial guess, Newton step, and line
        search candidates.  For Majorana Green's functions this can be used to
        enforce antisymmetry.

    tol:
        Stop when max(abs(R(X))) < tol.

    max_newton:
        Maximum number of accepted Newton steps.

    gmres_rtol, gmres_atol, gmres_maxiter:
        Parameters passed to scipy.sparse.linalg.gmres.

    max_abs_guard:
        Reject line-search candidates whose max(abs(X)) exceeds this value.
        Set to None to disable this guard.

    line_search_steps:
        Number of damping attempts.  The solver tries alpha = 1, 1/2, 1/4, ...

    verbose:
        If True, print residuals, GMRES diagnostics, and accepted step sizes.

    Returns
    -------
    x:
        Final iterate.

    info:
        NewtonInfo object with convergence diagnostics.
    """

    x = np.asarray(x0, dtype=np.complex128).copy()

    if projector is not None:
        x = np.asarray(projector(x), dtype=np.complex128)

    shape = x.shape
    size = x.size

    residual_history: list[float] = []
    gmres_info: list[int] = []
    gmres_matvecs: list[int] = []
    step_history: list[float] = []

    status = "max_iters"
    failure_stage = None
    failure_message = None
    residual_norm: float | None = None

    def apply_preconditioner(y: Array) -> Array:
        if preconditioner is None:
            return np.asarray(y, dtype=np.complex128)

        out = np.asarray(preconditioner(y), dtype=np.complex128)

        if out.shape != shape:
            raise ValueError(
                f"preconditioner returned shape {out.shape}, expected {shape}"
            )

        return out

    for newton_it in range(max_newton + 1):
        R = np.asarray(residual(x), dtype=np.complex128)

        if R.shape != shape:
            status = "bad_shape"
            failure_stage = "residual"
            failure_message = f"residual returned shape {R.shape}, expected {shape}"
            break

        if not np.all(np.isfinite(R)):
            status = "nonfinite"
            failure_stage = "residual"
            failure_message = finite_message(R)
            break

        residual_norm = max_norm(R)
        residual_history.append(residual_norm)

        if verbose:
            print(f"Newton step {newton_it:02d}: ||R||_max = {residual_norm:.6e}")

        if residual_norm < tol:
            status = "converged"
            break

        if newton_it == max_newton:
            status = "max_iters"
            break

        try:
            rhs = -apply_preconditioner(R).reshape(-1)
        except Exception as exc:
            status = "preconditioner_failed"
            failure_stage = "preconditioner"
            failure_message = repr(exc)
            break

        matvec_calls = {"n": 0}

        def matvec(v_flat: Array) -> Array:
            matvec_calls["n"] += 1

            dX = np.asarray(v_flat, dtype=np.complex128).reshape(shape)
            JdX = np.asarray(jvp(x, dX), dtype=np.complex128)

            if JdX.shape != shape:
                raise ValueError(f"jvp returned shape {JdX.shape}, expected {shape}")

            JdX = apply_preconditioner(JdX)

            return JdX.reshape(-1)

        A = LinearOperator(
            shape=(size, size),
            matvec=matvec,
            dtype=np.complex128,
        )

        try:
            delta_flat, gmres_status = gmres(
                A,
                rhs,
                **{_GMRES_RTOL_KEY: gmres_rtol},
                atol=gmres_atol,
                maxiter=gmres_maxiter,
            )
        except Exception as exc:
            status = "gmres_exception"
            failure_stage = "gmres"
            failure_message = repr(exc)
            break

        gmres_info.append(int(gmres_status))
        gmres_matvecs.append(int(matvec_calls["n"]))

        if verbose:
            print(
                "  GMRES info = "
                f"{gmres_status}, matvecs = {matvec_calls['n']}"
            )

        if gmres_status < 0:
            status = "gmres_failed"
            failure_stage = "gmres"
            failure_message = f"gmres info={gmres_status}"
            break

        delta = np.asarray(delta_flat, dtype=np.complex128).reshape(shape)

        if projector is not None:
            delta = np.asarray(projector(delta), dtype=np.complex128)

        if not np.all(np.isfinite(delta)):
            status = "nonfinite"
            failure_stage = "gmres_step"
            failure_message = finite_message(delta)
            break

        alpha = 1.0
        accepted = False

        for _ in range(line_search_steps):
            candidate = x + alpha * delta

            if projector is not None:
                candidate = np.asarray(projector(candidate), dtype=np.complex128)

            if not np.all(np.isfinite(candidate)):
                alpha *= 0.5
                continue

            if max_abs_guard is not None and max_norm(candidate) > max_abs_guard:
                alpha *= 0.5
                continue

            candidate_R = np.asarray(residual(candidate), dtype=np.complex128)

            if candidate_R.shape != shape:
                status = "bad_shape"
                failure_stage = "line_search"
                failure_message = (
                    f"candidate residual returned shape {candidate_R.shape}, "
                    f"expected {shape}"
                )
                break

            if not np.all(np.isfinite(candidate_R)):
                alpha *= 0.5
                continue

            candidate_norm = max_norm(candidate_R)

            if candidate_norm < residual_norm:
                x = candidate
                step_history.append(alpha)
                accepted = True

                if verbose:
                    print(f"  accepted alpha = {alpha:.6e}")

                break

            alpha *= 0.5

        if failure_stage == "line_search":
            break

        if not accepted:
            status = "line_search_failed"
            failure_stage = "line_search"
            failure_message = (
                f"could not reduce residual from {residual_norm:.3e}"
            )
            break

    converged = status == "converged"

    info = NewtonInfo(
        status=status,
        converged=converged,
        iterations=len(step_history),
        residual_norm=residual_norm,
        residual_history=tuple(residual_history),
        gmres_info=tuple(gmres_info),
        gmres_matvecs=tuple(gmres_matvecs),
        step_history=tuple(step_history),
        failure_stage=failure_stage,
        failure_message=failure_message,
    )

    return x, info
