"""Reusable numerical diagnostics.

This module contains small sanity checks used by notebooks and scripts:
convergence summaries, finite-array checks, Majorana antisymmetry checks, and
component-wise comparisons against reference data.

It should stay independent of the chi, eta, and replica equations.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def max_abs(x) -> float:
    """Return max(abs(x)) as a Python float."""

    arr = np.asarray(x)

    if arr.size == 0:
        return 0.0

    return float(np.max(np.abs(arr)))


def finite_array_summary(x) -> dict[str, float | int | bool]:
    """Summarize finite, NaN, infinity, and max-absolute content of an array."""

    arr = np.asarray(x)
    finite = np.isfinite(arr)

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    if np.any(finite):
        finite_max_abs = float(np.max(np.abs(arr)[finite]))
    else:
        finite_max_abs = float("nan")

    return {
        "finite": bool(np.all(finite)),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_max_abs": finite_max_abs,
        "size": int(arr.size),
    }


def antisymmetry_error(G) -> float:
    """Return max(abs(G + G.T)) for a Majorana Green's function."""

    G = np.asarray(G)

    if G.ndim != 2:
        raise ValueError("antisymmetry_error expects a matrix")

    if G.shape[0] != G.shape[1]:
        raise ValueError("antisymmetry_error expects a square matrix")

    return max_abs(G + G.T)


def _maybe_skip_first(num, ref, skip_first: bool):
    num = np.asarray(num, dtype=np.complex128)
    ref = np.asarray(ref, dtype=np.complex128)

    if num.shape != ref.shape:
        raise ValueError(f"shape mismatch: num {num.shape}, ref {ref.shape}")

    if skip_first:
        return num[1:], ref[1:]

    return num, ref


def component_max_error(num, ref, *, skip_first: bool = False) -> float:
    """Return max(abs(num - ref)) for one extracted component."""

    num, ref = _maybe_skip_first(num, ref, skip_first)

    if num.size == 0:
        return 0.0

    return max_abs(num - ref)


def component_mean_error(num, ref, *, skip_first: bool = False) -> float:
    """Return mean(abs(num - ref)) for one extracted component."""

    num, ref = _maybe_skip_first(num, ref, skip_first)

    if num.size == 0:
        return 0.0

    return float(np.mean(np.abs(num - ref)))


def relative_component_error(
    num,
    ref,
    *,
    floor: float = 1e-14,
    skip_first: bool = False,
) -> float:
    """Return max(abs(num-ref) / max(abs(ref), floor)).

    The floor prevents division by zero when the reference component is small or
    exactly zero.
    """

    num, ref = _maybe_skip_first(num, ref, skip_first)

    if num.size == 0:
        return 0.0

    denom = np.maximum(np.abs(ref), floor)
    return float(np.max(np.abs(num - ref) / denom))


def component_error_summary(
    num,
    ref,
    *,
    skip_first: bool = False,
    floor: float = 1e-14,
) -> dict[str, float | bool | int]:
    """Return a compact error summary for one component comparison."""

    num_used, ref_used = _maybe_skip_first(num, ref, skip_first)

    return {
        "skip_first": bool(skip_first),
        "num_points": int(num_used.size),
        "max_abs_error": component_max_error(num, ref, skip_first=skip_first),
        "mean_abs_error": component_mean_error(num, ref, skip_first=skip_first),
        "relative_error": relative_component_error(
            num,
            ref,
            floor=floor,
            skip_first=skip_first,
        ),
    }


def component_should_be_small(vals, *, tol: float = 1e-10) -> dict[str, float | bool]:
    """Check whether a component is numerically small."""

    err = max_abs(vals)

    return {
        "max_abs": err,
        "tol": float(tol),
        "passes": bool(err < tol),
    }


def solver_summary_dict(info) -> dict:
    """Return a flat dictionary summary for a solver info object.

    If the object has ``as_dict()``, that method is used.  Otherwise the
    function extracts common attributes such as status, converged, and
    residual_norm.
    """

    if hasattr(info, "as_dict") and callable(info.as_dict):
        return dict(info.as_dict())

    keys = [
        "method",
        "status",
        "converged",
        "residual_norm",
        "iterations",
        "failure_stage",
        "failure_message",
    ]

    out = {}
    for key in keys:
        if hasattr(info, key):
            out[key] = getattr(info, key)

    return out


def print_solver_summary(info) -> None:
    """Print a compact human-readable solver summary."""

    summary = solver_summary_dict(info)

    preferred = [
        "method",
        "status",
        "converged",
        "residual_norm",
        "newton_iterations",
        "iterations",
        "failure_stage",
        "failure_message",
    ]

    printed = set()

    for key in preferred:
        if key in summary:
            print(f"{key}: {summary[key]}")
            printed.add(key)

    for key, value in summary.items():
        if key not in printed:
            print(f"{key}: {value}")


def assert_converged(info) -> None:
    """Raise AssertionError if a solver info object did not converge."""

    converged = getattr(info, "converged", None)

    if converged is None and isinstance(info, Mapping):
        converged = info.get("converged")

    if not converged:
        status = getattr(info, "status", None)
        residual_norm = getattr(info, "residual_norm", None)

        if isinstance(info, Mapping):
            status = info.get("status", status)
            residual_norm = info.get("residual_norm", residual_norm)

        raise AssertionError(
            f"solver did not converge: status={status}, residual_norm={residual_norm}"
        )