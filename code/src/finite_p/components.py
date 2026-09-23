"""Shared R/L component extraction helpers.

This module contains only matrix-index bookkeeping for Green's functions with
R,L block ordering,

    G = [[G_RR, G_RL],
         [G_LR, G_LL]].

It is intentionally independent of the chi, eta, and replica equations.  The
same helpers can be used for single-replica chi, single-replica eta, and later
as building blocks for two-replica extractions.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .contour import ContourMeta, contour_index_for_time, to_numpy


def as_contour_meta(info_or_meta) -> ContourMeta:
    """Return a ContourMeta from either a meta object or a solve-info object.

    Solver info objects, such as ChiSolveInfo, store the contour metadata as
    ``info.contour``.  Passing either the info object or the metadata directly
    should therefore work in notebooks.
    """

    if isinstance(info_or_meta, ContourMeta):
        return info_or_meta

    meta = getattr(info_or_meta, "contour", None)
    if isinstance(meta, ContourMeta):
        return meta

    raise TypeError("expected ContourMeta or an object with a ContourMeta .contour")


def normalize_species(species: str) -> str:
    """Normalize an R/L species label."""

    species = str(species).upper()

    if species not in {"R", "L"}:
        raise ValueError("species must be 'R' or 'L'")

    return species


def normalize_component(component: str) -> str:
    """Normalize a two-letter component label such as RR, RL, LR, or LL."""

    component = str(component).upper()

    if len(component) != 2:
        raise ValueError("component must have two letters, e.g. 'RR' or 'RL'")

    normalize_species(component[0])
    normalize_species(component[1])

    return component


def rl_block_slice(info_or_meta, species: str) -> slice:
    """Return the matrix slice for the R or L block."""

    meta = as_contour_meta(info_or_meta)
    species = normalize_species(species)

    C = meta.N_contour

    if species == "R":
        return slice(0, C)

    return slice(C, 2 * C)


def get_rl_block(matrix, info_or_meta, row_species: str, col_species: str) -> np.ndarray:
    """Extract one R/L block from a full single-replica Green's function."""

    meta = as_contour_meta(info_or_meta)
    matrix = to_numpy(matrix)

    expected = (2 * meta.N_contour, 2 * meta.N_contour)
    if matrix.shape != expected:
        raise ValueError(f"matrix has shape {matrix.shape}, expected {expected}")

    row = rl_block_slice(meta, row_species)
    col = rl_block_slice(meta, col_species)

    return matrix[row, col]


def get_component_block(matrix, info_or_meta, component: str) -> np.ndarray:
    """Extract a block using a component label RR, RL, LR, or LL."""

    component = normalize_component(component)

    return get_rl_block(
        matrix,
        info_or_meta,
        row_species=component[0],
        col_species=component[1],
    )


def upper_real_t0_indices(info_or_meta) -> tuple[np.ndarray, np.ndarray, int]:
    """Return indices for extracting G(t,0) on the upper real-time branch.

    The reference point is the first upper real-time contour point.  Therefore
    the returned x-axis is the time separation

        0, dt, 2dt, ...

    This is the convention used for comparison with analytic G(t,0).
    """

    meta = as_contour_meta(info_or_meta)

    if meta.N_t < 1:
        raise ValueError("real-time extraction requires N_t >= 1")

    idx = meta.N_beta + np.arange(meta.N_t)
    ref = meta.N_beta
    t_vals = np.arange(meta.N_t) * meta.dt

    return t_vals, idx, ref


def upper_thermal_t0_indices(info_or_meta) -> tuple[np.ndarray, np.ndarray, int]:
    """Return indices for extracting G(tau,0) on the upper Euclidean segment."""

    meta = as_contour_meta(info_or_meta)

    if meta.N_beta < 1:
        raise ValueError("thermal extraction requires N_beta >= 1")
    if meta.N_t < 1:
        raise ValueError("thermal extraction uses the first real-time point as reference")

    idx = np.arange(meta.N_beta)
    ref = meta.N_beta
    tau_vals = -np.arange(meta.N_beta, 0, -1) * meta.euclidean_step

    return tau_vals, idx, ref


def extract_component_t0(
    matrix,
    info_or_meta,
    component: str,
    segment: str = "real",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one component G_ab(t,0) or G_ab(tau,0).

    Parameters
    ----------
    matrix:
        Full single-replica Green's function with R,L block ordering.

    info_or_meta:
        Either a ContourMeta object or a solve-info object with ``.contour``.

    component:
        One of ``RR``, ``RL``, ``LR``, or ``LL``.

    segment:
        ``"real"`` for the upper real-time branch, or ``"thermal"`` /
        ``"euclidean"`` for the upper Euclidean preparation segment.
    """

    segment = str(segment).lower()
    block = get_component_block(matrix, info_or_meta, component)

    if segment in {"real", "realtime", "real_time"}:
        x_vals, idx, ref = upper_real_t0_indices(info_or_meta)
    elif segment in {"thermal", "euclidean", "imaginary"}:
        x_vals, idx, ref = upper_thermal_t0_indices(info_or_meta)
    else:
        raise ValueError("segment must be 'real' or 'thermal'")

    return x_vals, block[idx, ref]


def extract_real_time_t0(
    matrix,
    info_or_meta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract G_RR(t,0) and G_RL(t,0) on the upper real-time branch.

    This preserves the old chi benchmark interface.
    """

    t_vals, G_RR = extract_component_t0(
        matrix,
        info_or_meta,
        component="RR",
        segment="real",
    )
    _, G_RL = extract_component_t0(
        matrix,
        info_or_meta,
        component="RL",
        segment="real",
    )

    return t_vals, G_RR, G_RL


def extract_thermal_t0(
    matrix,
    info_or_meta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract G_RR(tau,0) and G_RL(tau,0) on the upper thermal segment.

    This preserves the old chi benchmark interface.
    """

    tau_vals, G_RR = extract_component_t0(
        matrix,
        info_or_meta,
        component="RR",
        segment="thermal",
    )
    _, G_RL = extract_component_t0(
        matrix,
        info_or_meta,
        component="RL",
        segment="thermal",
    )

    return tau_vals, G_RR, G_RL


def extract_t0_components(
    matrix,
    info_or_meta,
    *,
    segment: str,
    components: Iterable[str] = ("RR", "RL", "LR", "LL"),
    include_complex: bool = True,
) -> pd.DataFrame:
    """Extract several components into a dataframe.

    The dataframe contains real, imaginary, and absolute-value columns for each
    requested component.  If ``include_complex`` is True, it also includes the
    raw complex column.
    """

    segment = str(segment).lower()

    if segment in {"real", "realtime", "real_time"}:
        x_name = "t"
    elif segment in {"thermal", "euclidean", "imaginary"}:
        x_name = "tau"
    else:
        raise ValueError("segment must be 'real' or 'thermal'")

    data: dict[str, np.ndarray] = {}

    x_reference = None

    for component in components:
        component = normalize_component(component)
        x_vals, vals = extract_component_t0(
            matrix,
            info_or_meta,
            component=component,
            segment=segment,
        )

        if x_reference is None:
            x_reference = x_vals
            data[x_name] = x_vals
        elif not np.array_equal(x_reference, x_vals):
            raise RuntimeError("inconsistent extraction axes")

        key = f"G_{component}"

        if include_complex:
            data[key] = vals

        data[f"{key}_real"] = np.real(vals)
        data[f"{key}_imag"] = np.imag(vals)
        data[f"{key}_abs"] = np.abs(vals)

    return pd.DataFrame(data)


def extract_real_time_t0_components(
    matrix,
    info_or_meta,
    components: Iterable[str] = ("RR", "RL", "LR", "LL"),
    *,
    include_complex: bool = True,
) -> pd.DataFrame:
    """Extract several G_ab(t,0) components on the upper real-time branch."""

    return extract_t0_components(
        matrix,
        info_or_meta,
        segment="real",
        components=components,
        include_complex=include_complex,
    )


def extract_thermal_t0_components(
    matrix,
    info_or_meta,
    components: Iterable[str] = ("RR", "RL", "LR", "LL"),
    *,
    include_complex: bool = True,
) -> pd.DataFrame:
    """Extract several G_ab(tau,0) components on the upper thermal segment."""

    return extract_t0_components(
        matrix,
        info_or_meta,
        segment="thermal",
        components=components,
        include_complex=include_complex,
    )


###### added later for eta #######

def extract_upper_real_time_slice(
    matrix,
    info_or_meta,
    component: str,
    t_ref: float,
    *,
    include_contact: bool = False,
):
    """Extract G_ab(t1,t_ref) with both points on the upper real-time branch.

    Unlike extract_real_time_t0, this returns absolute physical times t1.
    This is needed for eta, whose Green's function is not time-translation
    invariant after evaporation.
    """

    meta = as_contour_meta(info_or_meta)
    block = get_component_block(matrix, meta, component)

    idx = meta.N_beta + np.arange(meta.N_t)
    t1_vals = (np.arange(meta.N_t) + 1) * meta.dt

    ref = contour_index_for_time(meta, t_ref, branch="upper")
    vals = block[idx, ref]

    if include_contact:
        keep = t1_vals >= t_ref - 1e-12
    else:
        keep = t1_vals > t_ref + 1e-12

    return t1_vals[keep], vals[keep]


def extract_upper_real_time_slice_components(
    matrix,
    info_or_meta,
    t_ref: float,
    *,
    components: Iterable[str] = ("RR", "RL", "LR", "LL"),
    include_contact: bool = False,
    include_complex: bool = True,
) -> pd.DataFrame:
    """Extract several G_ab(t1,t_ref) components into a dataframe."""

    data = None

    for component in components:
        component = normalize_component(component)
        t1_vals, vals = extract_upper_real_time_slice(
            matrix,
            info_or_meta,
            component,
            t_ref,
            include_contact=include_contact,
        )

        if data is None:
            data = pd.DataFrame({"t1": t1_vals})
            data["t_ref"] = float(t_ref)

        if include_complex:
            data[f"G_{component}"] = vals

        data[f"G_{component}_real"] = np.real(vals)
        data[f"G_{component}_imag"] = np.imag(vals)
        data[f"G_{component}_abs"] = np.abs(vals)

    if data is None:
        return pd.DataFrame()

    return data
