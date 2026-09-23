"""Source-driven analytic large-p eta stitching formulas.

This module does not modify analytic.py.  It provides a generic version of the
eta evaporation formula where the pre-evaporation eta solution and the chi bath
solution are supplied as input functions.
"""

from __future__ import annotations

import numpy as np


def eta_large_p_g_from_sources(
    t1,
    t2,
    *,
    t_ev: float,
    eta_pre_g_func,
    chi_g_func,
    mu_eff: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return large-p eta exponents from generic eta-pre and chi sources.

    The source functions must have the form

        eta_pre_g_func(t1, t2) -> (g_R_eta_pre, g_L_eta_pre)
        chi_g_func(t1, t2)     -> (g_R_chi, g_L_chi)

    and they should accept NumPy arrays.

    This implements the same region-stitching formula as the existing
    eta_large_p_g_real_time, but without hardcoding the pre-evaporation eta
    solution or the chi bath solution.
    """

    t1, t2 = np.broadcast_arrays(
        np.asarray(t1, dtype=np.float64),
        np.asarray(t2, dtype=np.float64),
    )

    scalar_output = t1.shape == ()
    if scalar_output:
        t1 = t1.reshape(1)
        t2 = t2.reshape(1)

    shape = t1.shape

    g_R = np.zeros(shape, dtype=np.complex128)
    g_L = np.zeros(shape, dtype=np.complex128)

    t_ev_arr = np.full(shape, float(t_ev), dtype=np.float64)

    mask_pre = (t1 <= t_ev) & (t2 <= t_ev)
    mask_cross_12 = (t1 > t_ev) & (t2 <= t_ev)
    mask_cross_21 = (t1 <= t_ev) & (t2 > t_ev)
    mask_post = (t1 > t_ev) & (t2 > t_ev)

    # Region I: both points before evaporation/transition.
    pre_R, pre_L = eta_pre_g_func(t1, t2)
    pre_R = _as_complex_array(pre_R, shape)
    pre_L = _as_complex_array(pre_L, shape)

    g_R[mask_pre] = pre_R[mask_pre]
    g_L[mask_pre] = pre_L[mask_pre]

    # Region IIa: t2 <= t_ev < t1.
    if np.any(mask_cross_12):
        bd_R, bd_L = eta_pre_g_func(t_ev_arr, t2)
        chi_R_1e, chi_L_1e = chi_g_func(t1, t_ev_arr)

        bd_R = _as_complex_array(bd_R, shape)
        bd_L = _as_complex_array(bd_L, shape)
        chi_R_1e = _as_complex_array(chi_R_1e, shape)
        chi_L_1e = _as_complex_array(chi_L_1e, shape)

        endpoint = (
            np.real(chi_R_1e)
            + 1j * np.imag(chi_L_1e)
            + 1j * mu_eff * (t1 - t_ev)
        )

        g_R[mask_cross_12] = (bd_R + endpoint)[mask_cross_12]
        g_L[mask_cross_12] = (bd_L + endpoint)[mask_cross_12]

    # Region IIb: t1 <= t_ev < t2, by exchange/conjugation.
    if np.any(mask_cross_21):
        bd_R_swapped, bd_L_swapped = eta_pre_g_func(t_ev_arr, t1)
        chi_R_2e, chi_L_2e = chi_g_func(t2, t_ev_arr)

        bd_R_swapped = _as_complex_array(bd_R_swapped, shape)
        bd_L_swapped = _as_complex_array(bd_L_swapped, shape)
        chi_R_2e = _as_complex_array(chi_R_2e, shape)
        chi_L_2e = _as_complex_array(chi_L_2e, shape)

        endpoint_swapped = (
            np.real(chi_R_2e)
            + 1j * np.imag(chi_L_2e)
            + 1j * mu_eff * (t2 - t_ev)
        )

        g_R[mask_cross_21] = np.conj(
            bd_R_swapped + endpoint_swapped
        )[mask_cross_21]
        g_L[mask_cross_21] = np.conj(
            bd_L_swapped + endpoint_swapped
        )[mask_cross_21]

    # Region III: both points after evaporation/transition.
    if np.any(mask_post):
        chi_R_12, chi_L_12 = chi_g_func(t1, t2)
        chi_R_1e, chi_L_1e = chi_g_func(t1, t_ev_arr)
        chi_R_e2, chi_L_e2 = chi_g_func(t_ev_arr, t2)
        _, chi_L_ee = chi_g_func(t_ev_arr, t_ev_arr)
        _, eta_L_ee = eta_pre_g_func(t_ev_arr, t_ev_arr)

        chi_R_12 = _as_complex_array(chi_R_12, shape)
        chi_L_12 = _as_complex_array(chi_L_12, shape)
        chi_R_1e = _as_complex_array(chi_R_1e, shape)
        chi_L_1e = _as_complex_array(chi_L_1e, shape)
        chi_R_e2 = _as_complex_array(chi_R_e2, shape)
        chi_L_e2 = _as_complex_array(chi_L_e2, shape)
        chi_L_ee = _as_complex_array(chi_L_ee, shape)
        eta_L_ee = _as_complex_array(eta_L_ee, shape)

        g_R_post = (
            chi_R_12
            + 1j
            * np.imag(
                -chi_R_1e
                - chi_R_e2
                + chi_L_1e
                + chi_L_e2
            )
            + 1j * mu_eff * (t1 - t2)
        )

        g_L_post = (
            eta_L_ee
            + chi_L_12
            + np.real(
                chi_R_1e
                + chi_R_e2
                - chi_L_1e
                - chi_L_e2
                + chi_L_ee
            )
            + 1j * mu_eff * (t1 - t2)
        )

        g_R[mask_post] = g_R_post[mask_post]
        g_L[mask_post] = g_L_post[mask_post]

    if scalar_output:
        return complex(g_R[0]), complex(g_L[0])

    return g_R, g_L


def eta_large_p_wightman_from_sources(
    t1,
    t2,
    *,
    p: int,
    t_ev: float,
    eta_pre_g_func,
    chi_g_func,
    mu_eff: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source-driven eta Wightman components G_RR^>, G_RL^>."""

    g_R, g_L = eta_large_p_g_from_sources(
        t1,
        t2,
        t_ev=t_ev,
        eta_pre_g_func=eta_pre_g_func,
        chi_g_func=chi_g_func,
        mu_eff=mu_eff,
    )

    G_RR = -0.5j * np.exp(g_R / p)
    G_RL = -0.5 * np.exp(g_L / p)

    if np.shape(G_RR) == ():
        return complex(G_RR), complex(G_RL)

    return G_RR, G_RL


def eta_large_p_wightman_components_from_sources(
    t1,
    t2,
    *,
    p: int,
    t_ev: float,
    eta_pre_g_func,
    chi_g_func,
    mu_eff: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return RR, RL, LR, LL source-driven eta Wightman components."""

    G_RR, G_RL = eta_large_p_wightman_from_sources(
        t1,
        t2,
        p=p,
        t_ev=t_ev,
        eta_pre_g_func=eta_pre_g_func,
        chi_g_func=chi_g_func,
        mu_eff=mu_eff,
    )

    return {
        "RR": G_RR,
        "RL": G_RL,
        "LR": -G_RL,
        "LL": G_RR,
    }


def eta_large_p_anticommutator_from_sources(
    t1,
    t2,
    *,
    p: int,
    t_ev: float,
    eta_pre_g_func,
    chi_g_func,
    mu_eff: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return RR, RL, LR, LL source-driven eta anticommutator components."""

    g_R, g_L = eta_large_p_g_from_sources(
        t1,
        t2,
        t_ev=t_ev,
        eta_pre_g_func=eta_pre_g_func,
        chi_g_func=chi_g_func,
        mu_eff=mu_eff,
    )

    A_RR = np.real(np.exp(g_R / p))
    A_RL = np.imag(np.exp(g_L / p))

    return {
        "RR": A_RR,
        "RL": A_RL,
        "LR": -A_RL,
        "LL": A_RR,
    }


def make_grid_g_source(time_grid, g_R_grid, g_L_grid, *, tol: float = 1e-10):
    """Wrap precomputed square-grid g_R, g_L arrays as a source function.

    This is useful if the region-I eta answer is known only on a grid.

    The returned function uses nearest grid lookup and checks that requested
    times lie on the supplied grid.  It does not interpolate.
    """

    time_grid = np.asarray(time_grid, dtype=np.float64)
    g_R_grid = np.asarray(g_R_grid, dtype=np.complex128)
    g_L_grid = np.asarray(g_L_grid, dtype=np.complex128)

    if time_grid.ndim != 1:
        raise ValueError("time_grid must be one-dimensional")

    expected = (len(time_grid), len(time_grid))
    if g_R_grid.shape != expected:
        raise ValueError(f"g_R_grid has shape {g_R_grid.shape}, expected {expected}")
    if g_L_grid.shape != expected:
        raise ValueError(f"g_L_grid has shape {g_L_grid.shape}, expected {expected}")

    def source(t1, t2):
        t1_arr, t2_arr = np.broadcast_arrays(
            np.asarray(t1, dtype=np.float64),
            np.asarray(t2, dtype=np.float64),
        )

        i = _nearest_grid_indices(time_grid, t1_arr, tol=tol)
        j = _nearest_grid_indices(time_grid, t2_arr, tol=tol)

        return g_R_grid[i, j], g_L_grid[i, j]

    return source


def make_interpolating_g_source(
    time_grid,
    g_R_grid,
    g_L_grid,
    *,
    bounds_error: bool = True,
):
    """Wrap precomputed square-grid g_R, g_L arrays with interpolation.

    When ``bounds_error`` is false, out-of-range requests are clipped to the
    nearest endpoint.  This is useful with eta_large_p_g_from_sources because it
    evaluates source callables on whole broadcast grids before applying region
    masks, even though only the masked values are physically used.
    """

    from scipy.interpolate import RegularGridInterpolator

    time_grid = np.asarray(time_grid, dtype=np.float64)
    g_R_grid = np.asarray(g_R_grid, dtype=np.complex128)
    g_L_grid = np.asarray(g_L_grid, dtype=np.complex128)

    if time_grid.ndim != 1:
        raise ValueError("time_grid must be one-dimensional")

    expected = (len(time_grid), len(time_grid))
    if g_R_grid.shape != expected:
        raise ValueError(f"g_R_grid has shape {g_R_grid.shape}, expected {expected}")
    if g_L_grid.shape != expected:
        raise ValueError(f"g_L_grid has shape {g_L_grid.shape}, expected {expected}")

    kwargs = dict(
        bounds_error=bounds_error,
        fill_value=None if not bounds_error else None,
    )
    gR_re = RegularGridInterpolator((time_grid, time_grid), np.real(g_R_grid), **kwargs)
    gR_im = RegularGridInterpolator((time_grid, time_grid), np.imag(g_R_grid), **kwargs)
    gL_re = RegularGridInterpolator((time_grid, time_grid), np.real(g_L_grid), **kwargs)
    gL_im = RegularGridInterpolator((time_grid, time_grid), np.imag(g_L_grid), **kwargs)

    def source(t1, t2):
        t1_arr, t2_arr = np.broadcast_arrays(
            np.asarray(t1, dtype=np.float64),
            np.asarray(t2, dtype=np.float64),
        )

        if bounds_error:
            t1_eval = t1_arr
            t2_eval = t2_arr
        else:
            t1_eval = np.clip(t1_arr, time_grid[0], time_grid[-1])
            t2_eval = np.clip(t2_arr, time_grid[0], time_grid[-1])

        pts = np.column_stack([t1_eval.ravel(), t2_eval.ravel()])
        g_R = (gR_re(pts) + 1j * gR_im(pts)).reshape(t1_arr.shape)
        g_L = (gL_re(pts) + 1j * gL_im(pts)).reshape(t1_arr.shape)

        if g_R.shape == ():
            return complex(g_R), complex(g_L)

        return g_R, g_L

    return source


def _as_complex_array(value, shape):
    """Convert value to a complex array broadcastable to shape."""

    arr = np.asarray(value, dtype=np.complex128)

    if arr.shape == ():
        return np.full(shape, arr, dtype=np.complex128)

    return np.broadcast_to(arr, shape).astype(np.complex128, copy=False)


def _nearest_grid_indices(grid, values, *, tol: float):
    """Return nearest grid indices and require values to be grid-aligned."""

    flat = np.ravel(values)
    idx = np.searchsorted(grid, flat)

    idx = np.clip(idx, 0, len(grid) - 1)

    left = np.clip(idx - 1, 0, len(grid) - 1)
    choose_left = np.abs(grid[left] - flat) < np.abs(grid[idx] - flat)
    idx = np.where(choose_left, left, idx)

    err = np.abs(grid[idx] - flat)
    scale = np.maximum(1.0, np.abs(flat))

    if np.any(err > tol * scale):
        bad = flat[np.argmax(err / scale)]
        raise ValueError(f"requested time {bad} is not on the supplied grid")

    return idx.reshape(values.shape)







def main() -> None:
    """Benchmark the source-driven formulas against the existing analytic.py formulas.

    This check uses the old standard sources,

        eta_pre_source = eta_pre_large_p_g_real_time,
        chi_source     = chi_large_p_g_real_time,

    so the source-driven formula should reproduce eta_large_p_g_real_time.
    The plots show GRR and GRL, including both real and imaginary parts.
    """

    from .analytic import (
        beta_from_mu,
        chi_large_p_g_real_time,
        eta_large_p_g_real_time,
        eta_large_p_wightman_real_time,
        eta_pre_large_p_g_real_time,
    )

    import matplotlib.pyplot as plt

    cal_J = 0.5
    mu = 0.5
    p = 16
    t_ev = 6.0
    a_param = 1.0

    beta = beta_from_mu(cal_J, mu)

    t = np.linspace(0.0, 10.0, 101)
    T1, T2 = np.meshgrid(t, t, indexing="ij")

    def eta_pre_source(x, y):
        return eta_pre_large_p_g_real_time(
            x,
            y,
            beta=beta,
            a_param=a_param,
        )

    def chi_source(x, y):
        return chi_large_p_g_real_time(
            x,
            y,
            cal_J=cal_J,
            mu=mu,
        )

    def old_eta_wightman(x, y):
        return eta_large_p_wightman_real_time(
            x,
            y,
            beta=beta,
            cal_J=cal_J,
            mu=mu,
            p=p,
            t_ev=t_ev,
            a_param=a_param,
            nu=0.0,
        )

    def new_eta_wightman(x, y):
        return eta_large_p_wightman_from_sources(
            x,
            y,
            p=p,
            t_ev=t_ev,
            eta_pre_g_func=eta_pre_source,
            chi_g_func=chi_source,
            mu_eff=mu,
        )

    gR_old, gL_old = eta_large_p_g_real_time(
        T1,
        T2,
        beta=beta,
        cal_J=cal_J,
        mu=mu,
        t_ev=t_ev,
        a_param=a_param,
        nu=0.0,
    )

    gR_new, gL_new = eta_large_p_g_from_sources(
        T1,
        T2,
        t_ev=t_ev,
        eta_pre_g_func=eta_pre_source,
        chi_g_func=chi_source,
        mu_eff=mu,
    )

    GRR_old, GRL_old = old_eta_wightman(T1, T2)
    GRR_new, GRL_new = new_eta_wightman(T1, T2)

    print("Benchmark: source-driven eta formulas against analytic.py")
    print("parameters:")
    print(f"  cal_J   = {cal_J}")
    print(f"  mu      = {mu}")
    print(f"  p       = {p}")
    print(f"  beta    = {beta}")
    print(f"  t_ev    = {t_ev}")
    print(f"  a_param = {a_param}")
    print()
    print("errors:")
    print("  max |gR_old   - gR_new|   =", np.max(np.abs(gR_old - gR_new)))
    print("  max |gL_old   - gL_new|   =", np.max(np.abs(gL_old - gL_new)))
    print("  max |GRR_old  - GRR_new|  =", np.max(np.abs(GRR_old - GRR_new)))
    print("  max |GRL_old  - GRL_new|  =", np.max(np.abs(GRL_old - GRL_new)))

    def nonzero_symmetric_limit(x):
        vmax = float(np.nanmax(np.abs(x)))
        return vmax if vmax > 0 else 1e-15

    def shared_limits(a, b):
        amin = float(np.nanmin([np.nanmin(a), np.nanmin(b)]))
        amax = float(np.nanmax([np.nanmax(a), np.nanmax(b)]))
        if amin == amax:
            amin -= 1e-15
            amax += 1e-15
        return amin, amax

    # ------------------------------------------------------------
    # 1. Global old/new/difference comparison.
    # ------------------------------------------------------------
    rows = [
        (r"$\mathrm{Re}\,G^\eta_{RR}$", np.real(GRR_old), np.real(GRR_new)),
        (r"$\mathrm{Im}\,G^\eta_{RR}$", np.imag(GRR_old), np.imag(GRR_new)),
        (r"$\mathrm{Re}\,G^\eta_{RL}$", np.real(GRL_old), np.real(GRL_new)),
        (r"$\mathrm{Im}\,G^\eta_{RL}$", np.imag(GRL_old), np.imag(GRL_new)),
    ]

    fig, axs = plt.subplots(4, 3, figsize=(15, 16), constrained_layout=True)

    for row_idx, (label, old_val, new_val) in enumerate(rows):
        diff = new_val - old_val
        vmin, vmax = shared_limits(old_val, new_val)
        dv = nonzero_symmetric_limit(diff)

        im0 = axs[row_idx, 0].imshow(
            old_val,
            origin="lower",
            extent=[t[0], t[-1], t[0], t[-1]],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        axs[row_idx, 0].set_title("old " + label)
        axs[row_idx, 0].set_xlabel(r"$t_2$")
        axs[row_idx, 0].set_ylabel(r"$t_1$")
        axs[row_idx, 0].axvline(t_ev, color="white", linestyle="--", linewidth=0.8)
        axs[row_idx, 0].axhline(t_ev, color="white", linestyle="--", linewidth=0.8)
        plt.colorbar(im0, ax=axs[row_idx, 0])

        im1 = axs[row_idx, 1].imshow(
            new_val,
            origin="lower",
            extent=[t[0], t[-1], t[0], t[-1]],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        axs[row_idx, 1].set_title("new " + label)
        axs[row_idx, 1].set_xlabel(r"$t_2$")
        axs[row_idx, 1].set_ylabel(r"$t_1$")
        axs[row_idx, 1].axvline(t_ev, color="white", linestyle="--", linewidth=0.8)
        axs[row_idx, 1].axhline(t_ev, color="white", linestyle="--", linewidth=0.8)
        plt.colorbar(im1, ax=axs[row_idx, 1])

        im2 = axs[row_idx, 2].imshow(
            diff,
            origin="lower",
            extent=[t[0], t[-1], t[0], t[-1]],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-dv,
            vmax=dv,
        )
        axs[row_idx, 2].set_title("new - old " + label)
        axs[row_idx, 2].set_xlabel(r"$t_2$")
        axs[row_idx, 2].set_ylabel(r"$t_1$")
        axs[row_idx, 2].axvline(t_ev, color="black", linestyle="--", linewidth=0.8)
        axs[row_idx, 2].axhline(t_ev, color="black", linestyle="--", linewidth=0.8)
        plt.colorbar(im2, ax=axs[row_idx, 2])

    fig.suptitle("Source-driven analytic benchmark: old vs new", y=1.002)
    plt.show()

    # ------------------------------------------------------------
    # 2. Fine zoom near the evaporation surface.
    # ------------------------------------------------------------
    window = 0.5
    n_zoom = 401

    t_zoom = np.linspace(t_ev - window, t_ev + window, n_zoom)
    T1z, T2z = np.meshgrid(t_zoom, t_zoom, indexing="ij")

    GRR_zoom, GRL_zoom = new_eta_wightman(T1z, T2z)

    zoom_rows = [
        (r"$\mathrm{Re}\,G^\eta_{RR}$", np.real(GRR_zoom)),
        (r"$\mathrm{Im}\,G^\eta_{RR}$", np.imag(GRR_zoom)),
        (r"$\mathrm{Re}\,G^\eta_{RL}$", np.real(GRL_zoom)),
        (r"$\mathrm{Im}\,G^\eta_{RL}$", np.imag(GRL_zoom)),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    for ax, (label, val) in zip(axs.flat, zoom_rows):
        im = ax.imshow(
            val,
            origin="lower",
            extent=[t_zoom[0], t_zoom[-1], t_zoom[0], t_zoom[-1]],
            aspect="auto",
        )
        ax.axvline(t_ev, color="white", linestyle="--", linewidth=1)
        ax.axhline(t_ev, color="white", linestyle="--", linewidth=1)
        ax.set_title("zoom: " + label)
        ax.set_xlabel(r"$t_2$")
        ax.set_ylabel(r"$t_1$")
        plt.colorbar(im, ax=ax)

    fig.suptitle(rf"Zoom near $t_{{\rm ev}}={t_ev:g}$", y=1.002)
    plt.show()

    # ------------------------------------------------------------
    # 3. Line cuts across t1 = t_ev.
    # ------------------------------------------------------------
    t1_line = np.linspace(t_ev - window, t_ev + window, 1001)
    t2_fixed_values = [
        t_ev - 0.4,
        t_ev - 0.1,
        t_ev,
        t_ev + 0.1,
        t_ev + 0.4,
    ]

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    for t2_fixed in t2_fixed_values:
        GRR_line, GRL_line = new_eta_wightman(t1_line, t2_fixed)

        axs[0, 0].plot(
            t1_line,
            np.real(GRR_line),
            label=rf"$t_2={t2_fixed:.2f}$",
        )
        axs[0, 1].plot(
            t1_line,
            np.imag(GRR_line),
            label=rf"$t_2={t2_fixed:.2f}$",
        )
        axs[1, 0].plot(
            t1_line,
            np.real(GRL_line),
            label=rf"$t_2={t2_fixed:.2f}$",
        )
        axs[1, 1].plot(
            t1_line,
            np.imag(GRL_line),
            label=rf"$t_2={t2_fixed:.2f}$",
        )

    line_titles = [
        r"$\mathrm{Re}\,G^\eta_{RR}(t_1,t_2)$",
        r"$\mathrm{Im}\,G^\eta_{RR}(t_1,t_2)$",
        r"$\mathrm{Re}\,G^\eta_{RL}(t_1,t_2)$",
        r"$\mathrm{Im}\,G^\eta_{RL}(t_1,t_2)$",
    ]

    for ax, title in zip(axs.flat, line_titles):
        ax.axvline(t_ev, color="black", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel(r"$t_1$")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle(r"Line cuts across the evaporation surface", y=1.002)
    plt.show()

    # ------------------------------------------------------------
    # 4. Direct continuity test.
    # ------------------------------------------------------------
    deltas = np.logspace(-5, -1, 20)
    t2_fixed = t_ev - 0.3

    jump_RR = []
    jump_RL = []

    for delta in deltas:
        GRR_left, GRL_left = new_eta_wightman(t_ev - delta, t2_fixed)
        GRR_right, GRL_right = new_eta_wightman(t_ev + delta, t2_fixed)

        jump_RR.append(abs(GRR_right - GRR_left))
        jump_RL.append(abs(GRL_right - GRL_left))

    jump_RR = np.asarray(jump_RR)
    jump_RL = np.asarray(jump_RL)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(deltas, jump_RR, "o-", label=r"$|\Delta G_{RR}|$")
    ax.loglog(deltas, jump_RL, "o-", label=r"$|\Delta G_{RL}|$")
    ax.loglog(
        deltas,
        deltas * jump_RR[-1] / deltas[-1],
        "--",
        label=r"$\propto\delta$",
    )

    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(
        r"$|G(t_{\rm ev}+\delta,t_2)-G(t_{\rm ev}-\delta,t_2)|$"
    )
    ax.set_title(rf"Continuity test at $t_2={t2_fixed:.2f}$")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()

    plt.show()

    
if __name__ == "__main__":
    main()
