"""Reusable plotting helpers for evaporation numerics.

This module contains plotting utilities used by notebooks.  It should not solve
equations or compute self-energies.  It only visualizes arrays, extracted
components, solver histories, and full contour Green's-function blocks.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt

from finite_p.components import as_contour_meta, get_component_block


def _symmetric_vmax(data, floor: float = 1e-15) -> float:
    """Return a finite symmetric color scale for diverging heatmaps."""

    data = np.asarray(data)

    if data.size == 0:
        return floor

    vmax = np.nanmax(np.abs(data))

    if not np.isfinite(vmax) or vmax == 0:
        return floor

    return float(vmax)


def draw_contour_boundaries(
    ax,
    info_or_meta,
    *,
    color: str = "black",
    secondary_color: str = "gray",
    linewidth: float = 0.9,
    secondary_linewidth: float = 0.75,
) -> None:
    """Draw folded-contour boundary lines on a contour-index plot.

    The main vertical/horizontal line marks the fold between the upper and
    lower branches.  The secondary dotted lines mark the Euclidean/real-time
    interfaces.
    """

    meta = as_contour_meta(info_or_meta)

    # Main fold: upper contour ends and lower contour begins.
    ax.axvline(meta.N_fold - 0.5, color=color, linestyle="--", linewidth=linewidth)
    ax.axhline(meta.N_fold - 0.5, color=color, linestyle="--", linewidth=linewidth)

    # Euclidean/real-time interfaces.
    boundaries = [
        meta.N_beta - 0.5,
        meta.N_fold + meta.N_t - 0.5,
    ]

    for boundary in boundaries:
        ax.axvline(
            boundary,
            color=secondary_color,
            linestyle=":",
            linewidth=secondary_linewidth,
        )
        ax.axhline(
            boundary,
            color=secondary_color,
            linestyle=":",
            linewidth=secondary_linewidth,
        )


def plot_complex_comparison(
    x,
    y_num,
    y_ref=None,
    *,
    title: str = "",
    xlabel: str = "",
    num_label: str = "numerical",
    ref_label: str = "analytic",
    marker: str = "o",
    ref_style: str = "-",
    figsize: tuple[float, float] = (11.0, 4.2),
):
    """Plot real and imaginary parts of one complex curve.

    If ``y_ref`` is supplied, the reference curve is overlaid on both panels.
    """

    x = np.asarray(x)
    y_num = np.asarray(y_num)

    if y_ref is not None:
        y_ref = np.asarray(y_ref)

    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axs[0].plot(x, np.real(y_num), marker, label=num_label)
    if y_ref is not None:
        axs[0].plot(x, np.real(y_ref), ref_style, label=ref_label)
    axs[0].set_title("Re" if not title else f"Re {title}")

    axs[1].plot(x, np.imag(y_num), marker, label=num_label)
    if y_ref is not None:
        axs[1].plot(x, np.imag(y_ref), ref_style, label=ref_label)
    axs[1].set_title("Im" if not title else f"Im {title}")

    for ax in axs:
        ax.set_xlabel(xlabel)
        ax.grid(True, linestyle=":", alpha=0.65)
        ax.legend()

    return fig, axs


def plot_real_time_t0_comparison(
    t,
    G_RR_num,
    G_RL_num,
    G_RR_ref,
    G_RL_ref,
    *,
    species_label: str = r"\chi",
    skip_rr_contact: bool = True,
    figsize: tuple[float, float] = (11.0, 7.0),
):
    """Plot real-time G_RR(t,0), G_RL(t,0) against a reference solution."""

    t = np.asarray(t)
    G_RR_num = np.asarray(G_RR_num)
    G_RL_num = np.asarray(G_RL_num)
    G_RR_ref = np.asarray(G_RR_ref)
    G_RL_ref = np.asarray(G_RL_ref)

    rr_slice = slice(1, None) if skip_rr_contact else slice(None)

    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    axs[0, 0].plot(t[rr_slice], np.real(G_RR_num[rr_slice]), "o", label="numerical")
    axs[0, 0].plot(t[rr_slice], np.real(G_RR_ref[rr_slice]), "-", label="analytic")
    axs[0, 0].set_title(rf"Real time: Re $G^{{{species_label}}}_{{RR}}(t,0)$")

    axs[0, 1].plot(t[rr_slice], np.imag(G_RR_num[rr_slice]), "o", label="numerical")
    axs[0, 1].plot(t[rr_slice], np.imag(G_RR_ref[rr_slice]), "-", label="analytic")
    axs[0, 1].set_title(rf"Real time: Im $G^{{{species_label}}}_{{RR}}(t,0)$")

    axs[1, 0].plot(t, np.real(G_RL_num), "o", label="numerical")
    axs[1, 0].plot(t, np.real(G_RL_ref), "-", label="analytic")
    axs[1, 0].set_title(rf"Real time: Re $G^{{{species_label}}}_{{RL}}(t,0)$")

    axs[1, 1].plot(t, np.imag(G_RL_num), "o", label="numerical")
    axs[1, 1].plot(t, np.imag(G_RL_ref), "-", label="analytic")
    axs[1, 1].set_title(rf"Real time: Im $G^{{{species_label}}}_{{RL}}(t,0)$")

    for ax in axs.ravel():
        ax.set_xlabel(r"$t$")
        ax.grid(True, linestyle=":", alpha=0.65)
        ax.legend()

    return fig, axs


def plot_thermal_t0_comparison(
    tau,
    G_RR_num,
    G_RL_num,
    G_RR_ref,
    G_RL_ref,
    *,
    species_label: str = r"\chi",
    figsize: tuple[float, float] = (11.0, 7.0),
):
    """Plot thermal G_RR(tau,0), G_RL(tau,0) against a reference solution."""

    tau = np.asarray(tau)
    G_RR_num = np.asarray(G_RR_num)
    G_RL_num = np.asarray(G_RL_num)
    G_RR_ref = np.asarray(G_RR_ref)
    G_RL_ref = np.asarray(G_RL_ref)

    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    axs[0, 0].plot(tau, np.real(G_RR_num), "o", label="numerical")
    axs[0, 0].plot(tau, np.real(G_RR_ref), "-", label="analytic")
    axs[0, 0].set_title(rf"Thermal: Re $G^{{{species_label}}}_{{RR}}(\tau,0)$")

    axs[0, 1].plot(tau, np.imag(G_RR_num), "o", label="numerical")
    axs[0, 1].plot(tau, np.imag(G_RR_ref), "-", label="analytic")
    axs[0, 1].set_title(rf"Thermal: Im $G^{{{species_label}}}_{{RR}}(\tau,0)$")

    axs[1, 0].plot(tau, np.real(G_RL_num), "o", label="numerical")
    axs[1, 0].plot(tau, np.real(G_RL_ref), "-", label="analytic")
    axs[1, 0].set_title(rf"Thermal: Re $G^{{{species_label}}}_{{RL}}(\tau,0)$")

    axs[1, 1].plot(tau, np.imag(G_RL_num), "o", label="numerical")
    axs[1, 1].plot(tau, np.imag(G_RL_ref), "-", label="analytic")
    axs[1, 1].set_title(rf"Thermal: Im $G^{{{species_label}}}_{{RL}}(\tau,0)$")

    for ax in axs.ravel():
        ax.set_xlabel(r"$\tau$")
        ax.grid(True, linestyle=":", alpha=0.65)
        ax.legend()

    return fig, axs


def plot_t0_components(
    df,
    *,
    x_col: str = "t",
    components: Iterable[str] = ("RR", "RL", "LR", "LL"),
    species_label: str = r"\chi",
    figsize: tuple[float, float] = (12.0, 8.0),
):
    """Plot real and imaginary parts of extracted G_ab(t,0) components.

    The dataframe should be produced by
    ``components.extract_real_time_t0_components`` or
    ``components.extract_thermal_t0_components``.
    """

    components = tuple(str(comp).upper() for comp in components)

    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axs_flat = axs.ravel()

    x = df[x_col]

    for ax, comp in zip(axs_flat, components):
        real_col = f"G_{comp}_real"
        imag_col = f"G_{comp}_imag"

        ax.plot(x, df[real_col], "o-", label="real")
        ax.plot(x, df[imag_col], "s--", label="imag")
        ax.set_title(rf"$G^{{{species_label}}}_{{{comp}}}$")
        ax.set_xlabel(rf"${x_col}$")
        ax.grid(True, linestyle=":", alpha=0.65)
        ax.legend()

    return fig, axs


def plot_block_heatmaps(
    G,
    info_or_meta,
    *,
    species_label: str = r"\chi",
    components: Iterable[str] = ("RR", "RL", "LR", "LL"),
    title_prefix: str = "",
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (18.0, 8.0),
):
    """Plot full-contour heatmaps for the R/L blocks of a single-replica G.

    The plot has two rows and four columns.  The top row shows real parts and
    the bottom row shows imaginary parts.
    """

    meta = as_contour_meta(info_or_meta)
    components = tuple(str(comp).upper() for comp in components)

    fig, axs = plt.subplots(2, len(components), figsize=figsize, constrained_layout=True)

    if len(components) == 1:
        axs = np.asarray(axs).reshape(2, 1)

    for col, comp in enumerate(components):
        block = get_component_block(G, meta, comp)

        real_data = np.real(block)
        imag_data = np.imag(block)

        real_vmax = _symmetric_vmax(real_data)
        imag_vmax = _symmetric_vmax(imag_data)

        im0 = axs[0, col].imshow(
            real_data,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-real_vmax,
            vmax=real_vmax,
        )
        axs[0, col].set_title(rf"Re $G^{{{species_label}}}_{{{comp}}}$")
        draw_contour_boundaries(axs[0, col], meta)
        fig.colorbar(im0, ax=axs[0, col], fraction=0.046, pad=0.04)

        im1 = axs[1, col].imshow(
            imag_data,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-imag_vmax,
            vmax=imag_vmax,
        )
        axs[1, col].set_title(rf"Im $G^{{{species_label}}}_{{{comp}}}$")
        draw_contour_boundaries(axs[1, col], meta)
        fig.colorbar(im1, ax=axs[1, col], fraction=0.046, pad=0.04)

    for ax in axs.ravel():
        ax.set_xlabel(r"contour index $z_2$")
        ax.set_ylabel(r"contour index $z_1$")

    if title_prefix:
        title = rf"{title_prefix}: full contour $G^{{{species_label}}}$, $t_f={meta.t_f:g}$"
    else:
        title = rf"Full contour $G^{{{species_label}}}$, $t_f={meta.t_f:g}$"

    fig.suptitle(title, fontsize=14)

    return fig, axs


def plot_residual_history(
    info,
    *,
    figsize: tuple[float, float] = (6.5, 4.2),
):
    """Plot Newton residual history from a solver info object."""

    newton_info = getattr(info, "newton", info)
    residual_history = getattr(newton_info, "residual_history", None)

    if residual_history is None:
        raise TypeError("info does not contain residual_history")

    residual_history = np.asarray(residual_history, dtype=float)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.semilogy(np.arange(len(residual_history)), residual_history, "o-")
    ax.set_xlabel("Newton step")
    ax.set_ylabel(r"$\|R\|_{\max}$")
    ax.set_title("Newton residual history")
    ax.grid(True, linestyle=":", alpha=0.65)

    return fig, ax


##### added later for eta ########

def plot_upper_real_time_slice_comparison(
    t1,
    G_RR_num,
    G_RL_num,
    G_RR_ref,
    G_RL_ref,
    *,
    t_ref: float,
    species_label: str = r"\eta",
    title_prefix: str = "",
    figsize: tuple[float, float] = (11.0, 7.0),
):
    """Plot G_RR(t1,t_ref), G_RL(t1,t_ref) against a reference solution."""

    t1 = np.asarray(t1)
    G_RR_num = np.asarray(G_RR_num)
    G_RL_num = np.asarray(G_RL_num)
    G_RR_ref = np.asarray(G_RR_ref)
    G_RL_ref = np.asarray(G_RL_ref)

    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    label = rf"$t_{{\rm ref}}={t_ref}$"

    axs[0, 0].plot(t1, np.real(G_RR_num), "o", label="numerical")
    axs[0, 0].plot(t1, np.real(G_RR_ref), "-", label="analytic")
    axs[0, 0].set_title(
        rf"{title_prefix} Re $G^{{{species_label}}}_{{RR}}(t_1,t_{{\rm ref}})$"
    )

    axs[0, 1].plot(t1, np.imag(G_RR_num), "o", label="numerical")
    axs[0, 1].plot(t1, np.imag(G_RR_ref), "-", label="analytic")
    axs[0, 1].set_title(
        rf"{title_prefix} Im $G^{{{species_label}}}_{{RR}}(t_1,t_{{\rm ref}})$"
    )

    axs[1, 0].plot(t1, np.real(G_RL_num), "o", label="numerical")
    axs[1, 0].plot(t1, np.real(G_RL_ref), "-", label="analytic")
    axs[1, 0].set_title(
        rf"{title_prefix} Re $G^{{{species_label}}}_{{RL}}(t_1,t_{{\rm ref}})$"
    )

    axs[1, 1].plot(t1, np.imag(G_RL_num), "o", label="numerical")
    axs[1, 1].plot(t1, np.imag(G_RL_ref), "-", label="analytic")
    axs[1, 1].set_title(
        rf"{title_prefix} Im $G^{{{species_label}}}_{{RL}}(t_1,t_{{\rm ref}})$"
    )

    for ax in axs.ravel():
        ax.set_xlabel(r"$t_1$")
        ax.grid(True, linestyle=":", alpha=0.65)
        ax.legend(title=label)

    return fig, axs