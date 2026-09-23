#!/usr/bin/env python3
"""Generate data figures for the blast-freezing paper.

See ``code/figure-reproduction/README.md`` for the figure map.  It writes one standalone image
per plotted quantity; notebook panels that used subplots are split into
separate files. Manuscript Figures 2 and 4 are also assembled with panel labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import MaxNLocator
import numpy as np

# Support direct execution from any working directory as well as notebook imports.
_CODE_ROOT = Path(__file__).resolve().parents[1]
for _path in (_CODE_ROOT, _CODE_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from util.paths import REPO_ROOT, CODE_ROOT, FIGURE_INPUT_ROOT, FIGURE_ROOT, NUMERICS_DATA_ROOT


PAPER_FIG_DIR = FIGURE_INPUT_ROOT
NUMERICS_RESULTS_DIR = NUMERICS_DATA_ROOT

def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

from bulk_reconstruction import reconstruction as brl
from large_p.analytic import chi_large_p_g_real_time
from large_p.analytic_sources import eta_large_p_g_from_sources, make_interpolating_g_source
from finite_p.contour import ContourMeta
from large_p.evap_syk import (
    compute_size_operator_DeltaNchi_grid,
    compute_size_operator_DeltaNeta_grid,
    evapSYK2pt,
)
from large_p.formation_sources import (
    correlation_matrix_from_exp_g,
    coupled_syk_g_grids,
    generate_black_hole_formation_profile,
    reconstruction_coordinates,
    rescued_bh_epsilon,
    run_bulk_reconstruction_from_correlator,
    uniform_time_grid,
)
from finite_p.io_utils import slug_float
from large_p.largep_syk import coupledSYK2pt



@dataclass
class FigureParams:
    """Common numerical parameters for the generated data figures."""

    a: float = 1.0
    tev: float = 15.0
    mu: float = 0.1
    nu: float = 0.0
    p: int = 16
    dt: float = 0.5
    tmin: float = 0.0
    tmax: float = 80.0
    size_cutoff: float = 100.0
    size_fixed_v_time: float | None = None
    formation_beta: float = 10.0
    formation_t0: float = -20.0
    chi_cal_j: float = 0.5
    mi_tf: float = 25.0
    mi_kernel_norm_cutoff: float = 50.0
    mi_direction: str = "R"
    mi_component: int = 0
    mi_fixed_v_times: tuple[float, ...] = (6.5, 12.0, 17.5)
    mi_num_fixed_v_cuts: int = 3
    wavepacket_early_time: float | None = None
    wavepacket_late_time: float | None = None
    wavepacket_depth: float = 1.0
    wavepacket_early_u: float | None = None
    wavepacket_early_v: float | None = None
    wavepacket_late_u: float | None = None
    wavepacket_late_v: float | None = None
    wavepacket_mover: str = "right"

    def tgrid(self) -> np.ndarray:
        n_steps = int(round((self.tmax - self.tmin) / self.dt))
        return self.tmin + self.dt * np.arange(n_steps + 1)

    def eps(self) -> float:
        return rescued_bh_epsilon(self.formation_beta)

    def large_p_dict(self) -> dict[str, float]:
        return {
            "a": self.a,
            "tev": self.tev,
            "mu": self.mu,
            "nu": self.nu,
            "eps": self.eps(),
            "p": float(self.p),
        }


def save_figure(fig, out_dir: Path, filename: str, *, dpi: int = 300) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    try:
        return display_path(path)
    except ValueError:
        return str(path)


def finite_symmetric_limit(data: np.ndarray, floor: float = 1e-12) -> float:
    data = np.asarray(data)
    if not np.any(np.isfinite(data)):
        return floor
    vmax = np.nanmax(np.abs(data))
    if not np.isfinite(vmax) or vmax == 0:
        return floor
    return float(vmax)


def component_color_scale(data: np.ndarray, floor: float = 1e-12) -> tuple[str, float, float]:
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return "viridis", 0.0, 1.0

    dmin = float(np.min(finite))
    dmax = float(np.max(finite))
    if dmax - dmin < floor:
        center = 0.5 * (dmin + dmax)
        return "viridis", center - floor, center + floor

    if dmin < 0.0 < dmax:
        vmax = max(abs(dmin), abs(dmax), floor)
        return "RdBu_r", -vmax, vmax
    if dmax <= 0.0:
        return "Blues_r", dmin, dmax
    return "Reds", dmin, dmax


def bulk_norm_grid(gates: dict[tuple[int, int], np.ndarray], N: int, NT: int) -> np.ndarray:
    norms_mat = np.full((NT, NT), np.nan)
    for (u, v), U in gates.items():
        U11 = U[:N, :N]
        norms_mat[u, v] = np.linalg.norm(U11, ord="fro")
    return norms_mat


def build_large_p_reconstruction(params: FigureParams) -> dict[str, np.ndarray]:
    tgrid = params.tgrid()
    print(f"large-p reconstruction: NT={len(tgrid)}, t=[{tgrid[0]}, {tgrid[-1]}]")

    CorM, GRR, GRL, RzField, LzField = evapSYK2pt(tgrid, params.large_p_dict())
    A = np.real(CorM)
    C = np.imag(CorM)
    N = 2
    NT = len(tgrid)

    gates, KL_all, KR_all, is_legit, tf_list, tini_list = brl.compute_bulk_gates(A, N, NT)
    norms_mat = bulk_norm_grid(gates, N, NT)

    Tu, Tv = np.meshgrid(tgrid, tgrid, indexing="ij")
    TT = 0.5 * (Tu + Tv)
    ZZ = 0.5 * (Tv - Tu)

    return {
        "tgrid": tgrid,
        "CorM": CorM,
        "GRR": GRR,
        "GRL": GRL,
        "RzField": RzField,
        "LzField": LzField,
        "A": A,
        "C": C,
        "gates": gates,
        "KL_all": KL_all,
        "KR_all": KR_all,
        "is_legit": is_legit,
        "tf_list": tf_list,
        "tini_list": tini_list,
        "norms_mat": norms_mat,
        "TT": TT,
        "ZZ": ZZ,
        "N": N,
        "NT": NT,
    }


def plot_two_point_heatmap(recon: dict[str, np.ndarray], params: FigureParams, out_dir: Path) -> str:
    tgrid = recon["tgrid"]
    # evapSYK2pt returns the normalized exponentials used by the HKLL code:
    # GRR=e^{g_R/p} and GRL=e^{g_L/p}.  For the plotted Wightman function use
    # the convention in paper/supplementary.tex, Sec. S3.
    G_RR, G_RL = wightman_components(recon)

    panels = [
        (np.real(G_RR), r"$\mathrm{Re}\,G^{\eta,>}_{RR}$"),
        (np.imag(G_RR), r"$\mathrm{Im}\,G^{\eta,>}_{RR}$"),
        (np.real(G_RL), r"$\mathrm{Re}\,G^{\eta,>}_{RL}$"),
        (np.imag(G_RL), r"$\mathrm{Im}\,G^{\eta,>}_{RL}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), constrained_layout=True)
    for ax, (data, title) in zip(axes.ravel(), panels):
        cmap, vmin, vmax = component_color_scale(data)
        im = ax.pcolormesh(
            tgrid,
            tgrid,
            data,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.axvline(params.tev, color="black", linestyle=":", linewidth=0.85)
        ax.axhline(params.tev, color="black", linestyle=":", linewidth=0.85)
        ax.set_xlabel(r"$t_2$")
        ax.set_ylabel(r"$t_1$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    return save_figure(fig, out_dir, "fig2_two_point_heatmap.png")


def wightman_components(recon: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return -0.5j * recon["GRR"], -0.5 * recon["GRL"]


def annotate_evaporation_regions(ax, recon, params: FigureParams) -> None:
    """Separate I (v < tev), II (u < tev < v), and III (u > tev).

    Here u=t-z and v=t+z. Draw only the parts of the null boundaries
    supported by reconstructed gates, matching the regions in panel (c).
    """
    times = recon["tgrid"]
    evaporation_index = nearest_time_index(times, params.tev)
    if not np.isclose(times[evaporation_index], params.tev):
        raise ValueError("Region annotations require t_ev on the reconstruction grid")
    visible = np.isfinite(recon["norms_mat"])
    stroke = [path_effects.Stroke(linewidth=2.8, foreground="black"),
              path_effects.Normal()]
    for u, v in (
        (np.arange(evaporation_index + 1), np.full(evaporation_index + 1, evaporation_index)),
        (np.full(len(times) - evaporation_index, evaporation_index),
         np.arange(evaporation_index, len(times))),
    ):
        z = (times[v] - times[u]) / 2
        t = (times[v] + times[u]) / 2
        valid = visible[u, v]
        ax.plot(np.where(valid, z, np.nan), np.where(valid, t, np.nan),
                color="white", linestyle=(0, (5, 4)), linewidth=1.6,
                path_effects=stroke, zorder=5)

    span = params.tev - float(times[0])
    positions = {
        "I": (0.15 * span, float(times[0]) + 0.45 * span),
        "II": (0.45 * span, params.tev),
        "III": (0.4 * span, params.tev + 0.55 * (float(times[-1]) - params.tev)),
    }
    for label, (z, t) in positions.items():
        ax.text(z, t, label, fontsize=20, fontweight="bold", color="white",
                ha="center", va="center", zorder=6,
                path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")])



def plot_gate_color_map(
    recon: dict[str, np.ndarray],
    params: FigureParams,
    out_dir: Path,
    filename: str,
    *,
    title: str,
    draw_formation_lines: bool = False,
    draw_regions: bool = False,
) -> str:
    display_norms = recon["norms_mat"] / recon["N"]
    data = np.where(recon["ZZ"] >= 0, display_norms, np.nan)
    t_min = float(np.nanmin(recon["TT"]))
    t_max = float(np.nanmax(recon["TT"]))
    z_max = float(np.nanmax(np.where(recon["ZZ"] >= 0, recon["ZZ"], np.nan)))
    height = 5.8 * max((t_max - t_min) / max(z_max, params.dt), 1.0)

    fig, ax = plt.subplots(figsize=(5.8, height), constrained_layout=True)
    im = ax.pcolormesh(recon["ZZ"], recon["TT"], data, shading="auto", cmap="viridis")
    if draw_formation_lines:
        ax.axhline(0.0, color="white", linestyle="--", linewidth=0.9)
    if draw_regions:
        annotate_evaporation_regions(ax, recon, params)
    else:
        ax.axhline(params.tev, color="white", linestyle=":", linewidth=0.9)
    ax.set_xlim(0.0, z_max)
    ax.set_ylim(t_min, t_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"bulk depth $z$")
    ax.set_ylabel(r"bulk time $t$")
    ax.set_title(title)
    cbar = fig.colorbar(
        im,
        ax=ax,
        label=r"$\|U_{11}\|_F/N$",
        fraction=0.035,
        pad=0.03,
        shrink=0.35,
        aspect=16,
    )
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()
    return save_figure(fig, out_dir, filename)


def nearest_time_index(tgrid: np.ndarray, target_time: float) -> int:
    return int(np.argmin(np.abs(np.asarray(tgrid, dtype=float) - target_time)))


def default_wavepacket_times(params: FigureParams) -> tuple[float, float]:
    early_time = params.wavepacket_early_time
    if early_time is None:
        early_time = 0.5 * params.formation_t0

    late_time = params.wavepacket_late_time
    if late_time is None:
        late_time = min(params.tev + 2.0, params.tmax - 4.0 * params.dt)
        late_time = max(late_time, params.tev + params.dt)

    return float(early_time), float(late_time)


def wavepacket_point_from_time(center_time: float, depth: float) -> tuple[float, float]:
    if depth < 0.0:
        raise ValueError("wavepacket_depth must be nonnegative")
    return float(center_time - depth), float(center_time + depth)


def optional_wavepacket_point(u: float | None, v: float | None) -> tuple[float, float] | None:
    if u is None and v is None:
        return None
    if u is None or v is None:
        raise ValueError("wavepacket point requires both u and v")
    if v < u:
        raise ValueError("wavepacket point must satisfy v >= u")
    return float(u), float(v)


def default_wavepacket_points(params: FigureParams) -> tuple[tuple[float, float], tuple[float, float]]:
    early_time, late_time = default_wavepacket_times(params)
    early_point = optional_wavepacket_point(params.wavepacket_early_u, params.wavepacket_early_v)
    late_point = optional_wavepacket_point(params.wavepacket_late_u, params.wavepacket_late_v)
    if early_point is None:
        early_point = wavepacket_point_from_time(early_time, params.wavepacket_depth)
    if late_point is None:
        late_point = wavepacket_point_from_time(late_time, params.wavepacket_depth)
    return early_point, late_point


def require_post_evaporation_bulk_point(point: tuple[float, float], params: FigureParams) -> tuple[float, float]:
    u_start, v_start = point
    if not (params.tev < u_start < v_start):
        raise ValueError(
            "late wave-packet point must satisfy "
            f"t_ev < u < v; got t_ev={params.tev:g}, u={u_start:g}, v={v_start:g}"
        )
    return point


def wavepacket_start_indices(
    tgrid: np.ndarray,
    u_start: float,
    v_start: float,
) -> tuple[int, int, float, float]:
    if v_start < u_start:
        raise ValueError("wavepacket point must satisfy v >= u")

    t_min = float(tgrid[0])
    t_max = float(tgrid[-1])
    dt = float(tgrid[1] - tgrid[0])
    tol = 0.51 * dt
    if u_start < t_min - tol or v_start > t_max + tol:
        raise ValueError(
            f"wavepacket point ({u_start:g}, {v_start:g}) is outside "
            f"the reconstruction grid [{t_min:g}, {t_max:g}]"
        )

    u_idx = nearest_time_index(tgrid, u_start)
    v_idx = nearest_time_index(tgrid, v_start)
    if v_idx < u_idx:
        raise ValueError("nearest grid point has v index smaller than u index")
    return int(u_idx), int(v_idx), float(tgrid[u_idx]), float(tgrid[v_idx])


def wavepacket_is_left(params: FigureParams) -> bool:
    mover = params.wavepacket_mover.strip().lower()
    if mover in {"right", "r"}:
        return False
    if mover in {"left", "l"}:
        return True
    raise ValueError("wavepacket_mover must be 'right' or 'left'")


def wavepacket_observable_grids(
    R_links: np.ndarray,
    L_links: np.ndarray,
    recon: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tgrid = recon["tgrid"]
    dt = float(tgrid[1] - tgrid[0])
    NT = int(recon["NT"])
    N = int(recon["N"])
    NT_fine = 2 * NT
    dt_fine = 0.5 * dt
    t_fine = float(tgrid[0]) + (np.arange(NT_fine) + 0.5) * dt_fine
    z_fine = (np.arange(NT_fine) + 0.5) * dt_fine
    intensity = np.full((NT_fine, NT_fine), np.nan)
    z_component = np.full((NT_fine, NT_fine), np.nan)

    z_op = np.zeros(N)
    z_op[0::2] = 1.0
    z_op[1::2] = -1.0

    for u in range(NT):
        for v in range(u, NT):
            if not recon["is_legit"][u, v]:
                continue

            r_vec = R_links[u, v]
            if not np.any(np.isnan(r_vec)):
                r_weights = np.abs(r_vec) ** 2
                intensity[u + v, v - u] = np.sum(r_weights)
                z_component[u + v, v - u] = np.sum(z_op * r_weights)
            else:
                intensity[u + v, v - u] = 0.0
                z_component[u + v, v - u] = 0.0

            l_idx_z = v - u - 1
            if l_idx_z < 0:
                continue
            l_vec = L_links[u, v]
            if not np.any(np.isnan(l_vec)):
                l_weights = np.abs(l_vec) ** 2
                intensity[u + v, l_idx_z] = np.sum(l_weights)
                z_component[u + v, l_idx_z] = np.sum(z_op * l_weights)
            else:
                intensity[u + v, l_idx_z] = 0.0
                z_component[u + v, l_idx_z] = 0.0

    return intensity, z_component, z_fine, t_fine


def plot_wavepacket_observables(
    recon: dict[str, np.ndarray],
    params: FigureParams,
    out_dir: Path,
    start_point: tuple[float, float],
    filename: str,
    *,
    title: str,
    ylim: tuple[float, float] | None = None,
) -> dict[str, object]:
    tgrid = recon["tgrid"]
    requested_u, requested_v = start_point
    start_u, start_v, actual_u, actual_v = wavepacket_start_indices(tgrid, requested_u, requested_v)
    if not recon["is_legit"][start_u, start_v]:
        raise ValueError(
            f"wavepacket point ({actual_u:g}, {actual_v:g}) is not inside a valid reconstructed diamond"
        )
    start_time_actual = 0.5 * (actual_u + actual_v)
    start_depth_actual = 0.5 * (actual_v - actual_u)
    is_left = wavepacket_is_left(params)
    initial_state = np.array([1.0, 0.0], dtype=np.complex128)

    R_links, L_links = brl.simulate_wavepacket(
        recon["gates"],
        recon["N"],
        recon["NT"],
        start_u=start_u,
        start_v=start_v,
        initial_state=initial_state,
        is_left=is_left,
    )
    intensity, z_component, z_fine, t_fine = wavepacket_observable_grids(R_links, L_links, recon)
    z_max = float(np.nanmax(recon["ZZ"])) * 1.05
    if ylim is None:
        y_min, y_max = float(tgrid[0]), float(tgrid[-1])
    else:
        y_min, y_max = ylim
    # The manuscript uses polarization only; retain the original Z scale and data.
    fig_width = 5.8
    fig_height = max(4.8, fig_width * (y_max - y_min) / max(z_max, params.dt))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    im_z = ax.pcolormesh(
        z_fine, t_fine, z_component, shading="auto",
        cmap="RdBu_r", vmin=-1.0, vmax=1.0,
    )
    ax.axhline(0.0, color="white", linestyle="--", linewidth=0.9)
    ax.axhline(params.tev, color="white", linestyle=":", linewidth=0.9)
    ax.plot(start_depth_actual, start_time_actual, "o", ms=5, color="cyan", mec="black", mew=0.6)
    ax.set_xlim(0.0, z_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"bulk depth $z$")
    ax.set_ylabel(r"bulk time $t$")
    ax.set_title(title + "\n" + r"$Z$ component")
    fig.colorbar(
        im_z, ax=ax, label=r"$\langle \psi | Z | \psi \rangle$",
        fraction=0.04, pad=0.03, shrink=0.5, aspect=14,
    )
    path = save_figure(fig, out_dir, filename)

    return {
        "file": path,
        "start_u_index": int(start_u),
        "start_v_index": int(start_v),
        "start_u": actual_u,
        "start_v": actual_v,
        "start_time": start_time_actual,
        "start_depth": start_depth_actual,
        "requested_start_u": float(requested_u),
        "requested_start_v": float(requested_v),
        "mover": "left" if is_left else "right",
        "max_intensity": float(np.nanmax(intensity)),
        "max_abs_z_component": float(np.nanmax(np.abs(z_component))),
    }


def plot_formation_wavepacket_figures(
    formation_recon: dict[str, np.ndarray],
    late_recon: dict[str, np.ndarray],
    params: FigureParams,
    out_dir: Path,
) -> dict[str, object]:
    early_point, late_point = default_wavepacket_points(params)
    late_point = require_post_evaporation_bulk_point(late_point, params)
    return {
        "fig3e": plot_wavepacket_observables(
            formation_recon,
            params,
            out_dir,
            early_point,
            "fig3e_wavepacket_early.png",
            title=r"Early injection",
        ),
        "fig3f": plot_wavepacket_observables(
            late_recon,
            params,
            out_dir,
            late_point,
            "fig3f_wavepacket_late.png",
            title=r"Post-evaporation injection",
            ylim=(0.0, float(late_recon["tgrid"][-1])),
        ),
    }


def generate_figure2(params: FigureParams, out_dir: Path) -> dict[str, object]:
    """Regenerate the annotated gate map and Z-only wavepackets for Figure 2.

    The formation gate map and hand-drawn Penrose diagrams are source assets.
    Legacy fig3 filenames are retained for compatibility with older drafts.
    """
    from scripts.generate_figure2 import assemble_figure2

    recon = build_large_p_reconstruction(params)
    formation_recon = build_formation_reconstruction(params)
    plot_gate_color_map(recon, params, out_dir, "fig3a_gate_color_map.png",
                        title="HKLL gate transfer strength", draw_regions=True)
    # Preserve the manuscript's gate maps, including when writing to another directory.
    import shutil
    for name in ("fig3c_formation_gate_color_map.png",):
        target = out_dir / name
        if not target.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PAPER_FIG_DIR / name, target)
    wavepackets = plot_formation_wavepacket_figures(formation_recon, recon, params, out_dir)
    return {"files": assemble_figure2(out_dir), "wavepackets": wavepackets}



def linecut_times_from_wavepacket_defaults(params: FigureParams) -> tuple[float, float]:
    early_point, late_point = default_wavepacket_points(params)
    return 0.5 * (early_point[0] + early_point[1]), 0.5 * (late_point[0] + late_point[1])


def branch_continuous_wightman_linecut(
    t1: float,
    t2_grid: np.ndarray,
    params: FigureParams,
    eta_pre_g_func,
    chi_g_func,
) -> tuple[np.ndarray, np.ndarray]:
    g_R_values = []
    g_L_values = []
    for t2 in t2_grid:
        g_R, g_L = eta_large_p_g_from_sources(
            float(t1),
            float(t2),
            t_ev=params.tev,
            eta_pre_g_func=eta_pre_g_func,
            chi_g_func=chi_g_func,
            mu_eff=params.mu,
        )
        g_R_values.append(g_R)
        g_L_values.append(g_L)

    g_R_line = np.asarray(g_R_values, dtype=np.complex128)
    g_L_line = np.asarray(g_L_values, dtype=np.complex128)
    g_R_line = np.real(g_R_line) + 1j * np.unwrap(np.imag(g_R_line))
    g_L_line = np.real(g_L_line) + 1j * np.unwrap(np.imag(g_L_line))
    return -0.5j * np.exp(g_R_line / params.p), -0.5 * np.exp(g_L_line / params.p)


def plot_formation_two_point_linecuts(
    formation_recon: dict[str, np.ndarray],
    params: FigureParams,
    out_dir: Path,
) -> dict[str, object]:
    tgrid = formation_recon["tgrid"]
    eta_pre_g_func = formation_recon["eta_pre_g_func"]
    chi_g_func = formation_recon["chi_g_func"]
    early_time, late_time = linecut_times_from_wavepacket_defaults(params)
    panels = [
        ("early", early_time),
        ("late", late_time),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.4, 7.3),
        sharex=True,
        constrained_layout=True,
    )
    fixed_times: dict[str, float] = {}
    for col, (label, target_time) in enumerate(panels):
        t1_idx = nearest_time_index(tgrid, target_time)
        fixed_t1 = float(tgrid[t1_idx])
        fixed_times[label] = fixed_t1
        G_RR_line, G_RL_line = branch_continuous_wightman_linecut(
            fixed_t1,
            tgrid,
            params,
            eta_pre_g_func,
            chi_g_func,
        )

        im_ax = axes[0, col]
        abs_ax = axes[1, col]
        im_ax.plot(
            tgrid,
            np.imag(G_RR_line),
            color="tab:blue",
            lw=1.8,
            label=r"$\mathrm{Im}\,G^{\eta,>}_{RR}$",
        )
        im_ax.plot(
            tgrid,
            np.imag(G_RL_line),
            color="tab:orange",
            lw=1.8,
            label=r"$\mathrm{Im}\,G^{\eta,>}_{RL}$",
        )
        abs_ax.plot(
            tgrid,
            np.abs(G_RR_line),
            color="tab:blue",
            lw=1.8,
            label=r"$|G^{\eta,>}_{RR}|$",
        )
        abs_ax.plot(
            tgrid,
            np.abs(G_RL_line),
            color="tab:orange",
            lw=1.8,
            label=r"$|G^{\eta,>}_{RL}|$",
        )

        for ax in (im_ax, abs_ax):
            ax.axvline(fixed_t1, color="0.25", linestyle="--", linewidth=0.9)
            ax.axvline(0.0, color="0.55", linestyle="--", linewidth=0.8)
            ax.axvline(params.tev, color="0.55", linestyle=":", linewidth=0.8)
            ax.set_xlim(float(tgrid[0]), float(tgrid[-1]))
            ax.grid(True, linestyle=":", alpha=0.35)

        im_ax.set_title(rf"{label.capitalize()} fixed $t_1={fixed_t1:g}$")
        abs_ax.set_xlabel(r"$t_2$")

    axes[0, 0].set_ylabel(r"$\mathrm{Im}\,G^{\eta,>}(t_1,t_2)$")
    axes[1, 0].set_ylabel(r"$|G^{\eta,>}(t_1,t_2)|$")
    axes[0, 1].legend(loc="best", frameon=False, fontsize=8)
    axes[1, 1].legend(loc="best", frameon=False, fontsize=8)
    path = save_figure(fig, out_dir, "fig6_formation_two_point_linecuts.png")
    return {"file": path, "fixed_t1": fixed_times}


def kernel_row_sizes(K_local: np.ndarray, DeltaN_time: np.ndarray, N: int, tol: float = 1e-14) -> np.ndarray:
    NT = DeltaN_time.shape[0]
    V = K_local.reshape(N, NT, N)
    support = np.any(np.abs(V) > tol, axis=(0, 2))
    if not np.any(support):
        return np.full(N, np.nan)
    V_sub = V[:, support, :]
    D_sub = DeltaN_time[np.ix_(support, support)]
    return np.real(np.einsum("rta,ts,rsa->r", np.conj(V_sub), D_sub, V_sub))


def bulk_operator_size_matrix(
    K_all: np.ndarray,
    is_legit: np.ndarray,
    DeltaN_time: np.ndarray,
    N: int,
) -> np.ndarray:
    NT = DeltaN_time.shape[0]
    size_mat = np.full((NT, NT), np.nan)
    for u in range(NT):
        for v in range(u, NT):
            if u != v and not is_legit[u, v]:
                continue
            row_sizes = kernel_row_sizes(K_all[u, v], DeltaN_time, N)
            size_mat[u, v] = np.nanmean(row_sizes)
    return size_mat


def choose_fixed_v_index(recon: dict[str, np.ndarray], params: FigureParams, *size_mats: np.ndarray) -> int:
    tgrid = recon["tgrid"]
    after_evap = np.where(tgrid > params.tev + 1e-12)[0]
    if after_evap.size == 0:
        raise ValueError("no boundary time grid points after t_ev")

    pre_u = np.where(tgrid <= params.tev + 1e-12)[0]
    min_count = min(5, len(pre_u))

    def count_common_finite(v_idx: int) -> int:
        u_idx = pre_u[pre_u <= v_idx]
        if u_idx.size == 0:
            return 0
        finite = np.ones(u_idx.shape, dtype=bool)
        for mat in size_mats:
            finite &= np.isfinite(mat[u_idx, v_idx])
        return int(np.count_nonzero(finite))

    if params.size_fixed_v_time is not None:
        if params.size_fixed_v_time <= params.tev:
            raise ValueError("size_fixed_v_time must be greater than t_ev")
        valid_v = [
            int(v_idx)
            for v_idx in after_evap
            if count_common_finite(int(v_idx)) >= min_count
        ]
        candidates = valid_v if valid_v else [int(v_idx) for v_idx in after_evap]
        chosen = min(candidates, key=lambda idx: abs(tgrid[idx] - params.size_fixed_v_time))
        if count_common_finite(chosen) == 0:
            raise ValueError("requested size_fixed_v_time has no finite line-cut data")
        return int(chosen)

    preferred_time = params.tev + min(2.0, 0.25 * max(tgrid[-1] - params.tev, params.dt))
    preferred = int(after_evap[np.argmin(np.abs(tgrid[after_evap] - preferred_time))])

    ordered = sorted(after_evap, key=lambda idx: (abs(idx - preferred), idx))
    for v_idx in ordered:
        if count_common_finite(int(v_idx)) >= min_count:
            return int(v_idx)

    best_v = int(after_evap[0])
    best_count = -1
    for v_idx in after_evap:
        count = count_common_finite(int(v_idx))
        if count >= best_count:
            best_count = count
            best_v = int(v_idx)
    return int(best_v)


def fixed_v_line_coordinates(
    tgrid: np.ndarray,
    v_idx: int,
    *,
    u_max: float | None = None,
    data: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    u_idx = np.arange(v_idx + 1)
    valid = np.ones(u_idx.shape, dtype=bool)
    if u_max is not None:
        valid &= tgrid[u_idx] <= u_max + 1e-12
    if data is not None:
        valid &= np.isfinite(data[u_idx, v_idx])
    u_vals = tgrid[u_idx[valid]]
    v_val = tgrid[v_idx]
    z_vals = 0.5 * (v_val - u_vals)
    t_vals = 0.5 * (v_val + u_vals)
    return z_vals, t_vals


def draw_fixed_v_line(
    ax,
    tgrid: np.ndarray,
    v_idx: int,
    *,
    u_max: float | None = None,
    data: np.ndarray | None = None,
) -> None:
    z_vals, t_vals = fixed_v_line_coordinates(
        tgrid,
        v_idx,
        u_max=u_max,
        data=data,
    )
    if z_vals.size == 0:
        return
    ax.plot(z_vals, t_vals, "--", color="black", linewidth=2.2, alpha=0.8)
    ax.plot(z_vals, t_vals, "--", color="white", linewidth=1.2)


def plot_bulk_size_heatmap(
    recon: dict[str, np.ndarray],
    params: FigureParams,
    out_dir: Path,
    size_mat: np.ndarray,
    filename: str,
    *,
    title: str,
    colorbar_label: str,
    fixed_v_idx: int | None = None,
) -> str:
    display = np.where(
        np.isfinite(size_mat),
        np.clip(size_mat, 0.0, params.size_cutoff),
        np.nan,
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    im = ax.pcolormesh(
        recon["ZZ"],
        recon["TT"],
        display,
        shading="auto",
        cmap="cividis",
        vmin=0.0,
        vmax=params.size_cutoff,
    )
    ax.axhline(params.tev, color="white", linestyle=":", linewidth=0.9)
    if fixed_v_idx is not None:
        draw_fixed_v_line(
            ax,
            recon["tgrid"],
            fixed_v_idx,
            u_max=params.tev,
            data=size_mat,
        )
    ax.set_xlabel(r"bulk depth $z$")
    ax.set_ylabel(r"bulk time $t$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_label + rf" (clipped at {params.size_cutoff:g})")
    return save_figure(fig, out_dir, filename)


def plot_fixed_v_size_cut(
    recon: dict[str, np.ndarray],
    params: FigureParams,
    out_dir: Path,
    chi_size: np.ndarray,
    eta_size: np.ndarray,
    v_idx: int,
    filename: str,
    *,
    title_prefix: str,
) -> str:
    tgrid = recon["tgrid"]
    u_idx = np.arange(v_idx + 1)
    valid = (
        (tgrid[u_idx] <= params.tev + 1e-12)
        & (np.isfinite(chi_size[u_idx, v_idx]) | np.isfinite(eta_size[u_idx, v_idx]))
    )
    u_idx = u_idx[valid]

    fig, ax = plt.subplots(figsize=(5.8, 4.1), constrained_layout=True)
    ax.plot(
        tgrid[u_idx],
        chi_size[u_idx, v_idx],
        "o-",
        lw=1.5,
        ms=4,
        label=r"$\Delta L_\chi$",
    )
    ax.plot(
        tgrid[u_idx],
        eta_size[u_idx, v_idx],
        "s--",
        lw=1.5,
        ms=4,
        label=r"$\Delta L_\eta$",
    )
    ax.axvline(params.tev, color="black", linestyle=":", linewidth=0.9)
    ax.set_xlabel(r"boundary coordinate $u$")
    ax.set_ylabel(r"operator size")
    ax.set_title(rf"{title_prefix} fixed-$v$ cut, $v={tgrid[v_idx]:.2f}$")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend()
    return save_figure(fig, out_dir, filename)


def plot_bulk_operator_sizes(recon: dict[str, np.ndarray], params: FigureParams, out_dir: Path) -> dict[str, object]:
    tgrid = recon["tgrid"]
    tf_size = float(tgrid[-1])
    DeltaNchi = np.real(
        compute_size_operator_DeltaNchi_grid(tgrid, tf_size, params.large_p_dict(), invalid_value=0.0 + 0.0j)
    )
    DeltaNeta = np.real(
        compute_size_operator_DeltaNeta_grid(tgrid, tf_size, params.large_p_dict(), invalid_value=0.0 + 0.0j)
    )

    chi_L = bulk_operator_size_matrix(recon["KL_all"], recon["is_legit"], DeltaNchi, recon["N"])
    chi_R = bulk_operator_size_matrix(recon["KR_all"], recon["is_legit"], DeltaNchi, recon["N"])
    eta_L = bulk_operator_size_matrix(recon["KL_all"], recon["is_legit"], DeltaNeta, recon["N"])
    eta_R = bulk_operator_size_matrix(recon["KR_all"], recon["is_legit"], DeltaNeta, recon["N"])

    v_idx = choose_fixed_v_index(recon, params, chi_L, chi_R, eta_L, eta_R)

    files = {
        "chi_left_heatmap": plot_bulk_size_heatmap(
            recon,
            params,
            out_dir,
            chi_L,
            "fig4_chi_size_left_heatmap.png",
            title=r"Bulk $\chi$-size: left mover",
            colorbar_label=r"$\Delta L_\chi$",
            fixed_v_idx=v_idx,
        ),
        "chi_right_heatmap": plot_bulk_size_heatmap(
            recon,
            params,
            out_dir,
            chi_R,
            "fig4_chi_size_right_heatmap.png",
            title=r"Bulk $\chi$-size: right mover",
            colorbar_label=r"$\Delta L_\chi$",
            fixed_v_idx=v_idx,
        ),
        "eta_left_heatmap": plot_bulk_size_heatmap(
            recon,
            params,
            out_dir,
            eta_L,
            "fig4_eta_size_left_heatmap.png",
            title=r"Bulk $\eta$-size: left mover",
            colorbar_label=r"$\Delta L_\eta$",
            fixed_v_idx=v_idx,
        ),
        "eta_right_heatmap": plot_bulk_size_heatmap(
            recon,
            params,
            out_dir,
            eta_R,
            "fig4_eta_size_right_heatmap.png",
            title=r"Bulk $\eta$-size: right mover",
            colorbar_label=r"$\Delta L_\eta$",
            fixed_v_idx=v_idx,
        ),
        "left_fixed_v_cut": plot_fixed_v_size_cut(
            recon,
            params,
            out_dir,
            chi_L,
            eta_L,
            v_idx,
            "fig4_left_mover_fixed_v_cut.png",
            title_prefix="Left mover",
        ),
        "right_fixed_v_cut": plot_fixed_v_size_cut(
            recon,
            params,
            out_dir,
            chi_R,
            eta_R,
            v_idx,
            "fig4_right_mover_fixed_v_cut.png",
            title_prefix="Right mover",
        ),
    }

    return {"fixed_v_index": int(v_idx), "fixed_v_time": float(tgrid[v_idx]), "files": files}


def build_formation_reconstruction(params: FigureParams) -> dict[str, np.ndarray]:
    print(
        "formation reconstruction: "
        f"beta={params.formation_beta}, t0={params.formation_t0}, "
        f"tmax={params.tmax}, dt={params.dt}"
    )
    source_tgrid, psi, Dpsi = generate_black_hole_formation_profile(
        params.formation_beta,
        params.formation_t0,
        params.tev,
        params.dt,
    )
    source_grids = coupled_syk_g_grids(psi, Dpsi, params.p, coupled_syk_2pt=coupledSYK2pt)
    formation_g_source = make_interpolating_g_source(
        source_tgrid,
        source_grids["g_R_grid"],
        source_grids["g_L_grid"],
        bounds_error=False,
    )

    def chi_g_source(t1, t2):
        return chi_large_p_g_real_time(t1, t2, cal_J=params.chi_cal_j, mu=params.mu)

    full_tgrid = uniform_time_grid(params.formation_t0, params.tmax, params.dt)
    T1, T2 = np.meshgrid(full_tgrid, full_tgrid, indexing="ij")
    g_R_eta, g_L_eta = eta_large_p_g_from_sources(
        T1,
        T2,
        t_ev=params.tev,
        eta_pre_g_func=formation_g_source,
        chi_g_func=chi_g_source,
        mu_eff=params.mu,
    )

    GRR = np.exp(g_R_eta / params.p)
    GRL = np.exp(g_L_eta / params.p)
    CorM = correlation_matrix_from_exp_g(GRR, GRL)
    recon = run_bulk_reconstruction_from_correlator(
        CorM,
        full_tgrid,
        label="formation-driven eta",
        bulk_reconstruction_lib=brl,
    )
    recon["TT"], recon["ZZ"] = reconstruction_coordinates(full_tgrid)
    recon["GRR"] = GRR
    recon["GRL"] = GRL
    recon["eta_pre_g_func"] = formation_g_source
    recon["chi_g_func"] = chi_g_source
    return recon


def find_two_replica_run(params: FigureParams) -> tuple[Path, Path]:
    runs_root = NUMERICS_RESULTS_DIR / "eta_two_replica" / "runs"
    prefix = (
        f"mu_{slug_float(params.mu, 2)}"
        f"_p_{int(params.p)}"
        f"_dt_{slug_float(params.dt, 2)}"
        f"_tev_{slug_float(params.tev, 2)}"
        f"_tmax_{slug_float(params.mi_tf, 2)}"
    )
    candidates = sorted(
        [path for path in runs_root.glob(prefix + "*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no two-replica run directory found with prefix {prefix}")
    run_dir = candidates[-1]
    manifest_path = run_dir / "two_point_functions" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return run_dir, manifest_path


def load_two_replica_payload(params: FigureParams):
    import pandas as pd
    import torch

    run_dir, manifest_path = find_two_replica_run(params)
    manifest = pd.read_csv(manifest_path)
    tf_col = "t_f" if "t_f" in manifest.columns else "tf"
    row = manifest.iloc[np.argmin(np.abs(manifest[tf_col].to_numpy(float) - params.mi_tf))]
    payload_path = manifest_path.parent / row["file"]
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    print(f"loaded two-replica payload {payload_path}")
    return run_dir, payload_path, payload


def compute_renyi2_mi_heatmaps(params: FigureParams):
    from application.bulk_mi import (
        build_hkll_kernels_from_disconnected_eta,
        bulk_mutual_information_from_kernel,
        real_time_grid_from_meta,
        select_bulk_kernel,
    )

    run_dir, payload_path, payload = load_two_replica_payload(params)
    meta = ContourMeta(**payload["contour"])
    G_disc = payload["G_disc"].cpu().numpy()
    G_twisted = payload["G_twisted"].cpu().numpy()

    hkll = build_hkll_kernels_from_disconnected_eta(G_disc, meta, replica_index=1, N_bulk=2)
    KR_all = hkll["KR_all"]
    KL_all = hkll["KL_all"]
    is_legit = hkll["is_legit"]
    tgrid = real_time_grid_from_meta(meta)
    NT = meta.N_t

    Icl = np.full((NT, NT), np.nan)
    Iq = np.full((NT, NT), np.nan)
    Knorm = np.full((NT, NT), np.nan)

    for u in range(NT):
        for v in range(u, NT):
            t_bulk = 0.5 * (tgrid[u] + tgrid[v])
            z_bulk = 0.5 * (tgrid[v] - tgrid[u])
            try:
                K_info = select_bulk_kernel(
                    KR_all,
                    KL_all,
                    is_legit,
                    meta,
                    t_bulk=float(t_bulk),
                    z_bulk=float(z_bulk),
                    direction=params.mi_direction,
                    r=params.mi_component,
                    use_boundary_delta_at_z0=True,
                )
                Knorm[u, v] = K_info["K_norm"]
                if K_info["K_norm"] > params.mi_kernel_norm_cutoff:
                    continue
                mi = bulk_mutual_information_from_kernel(G_twisted, meta, K_info["K"])
                Icl[u, v] = mi["I2_classical"]
                Iq[u, v] = mi["I2_quantum"]
            except Exception:
                continue

    Tu, Tv = np.meshgrid(tgrid, tgrid, indexing="ij")
    TT = 0.5 * (Tu + Tv)
    ZZ = 0.5 * (Tv - Tu)
    return {
        "run_dir": run_dir,
        "payload_path": payload_path,
        "meta": meta,
        "tgrid": tgrid,
        "TT": TT,
        "ZZ": ZZ,
        "Icl": Icl,
        "Iq": Iq,
        "Knorm": Knorm,
    }


def plot_renyi2_heatmap(
    mi_data: dict[str, np.ndarray],
    out_dir: Path,
    filename: str,
    *,
    data_key: str,
    vmax: float,
    label: str,
    title: str,
    params: FigureParams | None = None,
) -> str:
    data = mi_data[data_key]
    finite = data[np.isfinite(data)]
    vmin = 0.0 if finite.size == 0 else min(0.0, float(np.nanmin(finite)))
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_over("red")

    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    im = ax.pcolormesh(
        mi_data["ZZ"],
        mi_data["TT"],
        data,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if params is not None:
        for v_idx in choose_mi_fixed_v_indices(mi_data, params):
            draw_fixed_v_line(
                ax,
                mi_data["tgrid"],
                v_idx,
                u_max=params.tev,
                data=data,
            )
    ax.set_xlabel(r"bulk depth $z$")
    ax.set_ylabel(r"bulk time $t$")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, extend="max")
    cb.set_label(label)
    return save_figure(fig, out_dir, filename)


def choose_mi_fixed_v_indices(
    mi_data: dict[str, np.ndarray],
    params: FigureParams,
    *,
    data_keys: tuple[str, ...] = ("Iq", "Icl"),
) -> list[int]:
    tgrid = mi_data["tgrid"]
    NT = len(tgrid)
    after_evap = np.where(tgrid > params.tev + 1e-12)[0]
    if after_evap.size == 0:
        raise ValueError("no MI grid points after t_ev")

    pre_u = np.where(tgrid <= params.tev + 1e-12)[0]
    min_count = min(4, max(1, len(pre_u)))

    valid_v: list[int] = []
    for v_idx in after_evap:
        u_idx = pre_u[pre_u <= v_idx]
        if u_idx.size == 0:
            continue
        finite = np.ones(u_idx.shape, dtype=bool)
        for key in data_keys:
            finite &= np.isfinite(mi_data[key][u_idx, v_idx])
        if np.count_nonzero(finite) >= min_count:
            valid_v.append(int(v_idx))

    if not valid_v:
        raise ValueError("no fixed-v MI cuts have enough finite points")

    if params.mi_fixed_v_times:
        chosen: list[int] = []
        for target_time in params.mi_fixed_v_times:
            nearest = min(valid_v, key=lambda idx: abs(tgrid[idx] - target_time))
            if nearest not in chosen:
                chosen.append(nearest)
        return chosen

    n_cuts = max(1, min(params.mi_num_fixed_v_cuts, len(valid_v)))
    if n_cuts == 1:
        return [valid_v[len(valid_v) // 2]]

    positions = np.linspace(0, len(valid_v) - 1, n_cuts)
    chosen = []
    for pos in positions:
        idx = valid_v[int(round(pos))]
        if idx not in chosen:
            chosen.append(idx)
    return chosen


def plot_renyi2_fixed_v_cuts(
    mi_data: dict[str, np.ndarray],
    params: FigureParams,
    out_dir: Path,
    *,
    data_key: str,
    filename: str,
    label: str,
    title: str,
    ymax: float,
    max_label=None,
) -> str:
    tgrid = mi_data["tgrid"]
    data = mi_data[data_key]
    v_indices = choose_mi_fixed_v_indices(mi_data, params)

    fig, ax = plt.subplots(figsize=(5.8, 4.1), constrained_layout=True)
    for v_idx in v_indices:
        u_idx = np.arange(v_idx + 1)
        valid = (tgrid[u_idx] <= params.tev + 1e-12) & np.isfinite(data[u_idx, v_idx])
        u_idx = u_idx[valid]
        if u_idx.size == 0:
            continue
        ax.plot(
            tgrid[u_idx],
            data[u_idx, v_idx],
            "o-",
            lw=1.5,
            ms=3.5,
            label=rf"$v={tgrid[v_idx]:.2f}$",
        )

    if max_label is not None:
        ax.axhline(ymax, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label=max_label)
    ax.axvline(params.tev, color="black", linestyle=":", linewidth=0.9)
    ax.set_xlabel(r"boundary coordinate $u$")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.set_ylim(bottom=0.0, top=1.05 * ymax)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(title="fixed boundary time")
    return save_figure(fig, out_dir, filename)


def plot_renyi2_fixed_v_line_cuts(
    mi_data: dict[str, np.ndarray],
    params: FigureParams,
    out_dir: Path,
) -> dict[str, object]:
    v_indices = choose_mi_fixed_v_indices(mi_data, params)
    tgrid = mi_data["tgrid"]
    files = {
        "quantum": plot_renyi2_fixed_v_cuts(
            mi_data,
            params,
            out_dir,
            data_key="Iq",
            filename="fig5a_renyi2_quantum_fixed_v_cuts.png",
            label=r"$I^{(2)}_{\rm q}$",
            title=r"Second Renyi quantum mutual information, fixed-$v$ cuts",
            ymax=2.0 * np.log(2.0),
            max_label=r"$2\log 2$",
        ),
        "classical": plot_renyi2_fixed_v_cuts(
            mi_data,
            params,
            out_dir,
            data_key="Icl",
            filename="fig5a_renyi2_classical_fixed_v_cuts.png",
            label=r"$I^{(2)}_{\rm cl}$",
            title=r"Second Renyi classical mutual information, fixed-$v$ cuts",
            ymax=np.log(2.0),
            max_label=r"$\log 2$",
        ),
    }
    return {
        "fixed_v_indices": [int(idx) for idx in v_indices],
        "fixed_v_times": [float(tgrid[idx]) for idx in v_indices],
        "files": files,
    }


########### Things in this band added by Shoaib ##########
def compute_renyi2_action_curve(params: FigureParams) -> dict[str, object]:
    """Compute S_2^eta / N_eta from the saved two-replica action difference."""

    import pandas as pd
    import torch

    from finite_p.action import eta_action_difference

    run_dir, manifest_path = find_two_replica_run(params)
    manifest = pd.read_csv(manifest_path)

    tf_col = "t_f" if "t_f" in manifest.columns else "tf"
    file_col = "file" if "file" in manifest.columns else "filename"

    rows = []

    for _, row in manifest.sort_values(tf_col).iterrows():
        payload_path = manifest_path.parent / row[file_col]
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)

        meta = ContourMeta(**payload["contour"])

        G_chi = payload["G_chi"].cpu().numpy()
        G_disc = payload["G_disc"].cpu().numpy()
        G_twisted = payload["G_twisted"].cpu().numpy()
        w = payload["w"].cpu().numpy() if "w" in payload else None

        action_row = eta_action_difference(
            G_chi,
            G_disc,
            G_twisted,
            meta,
            cal_J=params.chi_cal_j,
            p=int(params.p),
            t_ev=params.tev,
            a_param=params.a,
            w=w,
        )

        action_row["file"] = row[file_col]
        rows.append(action_row)

    action_table = pd.DataFrame(rows).sort_values("t_f").reset_index(drop=True)

    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "action_table": action_table,
    }


def plot_renyi2_action_difference(
    action_data: dict[str, object],
    params: FigureParams,
    out_dir: Path,
) -> str:
    """Plot manuscript Figure 4(b): second Renyi entropy."""

    action_table = action_data["action_table"]

    fig, ax = plt.subplots(figsize=(5.8, 4.1), constrained_layout=True)

    ax.plot(
        action_table["t_f"],
        action_table["S2_over_Neta"],
        "o-",
        lw=1.5,
        ms=4,
        label=r"$S_2^\eta/N_\eta$",
    )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    ax.axhline(np.log(2.0), color="gray", linestyle="--", linewidth=1.0, label=r"$\log 2$")
    ax.axvline(params.tev, color="black", linestyle="--", linewidth=1.0, alpha=0.75, label=r"$t_{\rm ev}$")

    ax.set_xlabel(r"$t_f$")
    ax.set_ylabel(r"$S_2^\eta/N_\eta$")
    ax.set_title(r"Second Renyi entropy")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=8)

    return save_figure(fig, out_dir, "fig5b_renyi2_action_difference.png")


###### code for p=4 ######
@dataclass
class P4FigureParams:
    """Parameters for Fig. 7 p=4 saved-data figures."""

    mu: float = 0.30
    p: int = 4
    dt: float = 0.10
    tev: float = 1.00
    run_tmax: float = 15.00
    run_stamp: str = "20260630_203051"

    cal_J: float = 0.5
    a_param: float = 1.0

    fixed_u: float = 0.5
    fixed_v: float = 1.5
    fixed_iq_tf_min: float = 1.5
    tf_max: float = 15.0

    heatmap_tf: float = 11.00
    direction: str = "R"
    component: int = 0
    kernel_norm_cutoff: float = 100.0


def find_p4_two_replica_run(params: P4FigureParams) -> tuple[Path, Path]:
    runs_root = NUMERICS_RESULTS_DIR / "eta_two_replica" / "runs"

    prefix = (
        f"mu_{slug_float(params.mu, 2)}"
        f"_p_{int(params.p)}"
        f"_dt_{slug_float(params.dt, 2)}"
        f"_tev_{slug_float(params.tev, 2)}"
        f"_tmax_{slug_float(params.run_tmax, 2)}"
    )

    run_dir = runs_root / f"{prefix}_{params.run_stamp}"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing p=4 run directory: {run_dir}")

    manifest_path = run_dir / "two_point_functions" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing p=4 manifest: {manifest_path}")

    return run_dir, manifest_path


def p4_manifest_columns(manifest):
    tf_col = "t_f" if "t_f" in manifest.columns else "tf"
    file_col = "file" if "file" in manifest.columns else "filename"
    return tf_col, file_col


def load_p4_payload_at_tf(params: P4FigureParams, target_tf: float):
    import pandas as pd
    import torch

    run_dir, manifest_path = find_p4_two_replica_run(params)
    manifest = pd.read_csv(manifest_path)

    tf_col, file_col = p4_manifest_columns(manifest)
    tf_values = manifest[tf_col].to_numpy(float)

    row_idx = int(np.argmin(np.abs(tf_values - target_tf)))
    actual_tf = float(tf_values[row_idx])

    if abs(actual_tf - target_tf) > 1e-9:
        raise FileNotFoundError(
            f"no p=4 payload at t_f={target_tf}; nearest available t_f is {actual_tf}"
        )

    payload_path = manifest_path.parent / manifest.iloc[row_idx][file_col]
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)

    print(f"loaded p=4 payload {payload_path}")
    return run_dir, manifest_path, payload_path, payload


def grid_index_for_time(meta, tgrid: np.ndarray, value: float, name: str) -> int:
    idx = int(round(value / meta.dt)) - 1

    if idx < 0 or idx >= len(tgrid):
        raise ValueError(f"{name}={value} is outside the real-time grid")

    if abs(tgrid[idx] - value) > 1e-9:
        raise ValueError(
            f"{name}={value} is not on the dt={meta.dt} grid; nearest is {tgrid[idx]}"
        )

    return idx


def compute_p4_s2_curve(params: P4FigureParams):
    import pandas as pd
    import torch

    from finite_p.action import eta_action_difference

    run_dir, manifest_path = find_p4_two_replica_run(params)
    manifest = pd.read_csv(manifest_path)

    tf_col, file_col = p4_manifest_columns(manifest)
    rows = []

    for _, row in manifest.sort_values(tf_col).iterrows():
        tf = float(row[tf_col])
        if tf > params.tf_max + 1e-12:
            continue

        print(f"computing Fig. 7(a) p=4 S2 for t_f={tf:g}")

        payload_path = manifest_path.parent / row[file_col]
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)

        meta = ContourMeta(**payload["contour"])

        G_chi = payload["G_chi"].cpu().numpy()
        G_disc = payload["G_disc"].cpu().numpy()
        G_twisted = payload["G_twisted"].cpu().numpy()
        w = payload["w"].cpu().numpy() if "w" in payload else None

        action_row = eta_action_difference(
            G_chi,
            G_disc,
            G_twisted,
            meta,
            cal_J=params.cal_J,
            p=int(params.p),
            t_ev=params.tev,
            a_param=params.a_param,
            w=w,
        )

        rows.append(action_row)

    return pd.DataFrame(rows).sort_values("t_f").reset_index(drop=True)


def compute_p4_fixed_point_iq_curve(params: P4FigureParams):
    import pandas as pd
    import torch

    from application.bulk_mi import (
        build_hkll_kernels_from_disconnected_eta,
        bulk_mutual_information_from_kernel,
        real_time_grid_from_meta,
    )

    run_dir, manifest_path = find_p4_two_replica_run(params)
    manifest = pd.read_csv(manifest_path)

    tf_col, file_col = p4_manifest_columns(manifest)
    rows = []

    for _, row in manifest.sort_values(tf_col).iterrows():
        tf = float(row[tf_col])

        if tf < params.fixed_iq_tf_min - 1e-12:
            continue
        if tf > params.tf_max + 1e-12:
            continue

        print(f"computing Fig. 7(a) p=4 fixed-point Iq for t_f={tf:g}")

        payload_path = manifest_path.parent / row[file_col]
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)

        meta = ContourMeta(**payload["contour"])
        G_disc = payload["G_disc"].cpu().numpy()
        G_twisted = payload["G_twisted"].cpu().numpy()

        try:
            if params.fixed_v <= params.fixed_u:
                raise ValueError(
                    f"need v > u, got u={params.fixed_u}, v={params.fixed_v}"
                )

            if params.fixed_v > meta.t_f + 1e-12:
                raise ValueError(
                    f"fixed v={params.fixed_v} is beyond this payload t_f={meta.t_f}"
                )

            hkll = build_hkll_kernels_from_disconnected_eta(
                G_disc,
                meta,
                replica_index=1,
                N_bulk=2,
            )

            KR_all = hkll["KR_all"]
            KL_all = hkll["KL_all"]
            is_legit = hkll["is_legit"]
            tgrid = real_time_grid_from_meta(meta)

            u_idx = grid_index_for_time(meta, tgrid, params.fixed_u, "u")
            v_idx = grid_index_for_time(meta, tgrid, params.fixed_v, "v")

            if not is_legit[u_idx, v_idx]:
                raise ValueError("HKLL point is not legitimate")

            if params.direction.upper() == "R":
                K = KR_all[u_idx, v_idx, params.component, :]
            elif params.direction.upper() == "L":
                K = KL_all[u_idx, v_idx, params.component, :]
            else:
                raise ValueError("direction must be 'R' or 'L'")

            K_norm = float(np.linalg.norm(K))

            if K_norm > params.kernel_norm_cutoff:
                raise ValueError(
                    f"K_norm={K_norm:g} exceeds cutoff={params.kernel_norm_cutoff:g}"
                )

            mi = bulk_mutual_information_from_kernel(G_twisted, meta, K)

            rows.append(
                {
                    "t_f": float(meta.t_f),
                    "I2_quantum": mi["I2_quantum"],
                    "K_norm": K_norm,
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "t_f": float(meta.t_f),
                    "I2_quantum": np.nan,
                    "K_norm": np.nan,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows).sort_values("t_f").reset_index(drop=True)


def plot_fig7a_p4_s2_and_fixed_iq(
    s2_table,
    iq_table,
    params: P4FigureParams,
    out_dir: Path,
) -> str:
    iq_finite = iq_table[np.isfinite(iq_table["I2_quantum"])].copy()

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    ax.plot(
        s2_table["t_f"],
        s2_table["S2_over_Neta"],
        "o-",
        lw=1.6,
        ms=4,
        color="black",
        label=r"$S_2^\eta/N_\eta$",
    )

    ax.plot(
        iq_finite["t_f"],
        iq_finite["I2_quantum"],
        "s-",
        lw=1.6,
        ms=4,
        color="tab:blue",
        label=rf"$I_q^{{(2)}}$, $u={params.fixed_u:g}$, $v={params.fixed_v:g}$",
    )

    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9)
    ax.axhline(np.log(2.0), color="gray", linestyle="--", linewidth=1.0, label=r"$\log 2$")
    ax.axhline(2.0 * np.log(2.0), color="gray", linestyle="-.", linewidth=1.0, label=r"$2\log 2$")
    ax.axvline(params.tev, color="black", linestyle="--", linewidth=1.0, alpha=0.75, label=r"$t_{\rm ev}$")

    ax.set_xlabel(r"$t_f$")
    ax.set_ylabel(r"second Renyi quantity")
    ax.set_title(
        rf"$p=4$: $S_2^\eta/N_\eta$ and $I_q^{{(2)}}$ "
        rf"at $u={params.fixed_u:g}$, $v={params.fixed_v:g}$"
    )
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=8)

    return save_figure(fig, out_dir, "fig7a_p4_s2_and_fixed_iq.png")


def compute_p4_right_mover_iq_heatmap(params: P4FigureParams) -> dict[str, object]:
    from application.bulk_mi import (
        build_hkll_kernels_from_disconnected_eta,
        bulk_mutual_information_from_kernel,
        real_time_grid_from_meta,
    )

    run_dir, manifest_path, payload_path, payload = load_p4_payload_at_tf(
        params,
        params.heatmap_tf,
    )

    meta = ContourMeta(**payload["contour"])
    G_disc = payload["G_disc"].cpu().numpy()
    G_twisted = payload["G_twisted"].cpu().numpy()

    hkll = build_hkll_kernels_from_disconnected_eta(
        G_disc,
        meta,
        replica_index=1,
        N_bulk=2,
    )

    KR_all = hkll["KR_all"]
    KL_all = hkll["KL_all"]
    is_legit = hkll["is_legit"]
    tgrid = real_time_grid_from_meta(meta)
    NT = meta.N_t

    Iq = np.full((NT, NT), np.nan)

    print(f"computing Fig. 7(b) p=4 right-mover Iq heatmap, t_f={meta.t_f:g}, NT={NT}")

    for u in range(NT):
        if u % 10 == 0:
            print(f"  row {u + 1}/{NT}")

        for v in range(u, NT):
            if not is_legit[u, v]:
                continue

            if params.direction.upper() == "R":
                K = KR_all[u, v, params.component, :]
            elif params.direction.upper() == "L":
                K = KL_all[u, v, params.component, :]
            else:
                raise ValueError("direction must be 'R' or 'L'")

            K = np.asarray(K, dtype=np.complex128)
            K_norm = float(np.linalg.norm(K))

            if K_norm > params.kernel_norm_cutoff:
                continue

            try:
                mi = bulk_mutual_information_from_kernel(G_twisted, meta, K)
                Iq[u, v] = mi["I2_quantum"]
            except Exception:
                continue

    Tu, Tv = np.meshgrid(tgrid, tgrid, indexing="ij")
    TT = 0.5 * (Tu + Tv)
    ZZ = 0.5 * (Tv - Tu)

    return {
        "TT": TT,
        "ZZ": ZZ,
        "Iq": Iq,
    }


def plot_fig7b_p4_right_mover_iq_heatmap(
    heatmap_data: dict[str, object],
    params: P4FigureParams,
    out_dir: Path,
) -> str:
    data = heatmap_data["Iq"]
    finite = data[np.isfinite(data)]
    vmin = 0.0 if finite.size == 0 else min(0.0, float(np.nanmin(finite)))

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_over("red")

    fig, ax = plt.subplots(figsize=(5.8, 4.7), constrained_layout=True)

    im = ax.pcolormesh(
        heatmap_data["ZZ"],
        heatmap_data["TT"],
        data,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=2.0 * np.log(2.0),
    )

    ax.axhline(params.tev, color="white", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.set_xlabel(r"bulk depth $z$")
    ax.set_ylabel(r"bulk time $t$")
    ax.set_title(rf"$p=4$ right-mover $I_q^{{(2)}}$, $t_f={params.heatmap_tf:g}$")

    cb = fig.colorbar(im, ax=ax, extend="max")
    cb.set_label(r"$I_q^{(2)}$")

    return save_figure(fig, out_dir, "fig7b_p4_right_mover_iq_heatmap.png")


def generate_fig7_p4_figures(out_dir: Path) -> dict[str, str]:
    params = P4FigureParams()

    s2_table = compute_p4_s2_curve(params)
    iq_table = compute_p4_fixed_point_iq_curve(params)

    fig7a = plot_fig7a_p4_s2_and_fixed_iq(
        s2_table,
        iq_table,
        params,
        out_dir,
    )

    heatmap_data = compute_p4_right_mover_iq_heatmap(params)
    fig7b = plot_fig7b_p4_right_mover_iq_heatmap(
        heatmap_data,
        params,
        out_dir,
    )

    return {
        "fig7a": fig7a,
        "fig7b": fig7b,
    }

######################




####################################################


def generate_figure4(out_dir: Path, *, source_dir: Path | None = None) -> dict[str, str]:
    """Assemble labeled manuscript Figure 4 from saved Renyi panels."""
    from scripts.generate_figure4 import assemble_figure4

    return assemble_figure4(out_dir, source_dir=source_dir)



def generate_all(params: FigureParams, out_dir: Path, *, skip_mi: bool = False) -> dict[str, object]:
    outputs: dict[str, object] = {"parameters": asdict(params), "files": {}}

    recon = build_large_p_reconstruction(params)
    outputs["files"]["fig2"] = plot_two_point_heatmap(recon, params, out_dir)
    outputs["files"]["fig3a"] = plot_gate_color_map(
        recon,
        params,
        out_dir,
        "fig3a_gate_color_map.png",
        title=r"HKLL gate transfer strength",
        draw_regions=True,
    )
    outputs["files"]["fig4"] = plot_bulk_operator_sizes(recon, params, out_dir)

    formation_recon = build_formation_reconstruction(params)
    outputs["files"]["fig3c"] = plot_gate_color_map(
        formation_recon,
        params,
        out_dir,
        "fig3c_formation_gate_color_map.png",
        title=r"Formation and evaporation gate transfer strength",
        draw_formation_lines=True,
    )
    formation_wavepackets = plot_formation_wavepacket_figures(formation_recon, recon, params, out_dir)
    outputs["files"]["fig3e"] = formation_wavepackets["fig3e"]["file"]
    outputs["files"]["fig3f"] = formation_wavepackets["fig3f"]["file"]
    outputs["formation_wavepackets"] = formation_wavepackets
    from scripts.generate_figure2 import assemble_figure2
    outputs["files"]["figure2_bulk_geometry"] = assemble_figure2(out_dir)
    fig6_linecuts = plot_formation_two_point_linecuts(formation_recon, params, out_dir)
    outputs["files"]["fig6"] = fig6_linecuts["file"]
    outputs["fig6_linecuts"] = fig6_linecuts

    if not skip_mi:
        try:
            mi_data = compute_renyi2_mi_heatmaps(params)
            fig5_quantum = plot_renyi2_heatmap(
                mi_data,
                out_dir,
                "fig5a_renyi2_quantum_mi_heatmap.png",
                data_key="Iq",
                vmax=2.0 * np.log(2.0),
                label=r"$I^{(2)}_{\rm q}$",
                title=r"Second Renyi quantum mutual information",
                params=params,
            )
            fig5_classical = plot_renyi2_heatmap(
                mi_data,
                out_dir,
                "fig5a_renyi2_classical_mi_heatmap.png",
                data_key="Icl",
                vmax=np.log(2.0),
                label=r"$I^{(2)}_{\rm cl}$",
                title=r"Second Renyi classical mutual information",
                params=params,
            )
            fig5_fixed_v_cuts = plot_renyi2_fixed_v_line_cuts(
                mi_data,
                params,
                out_dir,
            )
            outputs["files"]["fig5a"] = {
                "quantum_heatmap": fig5_quantum,
                "classical_heatmap": fig5_classical,
                "fixed_v_cuts": fig5_fixed_v_cuts,
            }
            outputs["two_replica_payload"] = display_path(mi_data["payload_path"])
        except Exception as exc:
            outputs["fig5a_error"] = str(exc)
            print(f"skipping the mutual-information panels (manuscript Figure 4(a)): {exc}")

        try:
            action_data = compute_renyi2_action_curve(params)
            fig5b_path = plot_renyi2_action_difference(action_data, params, out_dir)
            action_table = action_data["action_table"]
            outputs["files"]["fig5b"] = fig5b_path
            outputs["renyi2_action_summary"] = {
                "run_dir": display_path(action_data["run_dir"]),
                "manifest_path": display_path(action_data["manifest_path"]),
                "num_points": int(len(action_table)),
                "tf_min": float(action_table["t_f"].min()),
                "tf_max": float(action_table["t_f"].max()),
                "s2_over_Neta_min": float(action_table["S2_over_Neta"].min()),
                "s2_over_Neta_max": float(action_table["S2_over_Neta"].max()),
            }

        except Exception as exc:
            outputs["fig5b_error"] = str(exc)
            print(f"skipping the entropy panel (manuscript Figure 4(b)): {exc}")

    try:
        fig7_files = generate_fig7_p4_figures(out_dir)
        outputs["files"]["fig7"] = fig7_files
    except Exception as exc:
        outputs["fig7_error"] = str(exc)
        print(f"skipping Fig. 7: {exc}")
    


    if "fig5a" in outputs["files"] and "fig5b" in outputs["files"]:
        outputs["files"]["figure4_finite_p"] = generate_figure4(out_dir)

    manifest_path = out_dir / "generated_figures_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2) + "\n")
    print(f"wrote {manifest_path}")
    return outputs


def manuscript_parameters() -> dict[str, FigureParams]:
    """Presets matching the figure-parameter subsection of the current draft.

    formation_beta=10 sets the auxiliary epsilon; physical beta_eta=20 for
    the analytic correlators at J=0.5. The finite-p beta comes from the payload.
    """
    return {
        "geometry": FigureParams(),
        "operator_size": FigureParams(tev=6.0, mu=0.5, tmax=25.0,
                                     size_cutoff=100.0, size_fixed_v_time=8.0),
        "finite_p": FigureParams(tev=6.0, mu=0.5, tmax=25.0, mi_tf=25.0),
    }


def generate_manuscript_figures(out_dir: Path) -> dict[str, object]:
    """Reproduce the current manuscript with independent figure-group presets.

    Saved-data failures propagate instead of silently leaving stale figure files.
    """
    presets = manuscript_parameters()
    geometry = generate_figure2(presets["geometry"], out_dir)
    size_params = presets["operator_size"]
    recon = build_large_p_reconstruction(size_params)
    size = plot_bulk_operator_sizes(recon, size_params, out_dir)
    two_point = plot_two_point_heatmap(recon, size_params, out_dir)
    mi_params = presets["finite_p"]
    mi_data = compute_renyi2_mi_heatmaps(mi_params)
    plot_renyi2_heatmap(mi_data, out_dir, "fig5a_renyi2_quantum_mi_heatmap.png",
                       data_key="Iq", vmax=2.0 * np.log(2.0),
                       label=r"$I^{(2)}_{\rm q}$",
                       title="Second Renyi quantum mutual information", params=mi_params)
    action_data = compute_renyi2_action_curve(mi_params)
    plot_renyi2_action_difference(action_data, mi_params, out_dir)
    finite_p = generate_figure4(out_dir)
    result = {
        "parameters": {key: asdict(value) for key, value in presets.items()},
        "files": {"figure2": geometry["files"], "figure3": size["files"],
                  "figure4": finite_p, "figure5": two_point},
        "formation_wavepackets": geometry["wavepackets"],
        "size_fixed_v_time": size["fixed_v_time"],
        "finite_p_beta": mi_data["meta"].beta,
        "two_replica_payload": display_path(mi_data["payload_path"]),
    }
    (out_dir / "manuscript_figures_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--manuscript", action="store_true",
                        help="Generate current manuscript figures using their documented parameter presets.")
    parser.add_argument("--skip-mi", action="store_true", help="Skip the saved two-replica mutual-information and entropy panels (manuscript Figure 4).")
    parser.add_argument("--quick", action="store_true", help="Use a coarse grid for smoke testing.")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--size-fixed-v-time", type=float, default=None)
    parser.add_argument("--formation-t0", type=float, default=None)
    parser.add_argument("--formation-beta", type=float, default=None)
    parser.add_argument(
        "--mi-fixed-v-times",
        type=float,
        nargs="+",
        default=None,
        help="Fixed v times for auxiliary mutual-information line cuts (manuscript Figure 4(a) data).",
    )
    parser.add_argument("--mi-num-fixed-v-cuts", type=int, default=None)
    parser.add_argument("--wavepacket-early-time", type=float, default=None)
    parser.add_argument("--wavepacket-late-time", type=float, default=None)
    parser.add_argument("--wavepacket-depth", type=float, default=None)
    parser.add_argument("--wavepacket-early-u", type=float, default=None)
    parser.add_argument("--wavepacket-early-v", type=float, default=None)
    parser.add_argument("--wavepacket-late-u", type=float, default=None)
    parser.add_argument("--wavepacket-late-v", type=float, default=None)
    parser.add_argument("--wavepacket-mover", choices=("right", "left"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manuscript:
        overrides = [key for key, value in vars(args).items()
                     if key not in {"out_dir", "manuscript"} and value not in (None, False)]
        if overrides:
            raise ValueError("--manuscript uses fixed presets; remove overrides: " + ", ".join(overrides))
        generate_manuscript_figures(args.out_dir.resolve())
        return
    params = FigureParams(
    a=1.0,
    tev=6.0,
    mu=0.5,
    nu=0.0,
    p=16,
    dt=0.5,
    tmin=0.0,
    tmax=25.0,
    size_cutoff=100.0,
    size_fixed_v_time=None,
    formation_beta=10.0,
    formation_t0=-20.0,
    chi_cal_j=0.5,
    mi_tf=25.0,
    mi_kernel_norm_cutoff=50.0,
    mi_direction="R",
    mi_component=0,
    mi_fixed_v_times=(6.5, 12.0, 17.5),
)

    if args.quick:
        params.dt = 1.0
        params.tmax = 12.0
        params.formation_t0 = -8.0

    if args.dt is not None:
        params.dt = args.dt
    if args.tmax is not None:
        params.tmax = args.tmax
        params.mi_tf = args.tmax
    if args.size_fixed_v_time is not None:
        params.size_fixed_v_time = args.size_fixed_v_time
    if args.formation_t0 is not None:
        params.formation_t0 = args.formation_t0
    if args.formation_beta is not None:
        params.formation_beta = args.formation_beta
    if args.mi_fixed_v_times is not None:
        params.mi_fixed_v_times = tuple(args.mi_fixed_v_times)
    if args.mi_num_fixed_v_cuts is not None:
        params.mi_num_fixed_v_cuts = args.mi_num_fixed_v_cuts
        if args.mi_fixed_v_times is None:
            params.mi_fixed_v_times = ()
    if args.wavepacket_early_time is not None:
        params.wavepacket_early_time = args.wavepacket_early_time
    if args.wavepacket_late_time is not None:
        params.wavepacket_late_time = args.wavepacket_late_time
    if args.wavepacket_depth is not None:
        params.wavepacket_depth = args.wavepacket_depth
    if args.wavepacket_early_u is not None:
        params.wavepacket_early_u = args.wavepacket_early_u
    if args.wavepacket_early_v is not None:
        params.wavepacket_early_v = args.wavepacket_early_v
    if args.wavepacket_late_u is not None:
        params.wavepacket_late_u = args.wavepacket_late_u
    if args.wavepacket_late_v is not None:
        params.wavepacket_late_v = args.wavepacket_late_v
    if args.wavepacket_mover is not None:
        params.wavepacket_mover = args.wavepacket_mover

    generate_all(params, args.out_dir.resolve(), skip_mi=args.skip_mi)


if __name__ == "__main__":
    main()
