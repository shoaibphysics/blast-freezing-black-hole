"""Black-hole formation source helpers for large-p eta stitching.

The formation model keeps the first two stages of the rescued-black-hole
profile: an initial linear segment for t0 <= t < 0 and a black-hole segment for
0 <= t <= t_ev.  The traversable-wormhole segment is omitted.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve


def rescued_bh_epsilon(beta: float) -> float:
    """Return epsilon in the rescued-BH convention for a given beta."""

    pv_func = lambda x: x - beta * np.cos(x / 2)
    pv = fsolve(pv_func, 0.1)[0]
    return float((np.pi - pv) / 2)


def uniform_time_grid(
    t_start: float,
    t_stop: float,
    dt: float,
    *,
    atol: float = 1e-10,
) -> np.ndarray:
    """Return an inclusive uniform physical-time grid."""

    if dt <= 0:
        raise ValueError("dt must be positive")

    n_steps = int(round((t_stop - t_start) / dt))

    if n_steps < 0:
        raise ValueError("t_stop must be greater than or equal to t_start")

    if abs(t_start + n_steps * dt - t_stop) > atol:
        raise ValueError("time interval must be an integer multiple of dt")

    return t_start + np.arange(n_steps + 1) * dt


def formation_time_grid(
    t0: float,
    t_ev: float,
    dt: float,
    *,
    atol: float = 1e-10,
) -> np.ndarray:
    """Return the physical-time grid for the formation source."""

    if not t0 < 0.0:
        raise ValueError("t0 must be negative")
    if not t_ev >= 0.0:
        raise ValueError("t_ev must be nonnegative")
    if not dt > 0.0:
        raise ValueError("dt must be positive")

    n_pre = int(round((0.0 - t0) / dt))
    n_bh = int(round(t_ev / dt))

    if abs(t0 + n_pre * dt) > atol:
        raise ValueError("0 - t0 must be an integer multiple of dt")
    if abs(t_ev - n_bh * dt) > atol:
        raise ValueError("t_ev must be an integer multiple of dt")

    pre_times = t0 + np.arange(n_pre) * dt
    bh_times = np.arange(n_bh + 1) * dt
    return np.concatenate([pre_times, bh_times])


def black_hole_formation_profile(
    times: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return psi and Dpsi for the formation-only rescued-BH profile."""

    times = np.asarray(times, dtype=np.float64)
    eps = rescued_bh_epsilon(beta)

    sin_eps = np.sin(eps)
    tan_eps = np.tan(eps)

    psi = np.empty(times.shape, dtype=np.complex128)
    Dpsi = np.empty(times.shape, dtype=np.complex128)

    mask_bh = times >= 0.0
    t_bh = times[mask_bh]

    ephi = np.cosh(sin_eps * t_bh) / sin_eps
    phase = np.arctan(tan_eps * np.tanh(sin_eps * t_bh))
    Dpsi[mask_bh] = np.exp(1j * phase) / np.sqrt(ephi**2 - 1.0)
    Dpsi[mask_bh & np.isclose(times, 0.0)] = tan_eps
    psi[mask_bh] = 2.0 * np.arctan(
        np.tanh((sin_eps * t_bh - 1j * eps) / 2.0)
    )

    psi0 = 2.0 * np.arctan(np.tanh((-1j * eps) / 2.0))
    Dpsi0 = tan_eps
    psi[~mask_bh] = psi0 + times[~mask_bh] * Dpsi0
    Dpsi[~mask_bh] = Dpsi0

    return psi, Dpsi


def generate_black_hole_formation_profile(
    beta: float,
    t0: float,
    t_ev: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time_grid, psi, and Dpsi for the formation source."""

    time_grid = formation_time_grid(t0=t0, t_ev=t_ev, dt=dt)
    psi, Dpsi = black_hole_formation_profile(time_grid, beta=beta)
    return time_grid, psi, Dpsi


def continuous_complex_log_2d(z: np.ndarray) -> np.ndarray:
    """Return a branch-continuous complex log by unwrapping both axes."""

    z = np.asarray(z, dtype=np.complex128)
    phase = np.angle(z)

    for axis in range(phase.ndim):
        phase = np.unwrap(phase, axis=axis)

    return np.log(np.abs(z)) + 1j * phase


def coupled_syk_g_grids(
    psi: np.ndarray,
    Dpsi: np.ndarray,
    p: int,
    *,
    coupled_syk_2pt=None,
) -> dict[str, np.ndarray]:
    """Compute large-p g grids from a coupled-SYK psi profile."""

    if coupled_syk_2pt is None:
        try:
            from .largep_syk import coupledSYK2pt as coupled_syk_2pt
        except ImportError as exc:
            raise ImportError(
                "coupled_syk_g_grids requires large_p.largep_syk"
            ) from exc

    CorM, CLL_raw, CLR_raw, SigLL_raw, SigLR_raw = coupled_syk_2pt(psi, Dpsi, p)

    # coupledSYK2pt uses meshgrid indexing opposite to the t1,t2 source
    # convention used by analytic_sources, so transpose the square grids.
    exp_g_R_grid = CLL_raw.T
    exp_g_L_grid = (CLR_raw / 1j).T

    return {
        "CorM": CorM,
        "CLL_raw": CLL_raw,
        "CLR_raw": CLR_raw,
        "SigLL_raw": SigLL_raw,
        "SigLR_raw": SigLR_raw,
        "exp_g_R_grid": exp_g_R_grid,
        "exp_g_L_grid": exp_g_L_grid,
        "g_R_grid": p * continuous_complex_log_2d(exp_g_R_grid),
        "g_L_grid": p * continuous_complex_log_2d(exp_g_L_grid),
    }


def correlation_matrix_from_exp_g(
    exp_g_R: np.ndarray,
    exp_g_L: np.ndarray,
) -> np.ndarray:
    """Build the two-flavor boundary correlation matrix used by reconstruction."""

    I2 = np.eye(2)
    J2 = np.array([[0, 1], [-1, 0]])
    CorM = np.kron(exp_g_R, I2) + np.kron(exp_g_L, -1j * J2)
    return (CorM + CorM.conj().T) / 2.0


def bulk_norm_grid(gates: dict[tuple[int, int], np.ndarray], N: int, NT: int):
    """Return the ||U11||_F grid for reconstructed bulk gates."""

    norms_mat = np.full((NT, NT), np.nan)

    for (u, v), U in gates.items():
        U11 = U[:N, :N]
        s = np.linalg.svd(U11, compute_uv=False)
        norms_mat[u, v] = np.sqrt(np.sum(s**2))

    return norms_mat


def reconstruction_coordinates(tgrid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return bulk time and depth coordinate grids from boundary times."""

    tgrid = np.asarray(tgrid, dtype=np.float64)
    Tu, Tv = np.meshgrid(tgrid, tgrid, indexing="ij")
    TT = 0.5 * (Tu + Tv)
    ZZ = 0.5 * (Tv - Tu)
    return TT, ZZ


def run_bulk_reconstruction_from_correlator(
    CorM: np.ndarray,
    tgrid: np.ndarray,
    *,
    label: str,
    bulk_reconstruction_lib=None,
) -> dict[str, np.ndarray]:
    """Build bulk gates and kernels from the supplied correlator grids."""

    if bulk_reconstruction_lib is None:
        try:
            from bulk_reconstruction import reconstruction as bulk_reconstruction_lib
        except ImportError as exc:
            raise ImportError(
                "run_bulk_reconstruction_from_correlator requires "
                "bulk_reconstruction.reconstruction"
            ) from exc

    tgrid = np.asarray(tgrid, dtype=np.float64)
    NT = len(tgrid)
    N = 2
    A = np.real(CorM)
    C_mat = np.imag(CorM)

    print(f"{label}: computing bulk gates for NT={NT}")
    gates, KL_all, KR_all, is_legit, tf_list, tini_list = (
        bulk_reconstruction_lib.compute_bulk_gates(A, N, NT)
    )
    entropy_mat = bulk_reconstruction_lib.compute_bulk_entropy(
        C_mat, KL_all, is_legit, N, NT
    )
    Ipf = np.array(
        [
            bulk_reconstruction_lib.compute_past_future_mi(
                A, tini_list, tf_list, i, N
            )
            for i in range(NT)
        ]
    )
    norms_mat = bulk_norm_grid(gates, N, NT)
    TT, ZZ = reconstruction_coordinates(tgrid)

    print(f"{label}: gates={len(gates)}, legit diamonds={int(np.sum(is_legit))}")
    return {
        "label": label,
        "tgrid": tgrid,
        "CorM": CorM,
        "A": A,
        "C": C_mat,
        "gates": gates,
        "KL_all": KL_all,
        "KR_all": KR_all,
        "is_legit": is_legit,
        "tf_list": tf_list,
        "tini_list": tini_list,
        "entropy_mat": entropy_mat,
        "Ipf": Ipf,
        "norms_mat": norms_mat,
        "TT": TT,
        "ZZ": ZZ,
        "N": N,
        "NT": NT,
    }
