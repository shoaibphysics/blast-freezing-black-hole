"""Analytic reference formulas for the evaporation numerics package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LargePParams:
    """Large-p parameters for the nonzero-mu chi bath solution."""

    cal_J: float
    mu: float
    epsilon: float
    tanh_gamma: float
    sinh_gamma: float
    beta: float
    phi_G: float
    V_G: float


def large_p_params(cal_J: float, mu: float) -> LargePParams:
    """Return the large-p parameter package used by the chi bath benchmarks."""

    epsilon = mu / (2 * cal_J)
    tanh_gamma = np.sqrt((epsilon / 2) * (np.sqrt(4 + epsilon**2) - epsilon))
    sinh_gamma = tanh_gamma / np.sqrt(1 - tanh_gamma**2)
    beta = (2 / cal_J) * (1 / tanh_gamma) * np.arctan(1 / sinh_gamma)

    # Constant needed for chi solution for constant mu 
    root = np.sqrt(1 + (mu / 2) ** 2)
    phi_G = -0.5 * np.log(mu * (root - mu / 2))
    V_G = np.sqrt(mu * (mu / 2 + root))

    return LargePParams(
        cal_J=cal_J,
        mu=mu,
        epsilon=epsilon,
        tanh_gamma=tanh_gamma,
        sinh_gamma=sinh_gamma,
        beta=beta,
        phi_G=phi_G,
        V_G=V_G,
    )


def beta_from_mu(cal_J: float, mu: float) -> float:
    """Return the thermal beta associated with the large-p chi bath."""

    return large_p_params(cal_J, mu).beta


def solve_epsilon(
    beta: float,
    cal_J_eff: float,
    tol: float = 1e-14,
    max_iter: int = 200,
) -> float:
    """Solve sin(epsilon) = (pi - 2 epsilon)/(beta cal_J_eff)."""

    lo = 1e-15
    hi = np.pi / 2 - 1e-15

    def f(eps: float) -> float:
        return np.sin(eps) - (np.pi - 2 * eps) / (beta * cal_J_eff)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)

        if f(mid) < 0:
            lo = mid
        else:
            hi = mid

        if hi - lo < tol:
            break

    return 0.5 * (lo + hi)


def thermal_large_p_analytic(
    tau_vals: np.ndarray,
    beta: float,
    cal_J_eff: float,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic thermal large-p G_RR(tau,0), G_RL(tau,0).

    This is the common Euclidean preparation-segment formula written in the
    epsilon notation used in the draft.  It applies to eta and chi after using
    the correct effective coupling cal_J_eff. Even for chi, mu is taken to be zero
    on the thermal preparation branch. This is why this thermal solution is common for
    both eta and chi. 

    The RR prefactor is +i/2 for the thermal contour slice used in the
    numerical benchmark, where the extracted thermal point is earlier than the
    reference point 0 in contour ordering.
    """

    eps = solve_epsilon(beta, cal_J_eff)
    sin_eps = np.sin(eps)
    s_vals = np.abs(np.asarray(tau_vals, dtype=np.float64))

    exp_g_R = (
        sin_eps / np.sin(eps + cal_J_eff * sin_eps * s_vals)
    ) ** 2
    exp_g_L = (
        sin_eps / np.cos(cal_J_eff * sin_eps * s_vals)
    ) ** 2

    G_RR = 0.5j * np.exp(np.log(exp_g_R) / p)
    G_RL = -0.5 * np.exp(np.log(exp_g_L) / p)

    return G_RR, G_RL


def chi_real_time_analytic(
    t_vals: np.ndarray,
    p: int,
    cal_J: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic chi G_RR(t,0), G_RL(t,0) on the forward real-time branch."""

    params = large_p_params(cal_J, mu)
    t_vals = np.asarray(t_vals, dtype=np.complex128)

    theta = params.V_G * t_vals / 2 - 1j * np.arctanh(np.exp(-params.phi_G))

    arg_R = params.V_G / np.sin(theta)
    phase_R = np.unwrap(np.angle(arg_R))
    g_R = -1j * np.pi + 2 * (np.log(np.abs(arg_R)) + 1j * phase_R)

    arg_L = params.V_G / np.cos(theta)
    phase_L = np.unwrap(np.angle(arg_L))
    g_L = 2 * (np.log(np.abs(arg_L)) + 1j * phase_L)

    G_RR = -0.5j * np.exp(g_R / p)
    G_RL = -0.5 * np.exp(g_L / p)

    return G_RR, G_RL


def chi_thermal_analytic(
    tau_vals: np.ndarray,
    p: int,
    cal_J: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic chi thermal G_RR(tau,0), G_RL(tau,0)."""

    beta = beta_from_mu(cal_J, mu)
    return thermal_large_p_analytic(
        tau_vals=tau_vals,
        beta=beta,
        cal_J_eff=cal_J,
        p=p,
    )


def eta_mu0_real_time_analytic(
    t_vals: np.ndarray,
    beta: float,
    a_param: float,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic eta G_RR(t,0), G_RL(t,0) for the decoupled mu=0 eta solution."""

    cal_J_eta = a_param / 2
    eps = solve_epsilon(beta, cal_J_eta)
    sin_eps = np.sin(eps)
    t_vals = np.asarray(t_vals, dtype=np.float64)

    exp_g_R = np.exp(-1j * np.pi) * (
        sin_eps / np.sinh(0.5 * t_vals * a_param * sin_eps - 1j * eps)
    ) ** 2
    exp_g_L = (
        sin_eps / np.cosh(0.5 * t_vals * a_param * sin_eps)
    ) ** 2

    G_RR = -0.5j * np.exp(np.log(exp_g_R) / p)
    G_RL = -0.5 * np.exp(np.log(exp_g_L) / p)

    return G_RR, G_RL


def eta_mu0_thermal_analytic(
    tau_vals: np.ndarray,
    beta: float,
    a_param: float,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic eta thermal G_RR(tau,0), G_RL(tau,0).
    Here we are working in units where cal_J_eta = a/2
    """

    return thermal_large_p_analytic(
        tau_vals=tau_vals,
        beta=beta,
        cal_J_eff=a_param / 2,
        p=p,
    )


def _continuous_complex_log(z, *, unwrap_phase: bool = True) -> np.ndarray:
    """Return a branch-continuous log for analytic large-p expressions.

    The formulas in the note are naturally written for exp(g).  To use the
    cross-region and post-evaporation eta formulas, we need g itself.  This
    helper computes

        g = log|z| + i arg(z)

    and unwraps the phase along all array axes.  For one-dimensional benchmark
    slices this gives the continuous branch used in the large-p solution.
    """

    z = np.asarray(z, dtype=np.complex128)
    phase = np.angle(z)

    if unwrap_phase and phase.ndim > 0:
        for axis in range(phase.ndim):
            phase = np.unwrap(phase, axis=axis)

    return np.log(np.abs(z)) + 1j * phase


def chi_large_p_g_real_time(
    t1,
    t2,
    *,
    cal_J: float,
    mu: float,
    unwrap_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the large-p chi exponents g_R^chi(t1,t2), g_L^chi(t1,t2).

    This implements Eq. (chi bath solution) in the note:

        exp(g_R^chi) = exp(-i pi) [V_G / sin(V_G(t1-t2)/2 - i atanh(e^-phi_G))]^2,

        exp(g_L^chi) = [V_G / cos(V_G(t1-t2)/2 - i atanh(e^-phi_G))]^2.

    The parameter cal_J is kept for consistency with the existing parameter
    functions.  The closed-form V_G used here is the same convention already
    used by chi_real_time_analytic.
    """

    params = large_p_params(cal_J, mu)

    t1, t2 = np.broadcast_arrays(
        np.asarray(t1, dtype=np.float64),
        np.asarray(t2, dtype=np.float64),
    )

    theta = (
        params.V_G * (t1 - t2) / 2
        - 1j * np.arctanh(np.exp(-params.phi_G))
    )

    exp_g_R = np.exp(-1j * np.pi) * (params.V_G / np.sin(theta)) ** 2
    exp_g_L = (params.V_G / np.cos(theta)) ** 2

    g_R = _continuous_complex_log(exp_g_R, unwrap_phase=unwrap_phase)
    g_L = _continuous_complex_log(exp_g_L, unwrap_phase=unwrap_phase)

    return g_R, g_L


def eta_pre_large_p_g_real_time(
    t1,
    t2,
    *,
    beta: float,
    a_param: float,
    unwrap_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pre-evaporation eta exponents g_R^eta, g_L^eta.

    This implements the pre-evaporation eta solution in the note, with

        J_eta = a / 2,

    and epsilon fixed by

        sin(epsilon) = (pi - 2 epsilon) / (beta J_eta).
    """

    t1, t2 = np.broadcast_arrays(
        np.asarray(t1, dtype=np.float64),
        np.asarray(t2, dtype=np.float64),
    )

    cal_J_eta = a_param / 2
    eps = solve_epsilon(beta, cal_J_eta)
    sin_eps = np.sin(eps)

    exp_g_R = np.exp(-1j * np.pi) * (
        sin_eps / np.sinh(0.5 * a_param * sin_eps * (t1 - t2) - 1j * eps)
    ) ** 2

    exp_g_L = (
        sin_eps / np.cosh(0.5 * a_param * sin_eps * (t1 + t2))
    ) ** 2

    g_R = _continuous_complex_log(exp_g_R, unwrap_phase=unwrap_phase)
    g_L = _continuous_complex_log(exp_g_L, unwrap_phase=unwrap_phase)

    return g_R, g_L


def eta_large_p_g_real_time(
    t1,
    t2,
    *,
    beta: float,
    cal_J: float,
    mu: float,
    t_ev: float,
    a_param: float = 1.0,
    nu: float = 0.0,
    unwrap_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the large-p single-replica eta exponents in all real-time regions.

    This implements the eta Wightman solution in the note:

    1. before evaporation, use the decoupled eta black-hole solution;
    2. across the evaporation surface, use the endpoint chi contribution;
    3. after evaporation, use the fixed chi bath solution plus evaporation data.

    The parameter nu is the post-evaporation eta bilinear fraction appearing in
    the combination mu * (1 - nu).  The numerics used so far set V_eta = 0,
    corresponding to nu = 0.
    """

    t1, t2 = np.broadcast_arrays(
        np.asarray(t1, dtype=np.float64),
        np.asarray(t2, dtype=np.float64),
    )

    g_R = np.zeros(t1.shape, dtype=np.complex128)
    g_L = np.zeros(t1.shape, dtype=np.complex128)

    t_ev_arr = np.full(t1.shape, float(t_ev), dtype=np.float64)
    mu_eff = mu * (1.0 - nu)

    mask_pre = (t1 <= t_ev) & (t2 <= t_ev)
    mask_cross_12 = (t1 > t_ev) & (t2 <= t_ev)
    mask_cross_21 = (t1 <= t_ev) & (t2 > t_ev)
    mask_post = (t1 > t_ev) & (t2 > t_ev)

    # Region I: both points before evaporation.
    pre_R, pre_L = eta_pre_large_p_g_real_time(
        t1,
        t2,
        beta=beta,
        a_param=a_param,
        unwrap_phase=unwrap_phase,
    )
    g_R[mask_pre] = pre_R[mask_pre]
    g_L[mask_pre] = pre_L[mask_pre]

    # Region IIa: t2 < t_ev < t1.
    if np.any(mask_cross_12):
        bd_R, bd_L = eta_pre_large_p_g_real_time(
            t_ev_arr,
            t2,
            beta=beta,
            a_param=a_param,
            unwrap_phase=unwrap_phase,
        )
        chi_R_1e, chi_L_1e = chi_large_p_g_real_time(
            t1,
            t_ev_arr,
            cal_J=cal_J,
            mu=mu,
            unwrap_phase=unwrap_phase,
        )

        endpoint = (
            np.real(chi_R_1e)
            + 1j * np.imag(chi_L_1e)
            + 1j * mu_eff * (t1 - t_ev)
        )

        g_R[mask_cross_12] = (bd_R + endpoint)[mask_cross_12]
        g_L[mask_cross_12] = (bd_L + endpoint)[mask_cross_12]

    # Region IIb: t1 < t_ev < t2, obtained by conjugation and exchange.
    if np.any(mask_cross_21):
        bd_R_swapped, bd_L_swapped = eta_pre_large_p_g_real_time(
            t_ev_arr,
            t1,
            beta=beta,
            a_param=a_param,
            unwrap_phase=unwrap_phase,
        )
        chi_R_2e, chi_L_2e = chi_large_p_g_real_time(
            t2,
            t_ev_arr,
            cal_J=cal_J,
            mu=mu,
            unwrap_phase=unwrap_phase,
        )

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

    # Region III: both points after evaporation.
    if np.any(mask_post):
        chi_R_12, chi_L_12 = chi_large_p_g_real_time(
            t1,
            t2,
            cal_J=cal_J,
            mu=mu,
            unwrap_phase=unwrap_phase,
        )
        chi_R_1e, chi_L_1e = chi_large_p_g_real_time(
            t1,
            t_ev_arr,
            cal_J=cal_J,
            mu=mu,
            unwrap_phase=unwrap_phase,
        )
        chi_R_e2, chi_L_e2 = chi_large_p_g_real_time(
            t_ev_arr,
            t2,
            cal_J=cal_J,
            mu=mu,
            unwrap_phase=unwrap_phase,
        )
        _, chi_L_ee = chi_large_p_g_real_time(
            t_ev_arr,
            t_ev_arr,
            cal_J=cal_J,
            mu=mu,
            unwrap_phase=unwrap_phase,
        )
        _, eta_L_ee = eta_pre_large_p_g_real_time(
            t_ev_arr,
            t_ev_arr,
            beta=beta,
            a_param=a_param,
            unwrap_phase=unwrap_phase,
        )

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

    if g_R.shape == ():
        return complex(g_R), complex(g_L)

    return g_R, g_L


def eta_large_p_wightman_real_time(
    t1,
    t2,
    *,
    beta: float,
    cal_J: float,
    mu: float,
    p: int,
    t_ev: float,
    a_param: float = 1.0,
    nu: float = 0.0,
    unwrap_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return large-p eta Wightman components G_RR^>(t1,t2), G_RL^>(t1,t2).

    The convention is

        G_RR^> = -i/2 exp(g_R^eta / p),
        G_RL^> = -1/2 exp(g_L^eta / p).
    """

    g_R, g_L = eta_large_p_g_real_time(
        t1,
        t2,
        beta=beta,
        cal_J=cal_J,
        mu=mu,
        t_ev=t_ev,
        a_param=a_param,
        nu=nu,
        unwrap_phase=unwrap_phase,
    )

    G_RR = -0.5j * np.exp(g_R / p)
    G_RL = -0.5 * np.exp(g_L / p)

    if np.shape(G_RR) == ():
        return complex(G_RR), complex(G_RL)

    return G_RR, G_RL


def eta_large_p_wightman_components_real_time(
    t1,
    t2,
    *,
    beta: float,
    cal_J: float,
    mu: float,
    p: int,
    t_ev: float,
    a_param: float = 1.0,
    nu: float = 0.0,
    unwrap_phase: bool = True,
) -> dict[str, np.ndarray]:
    """Return RR, RL, LR, LL large-p eta Wightman components.

    With the R/L convention used in the contour code,

        G_LL^> = G_RR^>,
        G_LR^> = -G_RL^>.
    """

    G_RR, G_RL = eta_large_p_wightman_real_time(
        t1,
        t2,
        beta=beta,
        cal_J=cal_J,
        mu=mu,
        p=p,
        t_ev=t_ev,
        a_param=a_param,
        nu=nu,
        unwrap_phase=unwrap_phase,
    )

    return {
        "RR": G_RR,
        "RL": G_RL,
        "LR": -G_RL,
        "LL": G_RR,
    }


def eta_large_p_anticommutator_real_time(
    t1,
    t2,
    *,
    beta: float,
    cal_J: float,
    mu: float,
    p: int,
    t_ev: float,
    a_param: float = 1.0,
    nu: float = 0.0,
    unwrap_phase: bool = True,
) -> dict[str, np.ndarray]:
    """Return the large-p eta anticommutator matrix components.

    This implements

        A_RR = Re exp(g_R^eta / p),
        A_RL = Im exp(g_L^eta / p),

    together with A_LL = A_RR and A_LR = -A_RL.
    """

    g_R, g_L = eta_large_p_g_real_time(
        t1,
        t2,
        beta=beta,
        cal_J=cal_J,
        mu=mu,
        t_ev=t_ev,
        a_param=a_param,
        nu=nu,
        unwrap_phase=unwrap_phase,
    )

    A_RR = np.real(np.exp(g_R / p))
    A_RL = np.imag(np.exp(g_L / p))

    return {
        "RR": A_RR,
        "RL": A_RL,
        "LR": -A_RL,
        "LL": A_RR,
    }