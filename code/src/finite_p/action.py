"""Two-replica on-shell action helpers.

This module evaluates the eta-normalized probe action described in the
numerical action derivation in ``paper/supplementary.tex``, Sec. S5.

The main public function is

    eta_action_difference(...)

which takes the saved two-replica Green's functions

    G_chi, G_disc, G_twisted

and returns the action difference

    Delta i_eta = i_eta[twisted] - i_eta[disconnected].

The second Renyi entropy density is

    S_2^eta / N_eta = Re Delta i_eta.
"""

from __future__ import annotations

import numpy as np

from .contour import (
    block_weight_vector,
    contour_weights_from_meta,
    to_numpy,
)
from .eta import eta_region_masks
from .replica import (
    make_two_replica_eta_g0,
    two_replica_eta_self_energy_pre,
    two_replica_sigma_iii,
)


def complex_logdet(matrix) -> complex:
    """Return the principal complex log determinant of a square matrix.

    This uses ``np.linalg.slogdet`` instead of ``np.log(np.linalg.det(...))``
    for better numerical stability.
    """

    matrix = np.asarray(matrix, dtype=np.complex128)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("complex_logdet expects a square matrix")

    sign, logabsdet = np.linalg.slogdet(matrix)

    if sign == 0:
        raise np.linalg.LinAlgError("matrix is singular in complex_logdet")

    return complex(np.log(sign) + logabsdet)


def two_replica_weight_vector(meta, w=None) -> np.ndarray:
    """Return the four-block contour weight vector for a two-replica eta matrix.

    If ``w`` is supplied, it is interpreted as the one-contour weight vector
    saved with the solution.  Otherwise the weights are reconstructed from
    ``meta`` using its recorded Euclidean spacing (legacy dt if absent).
    """

    if w is None:
        w_torch = contour_weights_from_meta(meta)
        W_vec = to_numpy(block_weight_vector(w_torch, num_blocks=4))
    else:
        w = to_numpy(w).astype(np.complex128, copy=False)

        if w.shape != (meta.N_contour,):
            raise ValueError(
                f"w has shape {w.shape}, expected {(meta.N_contour,)}"
            )

        W_vec = np.tile(w, 4)

    return np.asarray(W_vec, dtype=np.complex128)


def eta_region_I_interaction_from_sigmaG(sigmaI_G: complex, p: int) -> complex:
    """Return the region-I interaction term from the Sigma_I G contraction.

    Since

    \[
    \Sigma_I
    =
    (-1)^{1+p\delta_{ab}/2}
    \frac{a^2 J^2}{p}
    (2G)^{p-1},
    \]

    the explicit region-I interaction satisfies

    \[
    i_{\eta,I}
    =
    \frac{1}{2p}
    \sum w_i w_j \Sigma_I G.
    \]
    """

    return sigmaI_G / (2.0 * p)


def eta_probe_action(
    G_eta,
    G_chi,
    meta,
    *,
    cal_J: float,
    p: int,
    t_ev: float,
    a_param: float,
    mode: str,
    w=None,
) -> dict:
    """Evaluate the eta-normalized probe action on one two-replica saddle.

    Parameters
    ----------
    G_eta:
        Two-replica eta Green's function.  Shape is
        ``(4 * N_contour, 4 * N_contour)``.

    G_chi:
        One-replica chi bath Green's function.  Shape is
        ``(2 * N_contour, 2 * N_contour)``.

    meta:
        ``ContourMeta`` for the saved solution.

    mode:
        Either ``"disconnected"`` or ``"twisted"``.  This determines the
        post-evaporation fixed chi source ``Sigma_III``.

    w:
        Optional one-contour weight vector.  If omitted, it is reconstructed
        from ``meta``.

    Returns
    -------
    A dictionary containing ``i_eta`` and its separate pieces.
    """

    if mode not in {"disconnected", "twisted"}:
        raise ValueError("mode must be 'disconnected' or 'twisted'")

    G_eta = to_numpy(G_eta).astype(np.complex128, copy=False)
    G_chi = to_numpy(G_chi).astype(np.complex128, copy=False)

    expected_eta_shape = (4 * meta.N_contour, 4 * meta.N_contour)
    expected_chi_shape = (2 * meta.N_contour, 2 * meta.N_contour)

    if G_eta.shape != expected_eta_shape:
        raise ValueError(f"G_eta has shape {G_eta.shape}, expected {expected_eta_shape}")

    if G_chi.shape != expected_chi_shape:
        raise ValueError(f"G_chi has shape {G_chi.shape}, expected {expected_chi_shape}")

    W_vec = two_replica_weight_vector(meta, w=w)
    WW = W_vec[:, None] * W_vec[None, :]

    mask_I, _, mask_III, _ = eta_region_masks(meta, t_ev=t_ev)

    G0 = make_two_replica_eta_g0(meta)

    Sigma_I = two_replica_eta_self_energy_pre(
        G_eta,
        meta,
        cal_J=cal_J,
        p=p,
        a_param=a_param,
        mask_I=mask_I,
    )

    Sigma_III = two_replica_sigma_iii(
        G_chi,
        meta,
        cal_J=cal_J,
        p=p,
        mask_III=mask_III,
        mode=mode,
    )

    Sigma_total = Sigma_I + Sigma_III

    # Relative determinant:
    #   -1/2 log det[1 - G0 W Sigma W]
    weighted_sigma = WW * Sigma_total
    log_matrix = np.eye(G0.shape[0], dtype=np.complex128) - G0 @ weighted_sigma
    i_log = -0.5 * complex_logdet(log_matrix)

    # Elementwise bilocal contraction:
    #   sum_{indices} w_i w_j Sigma_I(i,j) G_eta(i,j)
    #
    # This is intentionally NOT a matrix trace.
    sigmaI_G = np.sum(WW * Sigma_I * G_eta)

    i_sigmaI_G = 0.5 * sigmaI_G
    i_eta_I = eta_region_I_interaction_from_sigmaG(sigmaI_G, p=p)

    i_eta = i_log + i_sigmaI_G - i_eta_I

    return {
        "mode": mode,
        "t_f": float(meta.t_f),
        "i_eta": i_eta,
        "i_eta_real": float(np.real(i_eta)),
        "i_eta_imag": float(np.imag(i_eta)),
        "i_log": i_log,
        "i_log_real": float(np.real(i_log)),
        "i_log_imag": float(np.imag(i_log)),
        "i_sigmaI_G": i_sigmaI_G,
        "i_sigmaI_G_real": float(np.real(i_sigmaI_G)),
        "i_sigmaI_G_imag": float(np.imag(i_sigmaI_G)),
        "i_eta_I": i_eta_I,
        "i_eta_I_real": float(np.real(i_eta_I)),
        "i_eta_I_imag": float(np.imag(i_eta_I)),
        "sigmaI_G": sigmaI_G,
        "sigmaI_G_real": float(np.real(sigmaI_G)),
        "sigmaI_G_imag": float(np.imag(sigmaI_G)),
        "max_abs_Sigma_I": float(np.max(np.abs(Sigma_I))),
        "max_abs_Sigma_III": float(np.max(np.abs(Sigma_III))),
        "max_abs_Sigma_total": float(np.max(np.abs(Sigma_total))),
    }


def eta_action_difference(
    G_chi,
    G_disc,
    G_twisted,
    meta,
    *,
    cal_J: float,
    p: int,
    t_ev: float,
    a_param: float,
    w=None,
) -> dict:
    """Compute the two-replica eta action difference.

    The returned action difference is

    \[
    \Delta i_\eta
    =
    i_{\eta,\chi{\rm -tw}}
    -
    i_{\eta,\rm disc}.
    \]

    The second Renyi entropy density is

    \[
    S_2^\eta/N_\eta
    =
    \operatorname{Re}\Delta i_\eta.
    \]
    """

    disc = eta_probe_action(
        G_disc,
        G_chi,
        meta,
        cal_J=cal_J,
        p=p,
        t_ev=t_ev,
        a_param=a_param,
        mode="disconnected",
        w=w,
    )

    twisted = eta_probe_action(
        G_twisted,
        G_chi,
        meta,
        cal_J=cal_J,
        p=p,
        t_ev=t_ev,
        a_param=a_param,
        mode="twisted",
        w=w,
    )

    delta = twisted["i_eta"] - disc["i_eta"]

    return {
        "t_f": float(meta.t_f),
        "Delta_i_eta": delta,
        "Delta_i_eta_real": float(np.real(delta)),
        "Delta_i_eta_imag": float(np.imag(delta)),
        "S2_over_Neta": float(np.real(delta)),
        "i_disc": disc["i_eta"],
        "i_twisted": twisted["i_eta"],
        "i_disc_real": disc["i_eta_real"],
        "i_disc_imag": disc["i_eta_imag"],
        "i_twisted_real": twisted["i_eta_real"],
        "i_twisted_imag": twisted["i_eta_imag"],
        "i_log_disc": disc["i_log"],
        "i_log_twisted": twisted["i_log"],
        "i_sigmaI_G_disc": disc["i_sigmaI_G"],
        "i_sigmaI_G_twisted": twisted["i_sigmaI_G"],
        "i_eta_I_disc": disc["i_eta_I"],
        "i_eta_I_twisted": twisted["i_eta_I"],
        "disc_max_abs_Sigma_I": disc["max_abs_Sigma_I"],
        "twisted_max_abs_Sigma_I": twisted["max_abs_Sigma_I"],
        "disc_max_abs_Sigma_III": disc["max_abs_Sigma_III"],
        "twisted_max_abs_Sigma_III": twisted["max_abs_Sigma_III"],
        "disc_max_abs_Sigma_total": disc["max_abs_Sigma_total"],
        "twisted_max_abs_Sigma_total": twisted["max_abs_Sigma_total"],
    }


def eta_action_consistency_checks(action_table, *, t_ev=None, tol: float = 1e-7):
    """Return simple consistency checks for an action-difference table."""

    import pandas as pd

    df = pd.DataFrame(action_table)

    checks = {
        "min_Re_Delta_i_eta": float(df["Delta_i_eta_real"].min()),
        "max_Re_Delta_i_eta": float(df["Delta_i_eta_real"].max()),
        "max_abs_Im_Delta_i_eta": float(np.max(np.abs(df["Delta_i_eta_imag"]))),
        "log_2": float(np.log(2.0)),
        "any_negative_beyond_tol": bool((df["Delta_i_eta_real"] < -tol).any()),
        "any_above_log2_beyond_tol": bool(
            (df["Delta_i_eta_real"] > np.log(2.0) + tol).any()
        ),
    }

    if t_ev is not None:
        early = df["t_f"] <= float(t_ev) + tol
        if early.any():
            checks["max_abs_early_Delta_i_eta"] = float(
                np.max(np.abs(df.loc[early, "Delta_i_eta"]))
            )
        else:
            checks["max_abs_early_Delta_i_eta"] = np.nan

    return checks
