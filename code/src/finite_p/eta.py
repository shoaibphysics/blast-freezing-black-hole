"""Single-replica eta-sector Schwinger-Dyson solver.

This module solves the ordinary one-replica eta probe problem on one folded
Schwinger-Keldysh contour, using a fixed chi bath solution.

The eta self-energy has three regions:

    region I:   both contour times are before evaporation;
                eta has its own nonlinear SYK self-energy.

    region II:  one time is before evaporation and the other is after;
                there is no interaction self-energy.

    region III: both contour times are after evaporation;
                eta is driven by the fixed chi bath self-energy.

There is no replica signed swap in this module.  The signed swap belongs to the
two-replica eta problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chi import syk_block_sign
from .components import rl_block_slice
from .contour import (
    ContourMeta,
    antisymmetrize,
    block_weight_vector,
    make_contour_weights,
    make_free_majorana_g0,
    physical_time_on_contour,
    to_numpy,
)
from .newton import NewtonInfo, newton_krylov


@dataclass(frozen=True)
class EtaSolveInfo:
    """Metadata and convergence diagnostics for one eta solve."""

    cal_J: float
    mu: float
    p: int
    t_ev: float
    a_param: float
    eta_bilinear_nu: float
    contour: ContourMeta
    newton: NewtonInfo
    method: str = "single_replica_eta_newton_krylov"

    @property
    def status(self) -> str:
        return self.newton.status

    @property
    def converged(self) -> bool:
        return self.newton.converged

    @property
    def residual_norm(self) -> float | None:
        return self.newton.residual_norm

    def as_dict(self) -> dict:
        """Return a flat dictionary useful for pandas rows."""

        out = {
            "method": self.method,
            "cal_J": self.cal_J,
            "mu": self.mu,
            "p": self.p,
            "t_ev": self.t_ev,
            "a_param": self.a_param,
            "eta_bilinear_nu": self.eta_bilinear_nu,
            "status": self.status,
            "converged": self.converged,
            "residual_norm": self.residual_norm,
            "newton_iterations": self.newton.iterations,
            "gmres_info": self.newton.gmres_info,
            "gmres_matvecs": self.newton.gmres_matvecs,
            "step_history": self.newton.step_history,
        }
        out.update(self.contour.as_dict())
        return out


def eta_region_masks(
    meta: ContourMeta,
    t_ev: float,
    *,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return region-I, region-II, and region-III masks.

    Euclidean preparation points are assigned physical time zero by
    physical_time_on_contour, so they belong to the pre-evaporation region.
    The evaporation surface itself is included in region I, while region III
    starts strictly after t_ev.
    """

    t_real = to_numpy(physical_time_on_contour(meta)).astype(np.float64, copy=False)

    mask_I = (t_real[:, None] <= t_ev + tol) & (t_real[None, :] <= t_ev + tol)
    mask_III = (t_real[:, None] > t_ev + tol) & (t_real[None, :] > t_ev + tol)
    mask_II = ~(mask_I | mask_III)

    return mask_I, mask_II, mask_III, t_real


def make_eta_bilinear_vertex(
    meta: ContourMeta,
    *,
    mu: float,
    p: int,
    t_ev: float,
    nu: float = 0.0,
    tol: float = 1e-12,
) -> np.ndarray:
    """Build the optional post-evaporation eta bilinear vertex.

    The numerical calculations used in the current draft set this term to zero,
    which corresponds to nu = 0.  If nu is nonzero, this implements

        h_eta(t) = nu theta(t - t_ev)

    on the real-time contour segments.  The thermal preparation segments have
    h_eta = 0.
    """

    C = meta.N_contour
    V = np.zeros((2 * C, 2 * C), dtype=np.complex128)

    if mu == 0 or nu == 0:
        return V

    t_real = to_numpy(physical_time_on_contour(meta)).astype(np.float64, copy=False)
    active = t_real > t_ev + tol
    local = np.diag(active.astype(np.complex128))

    coeff = mu * nu / p
    V[:C, C:] = -1j * coeff * local
    V[C:, :C] = 1j * coeff * local

    return V


def _check_single_replica_shape(name: str, matrix, meta: ContourMeta) -> np.ndarray:
    """Convert a matrix to NumPy and check its one-replica R/L shape."""

    arr = to_numpy(matrix).astype(np.complex128, copy=False)
    expected = (2 * meta.N_contour, 2 * meta.N_contour)

    if arr.shape != expected:
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected}")

    return arr


def eta_self_energy_pre(
    G_eta,
    meta: ContourMeta,
    *,
    cal_J: float,
    p: int,
    a_param: float,
    mask_I: np.ndarray,
) -> np.ndarray:
    """Return the nonlinear pre-evaporation eta self-energy.

    In components,

        Sigma_eta,I =
            M_I (-1)^(1 + p delta_ab / 2)
            (a^2 J^2 / p) [2 G_eta,ab]^(p-1).
    """

    G_eta = _check_single_replica_shape("G_eta", G_eta, meta)

    if mask_I.shape != (meta.N_contour, meta.N_contour):
        raise ValueError("mask_I has the wrong shape")

    Sigma = np.zeros_like(G_eta, dtype=np.complex128)
    prefactor = (a_param**2) * (cal_J**2) / p

    for a, row_species in enumerate(("R", "L")):
        row = rl_block_slice(meta, row_species)

        for b, col_species in enumerate(("R", "L")):
            col = rl_block_slice(meta, col_species)

            delta_ab = 1 if a == b else 0
            sign = syk_block_sign(p, delta_ab)

            G_block = G_eta[row, col]
            Sigma_block = Sigma[row, col]

            Sigma_block[mask_I] = (
                sign
                * prefactor
                * (2.0 * G_block[mask_I]) ** (p - 1)
            )

    return Sigma


def eta_delta_self_energy_pre(
    G_eta,
    dG_eta,
    meta: ContourMeta,
    *,
    cal_J: float,
    p: int,
    a_param: float,
    mask_I: np.ndarray,
) -> np.ndarray:
    """Return the variation of the pre-evaporation eta self-energy."""

    G_eta = _check_single_replica_shape("G_eta", G_eta, meta)
    dG_eta = _check_single_replica_shape("dG_eta", dG_eta, meta)

    if mask_I.shape != (meta.N_contour, meta.N_contour):
        raise ValueError("mask_I has the wrong shape")

    dSigma = np.zeros_like(G_eta, dtype=np.complex128)
    prefactor = (a_param**2) * (cal_J**2) / p

    for a, row_species in enumerate(("R", "L")):
        row = rl_block_slice(meta, row_species)

        for b, col_species in enumerate(("R", "L")):
            col = rl_block_slice(meta, col_species)

            delta_ab = 1 if a == b else 0
            sign = syk_block_sign(p, delta_ab)

            G_block = G_eta[row, col]
            dG_block = dG_eta[row, col]
            dSigma_block = dSigma[row, col]

            dSigma_block[mask_I] = (
                sign
                * prefactor
                * (p - 1)
                * 2.0
                * (2.0 * G_block[mask_I]) ** (p - 2)
                * dG_block[mask_I]
            )

    return dSigma


def eta_fixed_chi_self_energy_post(
    G_chi,
    meta: ContourMeta,
    *,
    cal_J: float,
    p: int,
    mask_III: np.ndarray,
) -> np.ndarray:
    """Return the fixed post-evaporation eta source from the chi bath.

    This is the single-replica version of the post-evaporation source:

        Sigma_eta,III =
            M_III (-1)^(1 + p delta_ab / 2)
            (J^2 / p) [2 G_chi,ab]^(p-1).

    There is no signed replica swap here.
    """

    G_chi = _check_single_replica_shape("G_chi", G_chi, meta)

    if mask_III.shape != (meta.N_contour, meta.N_contour):
        raise ValueError("mask_III has the wrong shape")

    Sigma = np.zeros_like(G_chi, dtype=np.complex128)
    prefactor = (cal_J**2) / p

    for a, row_species in enumerate(("R", "L")):
        row = rl_block_slice(meta, row_species)

        for b, col_species in enumerate(("R", "L")):
            col = rl_block_slice(meta, col_species)

            delta_ab = 1 if a == b else 0
            sign = syk_block_sign(p, delta_ab)

            G_block = G_chi[row, col]
            Sigma_block = Sigma[row, col]

            Sigma_block[mask_III] = (
                sign
                * prefactor
                * (2.0 * G_block[mask_III]) ** (p - 1)
            )

    return Sigma


def make_eta_problem(
    *,
    G_chi,
    beta: float,
    t_f: float,
    dt: float,
    cal_J: float,
    mu: float,
    p: int,
    t_ev: float,
    a_param: float,
    eta_bilinear_nu: float = 0.0,
    device=None,
):
    """Construct the eta residual, JVP, and preconditioner for one contour."""

    w_torch, meta = make_contour_weights(beta=beta, t_f=t_f, dt=dt, device=device)

    G0 = to_numpy(make_free_majorana_g0(meta, device=device)).astype(
        np.complex128,
        copy=False,
    )
    w = to_numpy(w_torch).astype(np.complex128, copy=False)

    W_vec = to_numpy(block_weight_vector(w_torch, num_blocks=2)).astype(
        np.complex128,
        copy=False,
    )

    G_chi = _check_single_replica_shape("G_chi", G_chi, meta)

    mask_I, mask_II, mask_III, t_real = eta_region_masks(meta, t_ev=t_ev)

    Sigma_III = eta_fixed_chi_self_energy_post(
        G_chi,
        meta,
        cal_J=cal_J,
        p=p,
        mask_III=mask_III,
    )

    V_eta = make_eta_bilinear_vertex(
        meta,
        mu=mu,
        p=p,
        t_ev=t_ev,
        nu=eta_bilinear_nu,
    )

    # A local contour delta function carries only one contour weight.
    WV_eta = W_vec[:, None] * V_eta

    def weighted_nonlocal_kernel(Sigma: np.ndarray) -> np.ndarray:
        """Attach quadrature weights to a two-time nonlocal self-energy."""

        return W_vec[:, None] * Sigma * W_vec[None, :]

    fixed_kernel = weighted_nonlocal_kernel(Sigma_III)

    A_eta = np.eye(2 * meta.N_contour, dtype=np.complex128) - G0 @ (
        WV_eta + fixed_kernel
    )

    def residual(G_eta: np.ndarray) -> np.ndarray:
        Sigma_I = eta_self_energy_pre(
            G_eta,
            meta,
            cal_J=cal_J,
            p=p,
            a_param=a_param,
            mask_I=mask_I,
        )

        return A_eta @ G_eta - G0 - G0 @ weighted_nonlocal_kernel(Sigma_I) @ G_eta

    def jvp(G_eta: np.ndarray, dG_eta: np.ndarray) -> np.ndarray:
        Sigma_I = eta_self_energy_pre(
            G_eta,
            meta,
            cal_J=cal_J,
            p=p,
            a_param=a_param,
            mask_I=mask_I,
        )
        dSigma_I = eta_delta_self_energy_pre(
            G_eta,
            dG_eta,
            meta,
            cal_J=cal_J,
            p=p,
            a_param=a_param,
            mask_I=mask_I,
        )

        return (
            A_eta @ dG_eta
            - G0 @ (
                weighted_nonlocal_kernel(dSigma_I) @ G_eta
                + weighted_nonlocal_kernel(Sigma_I) @ dG_eta
            )
        )

    def preconditioner(B: np.ndarray) -> np.ndarray:
        """Apply the inverse of the fixed post-evaporation linear operator."""

        return np.linalg.solve(A_eta, B)

    return {
        "w": w,
        "meta": meta,
        "G0": G0,
        "G_chi": G_chi,
        "W_vec": W_vec,
        "V_eta": V_eta,
        "WV_eta": WV_eta,
        "mask_I": mask_I,
        "mask_II": mask_II,
        "mask_III": mask_III,
        "t_real": t_real,
        "Sigma_III": Sigma_III,
        "A_eta": A_eta,
        "residual": residual,
        "jvp": jvp,
        "preconditioner": preconditioner,
    }


def solve_eta(
    *,
    G_chi,
    beta: float,
    t_f: float,
    dt: float,
    cal_J: float,
    mu: float,
    p: int,
    t_ev: float,
    a_param: float,
    eta_bilinear_nu: float = 0.0,
    initial_G=None,
    initial_from_chi: bool = True,
    tol: float = 1e-8,
    max_newton: int = 12,
    gmres_rtol: float = 1e-4,
    gmres_atol: float = 0.0,
    gmres_maxiter: int = 70,
    max_abs_guard: float | None = 2.0,
    enforce_antisymmetry: bool = True,
    use_preconditioner: bool = True,
    line_search_steps: int = 14,
    verbose: bool = False,
    device=None,
) -> tuple[np.ndarray, np.ndarray, EtaSolveInfo]:
    """Solve the single-replica eta probe equation.

    Parameters
    ----------
    G_chi:
        Fixed one-replica chi bath Green's function on the same contour.

    initial_G:
        Optional eta initial guess.  This is where continuation in t_f enters:
        embed a previously converged eta solution onto the new contour and pass
        it here.

    initial_from_chi:
        If True and initial_G is not supplied, use G_chi as the eta seed.
        Otherwise use the free propagator G0.

    eta_bilinear_nu:
        Optional post-evaporation eta bilinear fraction.  The current draft
        numerics use eta_bilinear_nu = 0, so V_eta = 0.

    Returns
    -------
    G_eta:
        The converged one-replica eta Green's function as a NumPy array with
        R,L block ordering.

    w:
        The one-species contour weight vector as a NumPy array.

    info:
        Metadata and Newton convergence diagnostics.
    """

    problem = make_eta_problem(
        G_chi=G_chi,
        beta=beta,
        t_f=t_f,
        dt=dt,
        cal_J=cal_J,
        mu=mu,
        p=p,
        t_ev=t_ev,
        a_param=a_param,
        eta_bilinear_nu=eta_bilinear_nu,
        device=device,
    )

    meta = problem["meta"]
    G0 = problem["G0"]

    if initial_G is None:
        if initial_from_chi:
            x0 = problem["G_chi"]
        else:
            x0 = G0
    else:
        x0 = np.asarray(initial_G, dtype=np.complex128)
        expected = G0.shape
        if x0.shape != expected:
            raise ValueError(f"initial_G has shape {x0.shape}, expected {expected}")

    projector = antisymmetrize if enforce_antisymmetry else None
    preconditioner = problem["preconditioner"] if use_preconditioner else None

    G_eta, newton_info = newton_krylov(
        x0,
        residual=problem["residual"],
        jvp=problem["jvp"],
        preconditioner=preconditioner,
        projector=projector,
        tol=tol,
        max_newton=max_newton,
        gmres_rtol=gmres_rtol,
        gmres_atol=gmres_atol,
        gmres_maxiter=gmres_maxiter,
        max_abs_guard=max_abs_guard,
        line_search_steps=line_search_steps,
        verbose=verbose,
    )

    info = EtaSolveInfo(
        cal_J=float(cal_J),
        mu=float(mu),
        p=int(p),
        t_ev=float(t_ev),
        a_param=float(a_param),
        eta_bilinear_nu=float(eta_bilinear_nu),
        contour=meta,
        newton=newton_info,
    )

    return G_eta, problem["w"], info