"""Single-replica chi-sector Schwinger-Dyson solver.

This module solves the ordinary one-replica chi bath problem on one fixed
folded Schwinger-Keldysh contour.  It contains only chi-sector physics:
self-energies, Jacobian-vector products, and the Newton-Krylov solve.

Component extraction and plotting are intentionally kept outside this module.
Use components.py for extracting RR, RL, LR, LL blocks or G_ab(t, 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .components import rl_block_slice
from .contour import (
    ContourMeta,
    antisymmetrize,
    block_weight_vector,
    make_bilinear_vertex,
    make_contour_weights,
    make_free_majorana_g0,
    to_numpy,
)
from .newton import NewtonInfo, newton_krylov


@dataclass(frozen=True)
class ChiSolveInfo:
    """Metadata and convergence diagnostics for one chi solve."""

    cal_J: float
    mu: float
    p: int
    contour: ContourMeta
    newton: NewtonInfo
    method: str = "single_replica_chi_newton_krylov"

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


def _check_even_p(p: int) -> None:
    """Require even p so that (-1)^(1 + p delta_ab / 2) is unambiguous."""

    if p % 2 != 0:
        raise ValueError("This SYK sign convention requires even p.")


def syk_block_sign(p: int, delta_ab: int) -> int:
    """Return (-1)^(1 + p delta_ab / 2) for delta_ab = 0 or 1."""

    _check_even_p(p)

    exponent = 1 + (p // 2) * delta_ab
    return -1 if exponent % 2 else 1


def chi_self_energy(
    G: np.ndarray,
    meta: ContourMeta,
    cal_J: float,
    p: int,
) -> np.ndarray:
    """Return the chi SYK self-energy Sigma[G].

    The block convention is R,L.  In components,

        Sigma_ab,ij =
            (-1)^(1 + p delta_ab / 2)
            (cal_J^2 / p)
            [2 G_ab,ij]^(p-1).

    For p = 4, 16, 24, 32, ... this sign is -1 in all R/L blocks, matching
    the old working notebooks.  The function is nevertheless written with the
    general block sign kept explicit.
    """

    G = np.asarray(G, dtype=np.complex128)
    expected = (2 * meta.N_contour, 2 * meta.N_contour)

    if G.shape != expected:
        raise ValueError(f"G has shape {G.shape}, expected {expected}")

    Sigma = np.zeros_like(G, dtype=np.complex128)

    for a, row_species in enumerate(("R", "L")):
        row = rl_block_slice(meta, row_species)

        for b, col_species in enumerate(("R", "L")):
            col = rl_block_slice(meta, col_species)

            delta_ab = 1 if a == b else 0
            sign = syk_block_sign(p, delta_ab)

            G_block = G[row, col]
            Sigma[row, col] = sign * (cal_J**2 / p) * (2.0 * G_block) ** (p - 1)

    return Sigma


def chi_delta_self_energy(
    G: np.ndarray,
    dG: np.ndarray,
    meta: ContourMeta,
    cal_J: float,
    p: int,
) -> np.ndarray:
    """Return the variation delta Sigma[G; dG].

    This is the Jacobian action of the self-energy functional:

        delta Sigma_ab,ij =
            (-1)^(1 + p delta_ab / 2)
            (cal_J^2 / p)
            (p - 1) 2 [2 G_ab,ij]^(p-2) dG_ab,ij.
    """

    G = np.asarray(G, dtype=np.complex128)
    dG = np.asarray(dG, dtype=np.complex128)

    expected = (2 * meta.N_contour, 2 * meta.N_contour)
    if G.shape != expected:
        raise ValueError(f"G has shape {G.shape}, expected {expected}")
    if dG.shape != expected:
        raise ValueError(f"dG has shape {dG.shape}, expected {expected}")

    dSigma = np.zeros_like(G, dtype=np.complex128)

    for a, row_species in enumerate(("R", "L")):
        row = rl_block_slice(meta, row_species)

        for b, col_species in enumerate(("R", "L")):
            col = rl_block_slice(meta, col_species)

            delta_ab = 1 if a == b else 0
            sign = syk_block_sign(p, delta_ab)

            G_block = G[row, col]
            dG_block = dG[row, col]

            dSigma[row, col] = (
                sign
                * (cal_J**2 / p)
                * (p - 1)
                * 2.0
                * (2.0 * G_block) ** (p - 2)
                * dG_block
            )

    return dSigma


def make_chi_problem(
    *,
    beta: float,
    t_f: float,
    dt: float,
    cal_J: float,
    mu: float,
    p: int,
    device=None,
):
    """Construct the chi residual, JVP, and preconditioner for one contour."""

    w_torch, meta = make_contour_weights(beta=beta, t_f=t_f, dt=dt, device=device)

    G0 = to_numpy(make_free_majorana_g0(meta, device=device)).astype(
        np.complex128,
        copy=False,
    )
    V_mu = to_numpy(make_bilinear_vertex(meta, mu=mu, p=p, device=device)).astype(
        np.complex128,
        copy=False,
    )
    w = to_numpy(w_torch).astype(np.complex128, copy=False)

    W_vec = to_numpy(block_weight_vector(w_torch, num_blocks=2)).astype(
        np.complex128,
        copy=False,
    )

    # The local bilinear vertex contains a contour delta function, so it
    # carries only one contour weight.
    WV_mu = W_vec[:, None] * V_mu

    # This is the simple linear part used as a left preconditioner for GMRES.
    linear_mu = np.eye(2 * meta.N_contour, dtype=np.complex128) - G0 @ WV_mu

    def weighted_nonlocal_kernel(Sigma: np.ndarray) -> np.ndarray:
        """Attach quadrature weights to the two-time nonlocal self-energy."""

        return W_vec[:, None] * Sigma * W_vec[None, :]

    def residual(G: np.ndarray) -> np.ndarray:
        Sigma = chi_self_energy(G, meta, cal_J=cal_J, p=p)
        K = WV_mu + weighted_nonlocal_kernel(Sigma)

        return G - G0 - G0 @ K @ G

    def jvp(G: np.ndarray, dG: np.ndarray) -> np.ndarray:
        Sigma = chi_self_energy(G, meta, cal_J=cal_J, p=p)
        dSigma = chi_delta_self_energy(G, dG, meta, cal_J=cal_J, p=p)

        K = WV_mu + weighted_nonlocal_kernel(Sigma)
        dK = weighted_nonlocal_kernel(dSigma)

        return dG - G0 @ (dK @ G + K @ dG)

    def preconditioner(B: np.ndarray) -> np.ndarray:
        """Apply the inverse of the bilinear-only linear operator."""

        return np.linalg.solve(linear_mu, B)

    return {
        "w": w,
        "meta": meta,
        "G0": G0,
        "V_mu": V_mu,
        "W_vec": W_vec,
        "WV_mu": WV_mu,
        "linear_mu": linear_mu,
        "residual": residual,
        "jvp": jvp,
        "preconditioner": preconditioner,
    }


def solve_chi(
    *,
    beta: float,
    t_f: float,
    dt: float,
    cal_J: float,
    mu: float,
    p: int,
    initial_G=None,
    tol: float = 1e-9,
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
) -> tuple[np.ndarray, np.ndarray, ChiSolveInfo]:
    """Solve the single-replica chi Schwinger-Dyson equation.

    Parameters
    ----------
    initial_G:
        Optional initial guess.  This is where continuation in t_f enters:
        embed a previously converged solution onto the new contour and pass it
        here.

    Returns
    -------
    G:
        The converged one-replica chi Green's function as a NumPy array with
        R,L block ordering.

    w:
        The one-species contour weight vector as a NumPy array.

    info:
        Metadata and Newton convergence diagnostics.
    """

    problem = make_chi_problem(
        beta=beta,
        t_f=t_f,
        dt=dt,
        cal_J=cal_J,
        mu=mu,
        p=p,
        device=device,
    )

    meta = problem["meta"]
    G0 = problem["G0"]

    if initial_G is None:
        x0 = G0
    else:
        x0 = np.asarray(initial_G, dtype=np.complex128)
        expected = G0.shape
        if x0.shape != expected:
            raise ValueError(f"initial_G has shape {x0.shape}, expected {expected}")

    projector = antisymmetrize if enforce_antisymmetry else None
    preconditioner = problem["preconditioner"] if use_preconditioner else None

    G, newton_info = newton_krylov(
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

    info = ChiSolveInfo(
        cal_J=float(cal_J),
        mu=float(mu),
        p=int(p),
        contour=meta,
        newton=newton_info,
    )

    return G, problem["w"], info