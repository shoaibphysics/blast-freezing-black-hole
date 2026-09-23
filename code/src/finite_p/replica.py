"""Two-replica eta-sector solver.

This module is the two-replica analogue of eta.py.  It solves the eta probe
saddle in a fixed chi bath, with either

    disconnected: no replica twist in the post-evaporation chi source,

or

    twisted: the post-evaporation chi source is dressed by the signed replica
    swap P^\dagger(z_1) P(z_2).

Block convention
----------------
The full two-replica eta matrix has shape

    (4 * N_contour, 4 * N_contour),

with block order

    replica 1, R
    replica 1, L
    replica 2, R
    replica 2, L

and each block has size N_contour x N_contour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .chi import syk_block_sign
from .contour import (
    ContourMeta,
    antisymmetrize,
    block_weight_vector,
    branch_swap_matrices,
    contour_index_for_time,
    make_contour_weights,
    make_free_majorana_g0,
    to_numpy,
)
from .eta import eta_region_masks
from .newton import NewtonInfo, newton_krylov


ReplicaMode = Literal["disconnected", "twisted"]


@dataclass(frozen=True)
class TwoReplicaEtaSolveInfo:
    """Metadata and convergence diagnostics for one two-replica eta solve."""

    cal_J: float
    mu: float
    p: int
    t_ev: float
    a_param: float
    mode: ReplicaMode
    contour: ContourMeta
    newton: NewtonInfo
    method: str = "two_replica_eta_newton_krylov"

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
            "mode": self.mode,
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


def replica_species_slice(
    meta: ContourMeta,
    replica: int,
    species: str,
) -> slice:
    """Return the block slice for a 1-based replica and R/L species."""

    if replica not in {1, 2}:
        raise ValueError("replica must be 1 or 2")

    species_index = {"R": 0, "L": 1, "r": 0, "l": 1}[species]
    block = 2 * (replica - 1) + species_index
    C = meta.N_contour

    return slice(block * C, (block + 1) * C)


def _check_chi_shape(name: str, matrix, meta: ContourMeta) -> np.ndarray:
    """Convert a chi matrix to NumPy and check its one-replica R/L shape."""

    arr = to_numpy(matrix).astype(np.complex128, copy=False)
    expected = (2 * meta.N_contour, 2 * meta.N_contour)

    if arr.shape != expected:
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected}")

    return arr


def _check_eta_shape(name: str, matrix, meta: ContourMeta) -> np.ndarray:
    """Convert an eta matrix to NumPy and check its two-replica shape."""

    arr = to_numpy(matrix).astype(np.complex128, copy=False)
    expected = (4 * meta.N_contour, 4 * meta.N_contour)

    if arr.shape != expected:
        raise ValueError(f"{name} has shape {arr.shape}, expected {expected}")

    return arr


def make_two_replica_eta_g0(meta: ContourMeta, device=None) -> np.ndarray:
    """Free eta propagator for two uncoupled replicas."""

    one_replica = to_numpy(make_free_majorana_g0(meta, device=device)).astype(
        np.complex128,
        copy=False,
    )

    C = meta.N_contour
    G0 = np.zeros((4 * C, 4 * C), dtype=np.complex128)

    G0[: 2 * C, : 2 * C] = one_replica
    G0[2 * C :, 2 * C :] = one_replica

    return G0


def two_replica_eta_self_energy_pre(
    G_eta,
    meta: ContourMeta,
    *,
    cal_J: float,
    p: int,
    a_param: float,
    mask_I: np.ndarray,
) -> np.ndarray:
    """Nonlinear region-I two-replica eta self-energy.

    Componentwise,

        Sigma^I_{alpha a i, beta b j}
        =
        M_I(i,j) (-1)^(1 + p delta_ab / 2)
        (a^2 J^2 / p)
        [2 G_{alpha a i, beta b j}]^(p-1).

    The replica indices are not summed here; this is an elementwise bilocal
    self-energy in the full two-replica eta matrix.
    """

    G_eta = _check_eta_shape("G_eta", G_eta, meta)

    if mask_I.shape != (meta.N_contour, meta.N_contour):
        raise ValueError("mask_I has the wrong shape")

    Sigma = np.zeros_like(G_eta, dtype=np.complex128)
    prefactor = (a_param**2) * (cal_J**2) / p

    for alpha in (1, 2):
        for beta in (1, 2):
            for a, row_species in enumerate(("R", "L")):
                row = replica_species_slice(meta, alpha, row_species)

                for b, col_species in enumerate(("R", "L")):
                    col = replica_species_slice(meta, beta, col_species)

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


def two_replica_eta_delta_self_energy_pre(
    G_eta,
    dG_eta,
    meta: ContourMeta,
    *,
    cal_J: float,
    p: int,
    a_param: float,
    mask_I: np.ndarray,
) -> np.ndarray:
    """Jacobian action of the nonlinear region-I eta self-energy."""

    G_eta = _check_eta_shape("G_eta", G_eta, meta)
    dG_eta = _check_eta_shape("dG_eta", dG_eta, meta)

    if mask_I.shape != (meta.N_contour, meta.N_contour):
        raise ValueError("mask_I has the wrong shape")

    dSigma = np.zeros_like(G_eta, dtype=np.complex128)
    prefactor = (a_param**2) * (cal_J**2) / p

    for alpha in (1, 2):
        for beta in (1, 2):
            for a, row_species in enumerate(("R", "L")):
                row = replica_species_slice(meta, alpha, row_species)

                for b, col_species in enumerate(("R", "L")):
                    col = replica_species_slice(meta, beta, col_species)

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


def two_replica_sigma_iii_disconnected(
    G_chi,
    meta: ContourMeta,
    *,
    cal_J: float,
    p: int,
    mask_III: np.ndarray,
) -> np.ndarray:
    """Fixed post-evaporation chi source without replica twist."""

    G_chi = _check_chi_shape("G_chi", G_chi, meta)

    if mask_III.shape != (meta.N_contour, meta.N_contour):
        raise ValueError("mask_III has the wrong shape")

    Sigma = np.zeros((4 * meta.N_contour, 4 * meta.N_contour), dtype=np.complex128)
    prefactor = (cal_J**2) / p

    C = meta.N_contour

    for a, row_species in enumerate(("R", "L")):
        chi_row = slice(a * C, (a + 1) * C)

        for b, col_species in enumerate(("R", "L")):
            chi_col = slice(b * C, (b + 1) * C)

            delta_ab = 1 if a == b else 0
            sign = syk_block_sign(p, delta_ab)

            source = (
                sign
                * prefactor
                * (2.0 * G_chi[chi_row, chi_col]) ** (p - 1)
            )

            for alpha in (1, 2):
                row = replica_species_slice(meta, alpha, row_species)
                col = replica_species_slice(meta, alpha, col_species)
                Sigma_block = Sigma[row, col]
                Sigma_block[mask_III] = source[mask_III]

    return Sigma


def two_replica_sigma_iii_twisted(
    G_chi,
    meta: ContourMeta,
    *,
    cal_J: float,
    p: int,
    mask_III: np.ndarray,
    device=None,
) -> np.ndarray:
    """Fixed post-evaporation chi source with the signed replica twist.

    The twist is

        Sigma_III -> P^\dagger(z_1) Sigma_III P(z_2),

    where P is identity on the upper branch and the signed swap

        S = [[0, -1], [1, 0]]

    on the lower branch.
    """

    G_chi = _check_chi_shape("G_chi", G_chi, meta)

    if mask_III.shape != (meta.N_contour, meta.N_contour):
        raise ValueError("mask_III has the wrong shape")

    C = meta.N_contour
    P = to_numpy(branch_swap_matrices(meta, device=device)).astype(
        np.complex128,
        copy=False,
    )
    I_rep = np.eye(2, dtype=np.complex128)

    Sigma = np.zeros((4 * C, 4 * C), dtype=np.complex128)
    prefactor = (cal_J**2) / p

    for a, row_species in enumerate(("R", "L")):
        chi_row = slice(a * C, (a + 1) * C)

        for b, col_species in enumerate(("R", "L")):
            chi_col = slice(b * C, (b + 1) * C)

            delta_ab = 1 if a == b else 0
            sign = syk_block_sign(p, delta_ab)

            source = (
                sign
                * prefactor
                * (2.0 * G_chi[chi_row, chi_col]) ** (p - 1)
            )

            # X_{alpha beta i j} is diagonal in replica before the twist.
            X_rep = I_rep[:, :, None, None] * source[None, None, :, :]

            # Y_{alpha beta i j}
            # = P^\dagger(i)_{alpha alpha'} X_{alpha' beta' i j}
            #   P(j)_{beta' beta}.
            Y = np.einsum("ica,cdij,jdb->abij", np.conj(P), X_rep, P)

            for alpha in (1, 2):
                for beta in (1, 2):
                    row = replica_species_slice(meta, alpha, row_species)
                    col = replica_species_slice(meta, beta, col_species)

                    Sigma_block = Sigma[row, col]
                    Sigma_block[mask_III] = Y[alpha - 1, beta - 1][mask_III]

    return Sigma


def two_replica_sigma_iii(
    G_chi,
    meta: ContourMeta,
    *,
    cal_J: float,
    p: int,
    mask_III: np.ndarray,
    mode: ReplicaMode,
    device=None,
) -> np.ndarray:
    """Build the fixed region-III source for the requested replica saddle."""

    if mode == "disconnected":
        return two_replica_sigma_iii_disconnected(
            G_chi,
            meta,
            cal_J=cal_J,
            p=p,
            mask_III=mask_III,
        )

    if mode == "twisted":
        return two_replica_sigma_iii_twisted(
            G_chi,
            meta,
            cal_J=cal_J,
            p=p,
            mask_III=mask_III,
            device=device,
        )

    raise ValueError("mode must be 'disconnected' or 'twisted'")


def make_two_replica_eta_problem(
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
    mode: ReplicaMode = "twisted",
    device=None,
) -> dict:
    """Construct residual, JVP, and preconditioner for two-replica eta."""

    w_torch, meta = make_contour_weights(beta=beta, t_f=t_f, dt=dt, device=device)

    G0 = make_two_replica_eta_g0(meta, device=device)
    w = to_numpy(w_torch).astype(np.complex128, copy=False)

    W_vec = to_numpy(block_weight_vector(w_torch, num_blocks=4)).astype(
        np.complex128,
        copy=False,
    )

    G_chi = _check_chi_shape("G_chi", G_chi, meta)

    mask_I, mask_II, mask_III, t_real = eta_region_masks(meta, t_ev=t_ev)

    Sigma_III = two_replica_sigma_iii(
        G_chi,
        meta,
        cal_J=cal_J,
        p=p,
        mask_III=mask_III,
        mode=mode,
        device=device,
    )

    def weighted_nonlocal_kernel(Sigma: np.ndarray) -> np.ndarray:
        """Attach contour quadrature weights to a bilocal self-energy."""

        return W_vec[:, None] * Sigma * W_vec[None, :]

    fixed_kernel = weighted_nonlocal_kernel(Sigma_III)

    # This is the full-contour fixed linear operator.  Its inverse is used as
    # the GMRES preconditioner.
    A_eta = np.eye(4 * meta.N_contour, dtype=np.complex128) - G0 @ fixed_kernel

    def residual(G_eta: np.ndarray) -> np.ndarray:
        Sigma_I = two_replica_eta_self_energy_pre(
            G_eta,
            meta,
            cal_J=cal_J,
            p=p,
            a_param=a_param,
            mask_I=mask_I,
        )

        return A_eta @ G_eta - G0 - G0 @ weighted_nonlocal_kernel(Sigma_I) @ G_eta

    def jvp(G_eta: np.ndarray, dG_eta: np.ndarray) -> np.ndarray:
        Sigma_I = two_replica_eta_self_energy_pre(
            G_eta,
            meta,
            cal_J=cal_J,
            p=p,
            a_param=a_param,
            mask_I=mask_I,
        )
        dSigma_I = two_replica_eta_delta_self_energy_pre(
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
        "mask_I": mask_I,
        "mask_II": mask_II,
        "mask_III": mask_III,
        "t_real": t_real,
        "Sigma_III": Sigma_III,
        "A_eta": A_eta,
        "residual": residual,
        "jvp": jvp,
        "preconditioner": preconditioner,
        "mode": mode,
    }


def two_replica_initial_guess(
    G_chi,
    meta: ContourMeta,
    G0: np.ndarray,
) -> np.ndarray:
    """Initial guess with each replica diagonal block seeded by chi."""

    G_chi = _check_chi_shape("G_chi", G_chi, meta)

    C = meta.N_contour
    G = np.array(G0, dtype=np.complex128, copy=True)

    G[: 2 * C, : 2 * C] = G_chi
    G[2 * C :, 2 * C :] = G_chi

    return G


def solve_two_replica_eta(
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
    mode: ReplicaMode = "twisted",
    initial_G=None,
    max_newton: int = 12,
    gmres_rtol: float = 1e-4,
    gmres_atol: float = 0.0,
    gmres_maxiter: int = 60,
    tol: float = 1e-8,
    max_abs_guard: float | None = 2.0,
    line_search_steps: int = 14,
    enforce_antisymmetry: bool = True,
    use_preconditioner: bool = True,
    verbose: bool = False,
    return_problem: bool = False,
    device=None,
):
    """Solve one two-replica eta saddle.

    By default this returns

        G_eta, w, info

    matching the style of solve_chi and solve_eta.  If return_problem=True, it
    returns

        G_eta, w, info, problem

    where problem contains the masks, fixed Sigma_III, and linear operator.
    """

    problem = make_two_replica_eta_problem(
        G_chi=G_chi,
        beta=beta,
        t_f=t_f,
        dt=dt,
        cal_J=cal_J,
        mu=mu,
        p=p,
        t_ev=t_ev,
        a_param=a_param,
        mode=mode,
        device=device,
    )

    meta = problem["meta"]

    if initial_G is None:
        G0 = two_replica_initial_guess(
            problem["G_chi"],
            meta,
            problem["G0"],
        )
    else:
        G0 = _check_eta_shape("initial_G", initial_G, meta).copy()

    projector = antisymmetrize if enforce_antisymmetry else None
    preconditioner = problem["preconditioner"] if use_preconditioner else None

    G_eta, newton_info = newton_krylov(
        G0,
        problem["residual"],
        problem["jvp"],
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

    info = TwoReplicaEtaSolveInfo(
        cal_J=cal_J,
        mu=mu,
        p=p,
        t_ev=t_ev,
        a_param=a_param,
        mode=mode,
        contour=meta,
        newton=newton_info,
    )

    if return_problem:
        return G_eta, problem["w"], info, problem

    return G_eta, problem["w"], info


def solve_two_replica_eta_pair(
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
    initial_G_disc=None,
    initial_G_twisted=None,
    **solve_kwargs,
) -> dict:
    """Solve disconnected and chi-twisted two-replica eta saddles."""

    G_disc, w_disc, info_disc, problem_disc = solve_two_replica_eta(
        G_chi=G_chi,
        beta=beta,
        t_f=t_f,
        dt=dt,
        cal_J=cal_J,
        mu=mu,
        p=p,
        t_ev=t_ev,
        a_param=a_param,
        mode="disconnected",
        initial_G=initial_G_disc,
        return_problem=True,
        **solve_kwargs,
    )

    G_twisted, w_twisted, info_twisted, problem_twisted = solve_two_replica_eta(
        G_chi=G_chi,
        beta=beta,
        t_f=t_f,
        dt=dt,
        cal_J=cal_J,
        mu=mu,
        p=p,
        t_ev=t_ev,
        a_param=a_param,
        mode="twisted",
        initial_G=initial_G_twisted,
        return_problem=True,
        **solve_kwargs,
    )

    return {
        "G_chi": problem_disc["G_chi"],
        "G_disc": G_disc,
        "w_disc": w_disc,
        "info_disc": info_disc,
        "problem_disc": problem_disc,
        "Sigma_III_disc": problem_disc["Sigma_III"],
        "G_twisted": G_twisted,
        "w_twisted": w_twisted,
        "info_twisted": info_twisted,
        "problem_twisted": problem_twisted,
        "Sigma_III_twisted": problem_twisted["Sigma_III"],
    }


def two_replica_block_offset(
    meta: ContourMeta,
    replica: int,
    species: str,
) -> int:
    """Return the matrix offset for a 1-based replica and R/L species."""

    return replica_species_slice(meta, replica, species).start


def get_two_replica_eta_block(
    G_eta,
    meta: ContourMeta,
    replica_1: int,
    replica_2: int,
    species_1: str,
    species_2: str,
) -> np.ndarray:
    """Extract one C x C block from the full two-replica eta matrix."""

    G_eta = _check_eta_shape("G_eta", G_eta, meta)

    row = replica_species_slice(meta, replica_1, species_1)
    col = replica_species_slice(meta, replica_2, species_2)

    return G_eta[row, col]


def get_two_replica_eta_value(
    G_eta,
    meta: ContourMeta,
    replica_1: int,
    replica_2: int,
    species_1: str,
    species_2: str,
    idx_1: int,
    idx_2: int,
) -> complex:
    """Extract one matrix element from the full two-replica eta matrix."""

    G_eta = _check_eta_shape("G_eta", G_eta, meta)

    row = two_replica_block_offset(meta, replica_1, species_1) + idx_1
    col = two_replica_block_offset(meta, replica_2, species_2) + idx_2

    return complex(G_eta[row, col])


def boundary_overlap_from_two_replica_eta(
    G_eta,
    meta: ContourMeta,
    t_b: float,
) -> complex:
    """Return the replica overlap from the chi-twisted eta Green's function.

    The draft convention is

        O(t_b; t_f)
        =
        2 i G^{(2), eta; chi-tw, >}_{RR;12}(t_b, t_b; t_f).

    The greater component is represented as lower-upper on the folded contour,
    so the first insertion is replica 1 on the lower branch and the second is
    replica 2 on the upper branch.
    """

    idx_lower = contour_index_for_time(meta, t_b, branch="lower")
    idx_upper = contour_index_for_time(meta, t_b, branch="upper")

    value = get_two_replica_eta_value(
        G_eta,
        meta,
        replica_1=1,
        replica_2=2,
        species_1="R",
        species_2="R",
        idx_1=idx_lower,
        idx_2=idx_upper,
    )

    return 2j * value


def offdiag_replica_norm(
    G_eta,
    meta: ContourMeta,
) -> float:
    """Maximum absolute value in the off-diagonal replica blocks."""

    G_eta = _check_eta_shape("G_eta", G_eta, meta)

    C = meta.N_contour
    block_12 = G_eta[: 2 * C, 2 * C : 4 * C]
    block_21 = G_eta[2 * C : 4 * C, : 2 * C]

    return max(
        float(np.max(np.abs(block_12))),
        float(np.max(np.abs(block_21))),
    )


def pair_summary_row(
    pair: dict,
    *,
    t_b: float | None = None,
) -> dict:
    """Build a compact diagnostics row for a solved two-replica pair."""

    info_disc = pair["info_disc"]
    info_twisted = pair["info_twisted"]

    row = {
        "t_f": info_twisted.contour.t_f,
        "disc_status": info_disc.status,
        "disc_converged": info_disc.converged,
        "disc_residual_norm": info_disc.residual_norm,
        "twisted_status": info_twisted.status,
        "twisted_converged": info_twisted.converged,
        "twisted_residual_norm": info_twisted.residual_norm,
        "disc_offdiag_norm": offdiag_replica_norm(
            pair["G_disc"],
            info_disc.contour,
        ),
        "twisted_offdiag_norm": offdiag_replica_norm(
            pair["G_twisted"],
            info_twisted.contour,
        ),
        "disc_max_sigma_III": float(np.max(np.abs(pair["Sigma_III_disc"]))),
        "twisted_max_sigma_III": float(np.max(np.abs(pair["Sigma_III_twisted"]))),
    }
    row.update(info_twisted.contour.as_dict())

    if t_b is not None:
        overlap = boundary_overlap_from_two_replica_eta(
            pair["G_twisted"],
            info_twisted.contour,
            t_b,
        )
        row["O"] = overlap
        row["O_real"] = float(np.real(overlap))
        row["O_imag"] = float(np.imag(overlap))
        row["O_abs"] = float(abs(overlap))

    return row




###### Mutual information (boundary particle)

def two_replica_eta_contraction(
    G_eta,
    meta,
    *,
    replica_1,
    replica_2,
    branch_1,
    branch_2,
    t1,
    t2,
    species_1="R",
    species_2="R",
):
    """
    Return the normalized contraction

        2 i G^{(2),eta;chi-tw}_{species_1 species_2; replica_1 replica_2}
            (t1_branch_1, t2_branch_2; t_f).

    This is the object entering F2, F4, and K2.
    """

    idx_1 = contour_index_for_time(meta, t1, branch=branch_1)
    idx_2 = contour_index_for_time(meta, t2, branch=branch_2)

    value = get_two_replica_eta_value(
        G_eta,
        meta,
        replica_1=replica_1,
        replica_2=replica_2,
        species_1=species_1,
        species_2=species_2,
        idx_1=idx_1,
        idx_2=idx_2,
    )

    return 2j * value


def _real_with_diagnostic(z, name, imag_tol=1e-7):
    """
    Return the real part of a nearly-real complex number.

    The imaginary part is kept as a diagnostic in the output table. If it is not
    small, the function prints a warning.
    """

    z = complex(z)

    if abs(z.imag) > imag_tol:
        print(f"warning: {name} has non-small imaginary part: {z.imag}")

    return float(z.real)


def eta_mutual_information_at_tb(G_twisted, meta, t_b, *, imag_tol=1e-7):
    """
    Compute F2, F4, K2 and the classical / quantum second Renyi mutual
    information from one chi-twisted two-replica eta solution.
    """

    if t_b > meta.t_f + 1e-12:
        raise ValueError(f"t_b={t_b} is outside the contour with t_f={meta.t_f}")

    G11_mp = two_replica_eta_contraction(
        G_twisted, meta,
        replica_1=1, replica_2=1,
        branch_1="lower", branch_2="upper",
        t1=t_b, t2=t_b,
    )

    G22_mp = two_replica_eta_contraction(
        G_twisted, meta,
        replica_1=2, replica_2=2,
        branch_1="lower", branch_2="upper",
        t1=t_b, t2=t_b,
    )

    G12_mp = two_replica_eta_contraction(
        G_twisted, meta,
        replica_1=1, replica_2=2,
        branch_1="lower", branch_2="upper",
        t1=t_b, t2=t_b,
    )

    G21_mp = two_replica_eta_contraction(
        G_twisted, meta,
        replica_1=2, replica_2=1,
        branch_1="lower", branch_2="upper",
        t1=t_b, t2=t_b,
    )

    G12_mm = two_replica_eta_contraction(
        G_twisted, meta,
        replica_1=1, replica_2=2,
        branch_1="lower", branch_2="lower",
        t1=t_b, t2=t_b,
    )

    G12_pp = two_replica_eta_contraction(
        G_twisted, meta,
        replica_1=1, replica_2=2,
        branch_1="upper", branch_2="upper",
        t1=t_b, t2=t_b,
    )

    F2 = G12_mp
    K2 = G11_mp
    F4 = G11_mp * G22_mp - G12_mp * G21_mp - G12_mm * G12_pp # made changes here

    F2_real = _real_with_diagnostic(F2, "F2", imag_tol=imag_tol)
    F4_real = _real_with_diagnostic(F4, "F4", imag_tol=imag_tol)
    K2_real = _real_with_diagnostic(K2, "K2", imag_tol=imag_tol)

    if F4_real <= 0:
        exp_minus_I2_classical = np.nan
        I2_classical = np.nan
        print(f"warning: F4_real={F4_real} is not positive")
    else:
        exp_minus_I2_classical = (1.0 + 2.0 * F2_real + F4_real) / (
            4.0 * np.sqrt(F4_real)
        )
        I2_classical = -np.log(exp_minus_I2_classical)

    quantum_denominator = 1.0 + 2.0 * K2_real + F4_real

    if quantum_denominator <= 0:
        exp_minus_I2_quantum = np.nan
        I2_quantum = np.nan
        print(f"warning: quantum denominator={quantum_denominator} is not positive")
    else:
        exp_minus_I2_quantum = 0.5 * (1.0 + 2.0 * F2_real + F4_real) / (
            quantum_denominator
        )
        I2_quantum = -np.log(exp_minus_I2_quantum)

    return {
        "t_f": float(meta.t_f),
        "t_b": float(t_b),

        "F2": F2,
        "F2_real": F2_real,
        "F2_imag": float(np.imag(F2)),

        "F4": F4,
        "F4_real": F4_real,
        "F4_imag": float(np.imag(F4)),

        "K2": K2,
        "K2_real": K2_real,
        "K2_imag": float(np.imag(K2)),

        "G11_minus_plus": G11_mp,
        "G22_minus_plus": G22_mp,
        "G12_minus_plus": G12_mp,
        "G21_minus_plus": G21_mp,
        "G12_minus_minus": G12_mm,
        "G12_plus_plus": G12_pp,

        "exp_minus_I2_classical": exp_minus_I2_classical,
        "I2_classical": I2_classical,

        "exp_minus_I2_quantum": exp_minus_I2_quantum,
        "I2_quantum": I2_quantum,
    }