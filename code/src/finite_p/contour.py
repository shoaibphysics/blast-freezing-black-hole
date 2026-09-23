"""Shared contour utilities for the evaporation numerics package.

This module contains only low-level contour bookkeeping: folded-contour grid
sizes, quadrature weights, free Majorana propagators, local bilinear vertices,
and branch swap matrices.  It should not contain any
chi, eta, replica, or action-specific nonlinear physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ContourMeta:
    """Grid-size bookkeeping for the full folded Schwinger-Keldysh contour."""

    beta: float
    t_f: float
    dt: float
    N_beta: int
    N_t: int
    N_fold: int
    N_contour: int
    dt_euclidean: float | None = None

    @property
    def euclidean_step(self) -> float:
        """Use the recorded Euclidean step, or legacy dt for old snapshots."""
        return self.dt if self.dt_euclidean is None else self.dt_euclidean

    @property
    def beta_grid(self) -> float:
        return 4 * self.N_beta * self.euclidean_step

    def as_dict(self) -> dict[str, float | int | None]:
        """Return metadata in a dictionary form convenient for dataframes."""

        return {
            "beta": self.beta,
            "t_f": self.t_f,
            "dt": self.dt,
            "N_beta": self.N_beta,
            "N_t": self.N_t,
            "N_fold": self.N_fold,
            "N_contour": self.N_contour,
            "dt_euclidean": self.dt_euclidean,
        }


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Return the requested torch device, or choose CUDA if it is available."""

    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_contour_weights(
    beta: float,
    t_f: float,
    dt: float,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, ContourMeta]:
    """Build the folded-contour quadrature weights.

    The contour ordering is

        upper Euclidean, upper real, lower real, lower Euclidean.

    The returned vector has length N_contour = 2 * (N_beta + N_t).  These
    weights already include the orientation of each contour segment. The
    real-time step remains dt; each Euclidean segment has length beta / 4.
    """

    device = get_device(device)

    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("beta and dt must be positive and finite")
    N_beta = int(round(beta / (4 * dt)))
    if N_beta < 1:
        raise ValueError("dt is too coarse: each Euclidean segment needs at least one point")
    N_t = int(round(t_f / dt))
    N_fold = N_beta + N_t
    N_contour = 2 * N_fold

    meta = ContourMeta(
        beta=float(beta),
        t_f=float(t_f),
        dt=float(dt),
        N_beta=N_beta,
        N_t=N_t,
        N_fold=N_fold,
        N_contour=N_contour,
        dt_euclidean=float(beta / (4 * N_beta)),
    )

    return contour_weights_from_meta(meta, device=device), meta


def contour_weights_from_meta(
    meta: ContourMeta,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Reconstruct recorded quadrature without reinterpreting legacy beta."""
    w = torch.zeros(meta.N_contour, dtype=torch.complex128, device=get_device(device))
    w[:meta.N_beta] = -1j * meta.euclidean_step
    w[meta.N_beta:meta.N_fold] = meta.dt
    w[meta.N_fold:meta.N_fold + meta.N_t] = -meta.dt
    w[meta.N_fold + meta.N_t:] = -1j * meta.euclidean_step
    return w


def real_time_mask(
    meta: ContourMeta,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return a boolean mask selecting the two real-time contour segments."""

    device = get_device(device)
    mask = torch.zeros(meta.N_contour, dtype=torch.bool, device=device)

    mask[meta.N_beta : meta.N_fold + meta.N_t] = True

    return mask


def physical_time_on_contour(
    meta: ContourMeta,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return the physical real time associated with each contour point.

    Euclidean preparation points are assigned physical time zero.  On the upper
    real branch the grid is dt, 2dt, ..., t_f.  On the lower real branch the
    grid is t_f, t_f-dt, ..., dt.
    """

    device = get_device(device)
    t_real = torch.zeros(meta.N_contour, dtype=torch.float64, device=device)

    if meta.N_t > 0:
        upper_times = torch.arange(
            1,
            meta.N_t + 1,
            dtype=torch.float64,
            device=device,
        ) * meta.dt

        lower_times = meta.t_f - torch.arange(
            0,
            meta.N_t,
            dtype=torch.float64,
            device=device,
        ) * meta.dt

        t_real[meta.N_beta : meta.N_fold] = upper_times
        t_real[meta.N_fold : meta.N_fold + meta.N_t] = lower_times

    return t_real


def make_free_majorana_g0(
    meta: ContourMeta,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Free two-sided Majorana propagator in the R/L block convention.

    The output has shape (2 N_contour, 2 N_contour), with block order R, L.
    The diagonal R/R and L/L blocks are contour sign functions, while the
    off-diagonal R/L and L/R blocks encode the thermofield left-right
    contraction convention.
    """

    device = get_device(device)
    C = meta.N_contour

    idx = torch.arange(C, device=device)
    diff = (idx[:, None] - idx[None, :]).to(torch.float64)
    Gdiag = (-0.5j * torch.sign(diff)).to(torch.complex128)

    G0 = torch.zeros((2 * C, 2 * C), dtype=torch.complex128, device=device)
    G0[:C, :C] = Gdiag
    G0[C:, C:] = Gdiag
    G0[:C, C:] = -0.5
    G0[C:, :C] = 0.5

    return G0


def make_bilinear_vertex(
    meta: ContourMeta,
    mu: float,
    p: int,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Build the local R/L bilinear vertex.

    The bilinear is active only on the real-time branches.  This implements the
    convention that the thermal preparation segments have h_mu = 0.

    C = meta.N_contour

    V_RR = V[:C, :C]
    V_RL = V[:C, C:]
    V_LR = V[C:, :C]
    V_LL = V[C:, C:]
    """


    device = get_device(device)
    C = meta.N_contour

    V = torch.zeros((2 * C, 2 * C), dtype=torch.complex128, device=device)

    mask = real_time_mask(meta, device=device).to(torch.complex128)
    local = torch.diag(mask)

    V[:C, C:] = -1j * (mu / p) * local
    V[C:, :C] = 1j * (mu / p) * local

    return V


def signed_swap_matrix(
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return the signed replica swap S = [[0, -1], [1, 0]]."""

    device = get_device(device)

    S = torch.zeros((2, 2), dtype=torch.complex128, device=device)
    S[0, 1] = -1.0
    S[1, 0] = 1.0

    return S


def branch_swap_matrices(
    meta: ContourMeta,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return P(z): identity on the upper branch and S on the lower branch.

    The output has shape (N_contour, 2, 2).  For each contour point z_i, P[i]
    is the two-replica frame rotation used in the chi-twisted convention.
    """

    device = get_device(device)
    C = meta.N_contour

    P = torch.zeros((C, 2, 2), dtype=torch.complex128, device=device)

    is_upper = torch.arange(C, device=device) < meta.N_fold

    P[is_upper, 0, 0] = 1.0
    P[is_upper, 1, 1] = 1.0

    P[~is_upper, 0, 1] = -1.0
    P[~is_upper, 1, 0] = 1.0

    return P


def contour_index_for_time(
    meta: ContourMeta,
    t: float,
    branch: str = "upper",
) -> int:
    """Map a real physical time to an upper- or lower-branch contour index."""

    n_t = int(round(t / meta.dt))

    if n_t < 1 or n_t > meta.N_t:
        raise ValueError(f"t={t} is outside the real-time contour for t_f={meta.t_f}")

    if branch in {"upper", "forward", "u", "+"}:
        return meta.N_beta + n_t - 1

    if branch in {"lower", "backward", "l", "-"}:
        return meta.N_fold + (meta.N_t - n_t)

    raise ValueError("branch must be upper/forward/+ or lower/backward/-")


def contour_index_map(
    old_meta: ContourMeta,
    new_meta: ContourMeta,
) -> np.ndarray:
    """Map each new one-species contour index to a nearest old contour index.

    This is used for continuation in t_f: a previously converged solution on
    one contour is embedded as an initial guess on a nearby contour.
    """

    old_N_fold = old_meta.N_fold
    old_N_contour = old_meta.N_contour
    new_N_fold = new_meta.N_fold
    new_N_contour = new_meta.N_contour

    out = np.zeros(new_N_contour, dtype=int)

    def clip_int(x: float, lo: int, hi: int) -> int:
        return int(min(max(round(x), lo), hi))

    for k in range(new_N_contour):
        if k < new_meta.N_beta:
            frac = k / max(new_meta.N_beta - 1, 1)
            out[k] = clip_int(
                frac * max(old_meta.N_beta - 1, 0),
                0,
                max(old_meta.N_beta - 1, 0),
            )

        elif k < new_N_fold:
            t = (k - new_meta.N_beta + 1) * new_meta.dt
            n_old = clip_int(t / old_meta.dt - 1, 0, old_meta.N_t - 1)
            out[k] = old_meta.N_beta + n_old

        elif k < new_N_fold + new_meta.N_t:
            m = k - new_N_fold
            t = new_meta.t_f - m * new_meta.dt
            t = min(max(t, 0.0), old_meta.t_f)
            m_old = clip_int((old_meta.t_f - t) / old_meta.dt, 0, old_meta.N_t - 1)
            out[k] = old_N_fold + m_old

        else:
            q = k - (new_N_fold + new_meta.N_t)
            frac = q / max(new_meta.N_beta - 1, 1)
            q_old = clip_int(
                frac * max(old_meta.N_beta - 1, 0),
                0,
                max(old_meta.N_beta - 1, 0),
            )
            out[k] = old_N_fold + old_meta.N_t + q_old

    if np.any(out < 0) or np.any(out >= old_N_contour):
        raise ValueError("constructed invalid contour index map")

    return out


def embed_block_matrix(
    matrix_old,
    old_meta: ContourMeta,
    new_meta: ContourMeta,
    num_blocks: int,
) -> torch.Tensor:
    """Embed a block contour matrix onto a new contour.

    The matrix is assumed to have shape

        (num_blocks * old_N_contour, num_blocks * old_N_contour),

    where each block is an old_N_contour by old_N_contour contour matrix.  The
    same contour index map is applied to every block.
    """

    matrix_np = to_numpy(matrix_old).astype(np.complex128, copy=False)

    expected = (
        num_blocks * old_meta.N_contour,
        num_blocks * old_meta.N_contour,
    )
    if matrix_np.shape != expected:
        raise ValueError(f"matrix_old has shape {matrix_np.shape}, expected {expected}")

    idx = contour_index_map(old_meta, new_meta)

    out = np.zeros(
        (
            num_blocks * new_meta.N_contour,
            num_blocks * new_meta.N_contour,
        ),
        dtype=np.complex128,
    )

    for row_block in range(num_blocks):
        for col_block in range(num_blocks):
            old_block = matrix_np[
                row_block * old_meta.N_contour : (row_block + 1) * old_meta.N_contour,
                col_block * old_meta.N_contour : (col_block + 1) * old_meta.N_contour,
            ]

            new_block = old_block[np.ix_(idx, idx)]

            out[
                row_block * new_meta.N_contour : (row_block + 1) * new_meta.N_contour,
                col_block * new_meta.N_contour : (col_block + 1) * new_meta.N_contour,
            ] = new_block

    out_tensor = torch.as_tensor(out, dtype=torch.complex128)

    if torch.is_tensor(matrix_old):
        out_tensor = out_tensor.to(matrix_old.device)

    return out_tensor


def block_weight_vector(w: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Repeat one contour weight vector for a block matrix."""

    return torch.cat([w] * num_blocks)


def to_numpy(x) -> np.ndarray:
    """Convert a torch tensor or array-like object to a NumPy array."""

    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def antisymmetrize(matrix: np.ndarray) -> np.ndarray:
    """Project a matrix onto its antisymmetric part."""

    return 0.5 * (matrix - matrix.T)
