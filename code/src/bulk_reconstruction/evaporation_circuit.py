from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from . import reconstruction as brl

@dataclass
class BulkCircuitRegion:
    """A named collection of two-leg Gaussian gates."""

    name: str
    gates: dict
    local_to_global: dict | None = None
    metadata: dict | None = None


@dataclass
class EvaporationBulkCircuit:
    """A five-part circuit joining past/future regions through an SVD bridge."""

    N: int
    tev_idx: int
    t_in: int
    t_f: int
    f_region_t_f: int
    rank: int
    singular_values: np.ndarray
    U_pf: np.ndarray
    V1: np.ndarray
    V2: np.ndarray
    A_p: np.ndarray
    A_f: np.ndarray
    A_pf: np.ndarray
    regions: dict
    basis_rotation_info: dict | None = None

    @property
    def all_gates(self):
        """Merged gate dictionary with keys (region_name, u, v)."""
        merged = {}
        for name, region in self.regions.items():
            for (u, v), gate in region.gates.items():
                merged[(name, u, v)] = gate
        return merged


def inv_sqrt_hermitian(M, ridge=1e-12):
    """Return M^{-1/2} for a positive Hermitian matrix, with a small floor."""
    M = (M + M.conj().T) / 2
    evals, evecs = la.eigh(M)
    evals = np.clip(evals, ridge, None)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.conj().T


def extract_past_future_blocks(A, N, tev_idx, t_in=None, t_f=None):
    """
    Extract the past/future anticommutator blocks for the SVD bridge.

    The default split is P=[0, tev_idx) and
    F=[tev_idx, NT).  Pass t_in/t_f to use the stable window found by
    bulk_reconstruction_lib.compute_bulk_gates.
    """
    NT = A.shape[0] // N
    if A.shape != (NT * N, NT * N):
        raise ValueError("A must have shape (NT*N, NT*N)")

    if t_in is None:
        t_in = 0
    if t_f is None:
        t_f = NT
    t_in = int(t_in)
    t_f = int(t_f)
    tev_idx = int(tev_idx)

    if not (0 <= t_in < tev_idx < t_f <= NT):
        raise ValueError(
            f"Need 0 <= t_in < tev_idx < t_f <= NT, got "
            f"t_in={t_in}, tev_idx={tev_idx}, t_f={t_f}, NT={NT}"
        )

    idx_p = np.arange(t_in * N, tev_idx * N)
    idx_f = np.arange(tev_idx * N, t_f * N)
    A_p = A[np.ix_(idx_p, idx_p)]
    A_f = A[np.ix_(idx_f, idx_f)]
    A_pf = A[np.ix_(idx_p, idx_f)]
    return t_in, t_f, idx_p, idx_f, A_p, A_f, A_pf


def normalized_past_future_overlap(A_p, A_pf, A_f, ridge=1e-12):
    """Compute U = A_p^{-1/2} A_pf A_f^{-1/2}."""
    return inv_sqrt_hermitian(A_p, ridge) @ A_pf @ inv_sqrt_hermitian(A_f, ridge)


def svd_isometries(U_pf, N, rank=None, tol=1e-10):
    """
    SVD of U_pf = V1 D V2^dagger.

    V1 acts on the past side and V2 acts on the future side.  For the
    evaporating SYK application rank is expected to be 2*N=4.
    """
    U_svd, singular_values, Vh = np.linalg.svd(U_pf, full_matrices=False)
    if rank is None:
        rank = 2 * N
    rank = int(rank)
    if rank % N != 0:
        raise ValueError(f"rank must be a multiple of N, got rank={rank}")
    if rank > len(singular_values):
        raise ValueError(f"rank={rank} exceeds available singular values")
    if singular_values[rank - 1] < tol:
        raise ValueError(
            f"Requested rank {rank}, but sigma[{rank - 1}]={singular_values[rank - 1]:g}"
        )

    V1 = U_svd[:, :rank]
    V2 = Vh.conj().T[:, :rank]
    return V1, singular_values[:rank], V2


def _is_degenerate_pair(singular_values, pair, tol):
    """Return True if two singular values are degenerate within relative tol."""
    a, b = pair
    scale = max(1.0, abs(singular_values[a]), abs(singular_values[b]))
    return abs(singular_values[a] - singular_values[b]) <= tol * scale


def rotate_degenerate_pairs_by_flavor_polarization(
    V1,
    V2,
    singular_values,
    N,
    degeneracy_tol=1e-8,
):
    """
    Use degenerate-subspace freedom to diagonalize V1 flavor polarization.

    For the N=2 evaporation circuit the leading singular values are typically
    pairwise degenerate.  Within each degenerate pair, rotate V1 and V2 by the
    same 2x2 unitary Q so that V1 diagonalizes I_time \\otimes diag(1,-1).
    The pair's singular values are replaced by their average, making the
    rotation exactly compatible with the truncated SVD inside the degenerate
    block.
    """
    V1_rot = V1.copy()
    V2_rot = V2.copy()
    singular_values_rot = singular_values.copy()
    rotation_info = {
        "applied": False,
        "degeneracy_tol": degeneracy_tol,
        "pairs": [],
    }

    if N != 2:
        rotation_info["reason"] = "flavor-polarization rotation is implemented for N=2"
        return V1_rot, singular_values_rot, V2_rot, rotation_info

    ell = V1.shape[0] // N
    Z_flavor = np.kron(np.eye(ell), np.diag([1.0, -1.0]))

    for a in range(0, len(singular_values_rot) - 1, 2):
        pair = (a, a + 1)
        singular_before = singular_values_rot[[a, a + 1]].copy()
        pair_info = {
            "pair": pair,
            "singular_values_before": singular_before,
            "degenerate": _is_degenerate_pair(singular_values_rot, pair, degeneracy_tol),
        }
        if not pair_info["degenerate"]:
            rotation_info["pairs"].append(pair_info)
            continue

        V_pair = V1_rot[:, [a, a + 1]]
        H = V_pair.conj().T @ Z_flavor @ V_pair
        H = (H + H.conj().T) / 2
        evals, Q = np.linalg.eigh(H)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        Q = Q[:, order]

        V1_rot[:, [a, a + 1]] = V1_rot[:, [a, a + 1]] @ Q
        V2_rot[:, [a, a + 1]] = V2_rot[:, [a, a + 1]] @ Q
        singular_values_rot[[a, a + 1]] = np.mean(singular_before)

        pair_info.update(
            {
                "applied": True,
                "flavor_polarizations": evals,
                "rotation": Q,
                "singular_values_after": singular_values_rot[[a, a + 1]].copy(),
            }
        )
        rotation_info["pairs"].append(pair_info)
        rotation_info["applied"] = True

    return V1_rot, singular_values_rot, V2_rot, rotation_info


def shift_gate_coordinates(gates, du=0, dv=0):
    """Translate a gate dictionary by integer light-cone offsets."""
    return {(u + du, v + dv): gate for (u, v), gate in gates.items()}


def place_v1_gates(local_gates, tev_idx):
    """
    Place the V1 isometry gates with an up-down coordinate reflection.

    bulk_reconstruction_lib.isometry_to_diamond_gates puts the output edge on
    local u=0 with output order v=0,1,...,ell-1.  The past horizon edge is
    ordered as u=t,t-1,...,t-ell+1, so we reflect the local output
    coordinate before embedding.  The resulting gates have z=v-u>0.

    The coordinate choice is such that the left corner gate is at (u,v)=(tev_idx-1,tev_idx)
    (The last boundary site is at (u,v)=(tev_idx-1,tev_idx-1))
    """
    placed = {}
    local_to_global = {}
    for (u_local, v_local), gate in local_gates.items():
        u_global = tev_idx - 1 - v_local
        v_global = tev_idx - u_local
        placed[(u_global, v_global)] = gate
        local_to_global[(u_local, v_local)] = (u_global, v_global)
    return placed, local_to_global


def place_v2_gates(local_gates, tev_idx):
    """
    Place the V2 isometry gates without the V1 up-down reflection.

    The future side uses the same orientation as isometry_to_diamond_gates,
    shifted one step away from the z=0 bridge so that all V2 gates sit at
    z=v-u>0.

    the coordinate is chosen such that the left corner gate is at
    (u,v)=(tev_idx-1, tev_idx)
    (The first boundary site is at (u,v)=(tev_idx, tev_idx))
    """
    placed = {}
    local_to_global = {}
    for (u_local, v_local), gate in local_gates.items():
        u_global = tev_idx - 1 + u_local
        v_global = tev_idx + v_local
        placed[(u_global, v_global)] = gate
        local_to_global[(u_local, v_local)] = (u_global, v_global)
    return placed, local_to_global


def diagonal_coupler_gate(d1, d2):
    """
    The 2N x 2N nontrivial diagonal-coupler bridge gate for N=2.

    The first two modes stay on one leg and the last two modes stay on the
    other leg.  The gate is unitary for |d_i| <= 1.
    """
    d1 = float(np.clip(d1, 0.0, 1.0))
    d2 = float(np.clip(d2, 0.0, 1.0))
    c1 = np.sqrt(max(0.0, 1.0 - d1 * d1))
    c2 = np.sqrt(max(0.0, 1.0 - d2 * d2))
    return np.array(
        [
            [d1, 0.0, c1, 0.0],
            [0.0, d2, 0.0, c2],
            [-c1, 0.0, d1, 0.0],
            [0.0, -c2, 0.0, d2],
        ],
        dtype=float,
    )


def swap_gate(N):
    """Swap the two N-flavor legs of a local two-leg gate."""
    eye = np.eye(N)
    zero = np.zeros((N, N))
    return np.block([[zero, eye], [-eye, zero]])


def build_bridge_gates(singular_values, N, tev_idx):
    """
    Build the four central bridge gates: two swaps and two D-gates.

    For the implemented two-flavor construction N=2 and rank=4.  The two extra input sites
    are u=t,t+1.  The swap gates sit at z=v-u=0, and the nontrivial D gates
    sit at z=v-u=\\pm 1.

    The four gates are at the coordinates (u,v) = (t,t), (t,t+1), (t+1,t+1), (t+1,t) 
    with t=tev_idx.
    """
    if N != 2:
        raise NotImplementedError("The Sec. 2 bridge gate formula is implemented for N=2")
    if len(singular_values) < 2 * N:
        raise ValueError(f"Need at least {2 * N} singular values")

    d = np.clip(np.asarray(singular_values[: 2 * N], dtype=float), 0.0, 1.0)
    gates = {
        (tev_idx+1, tev_idx): diagonal_coupler_gate(d[0], d[1]),
        (tev_idx, tev_idx): swap_gate(N),
        (tev_idx, tev_idx+1): diagonal_coupler_gate(d[2], d[3]),
        (tev_idx + 1, tev_idx + 1): swap_gate(N),
    }
    metadata = {
        "z_values": {key: key[1] - key[0] for key in gates},
        "extra_input_u": [tev_idx + 1, tev_idx],
        "singular_value_pairs": [(d[0], d[1]), (d[2], d[3])],
    }
    return gates, metadata


def compute_region_gates(A_region, N, offset):
    """Construct ordinary HKLL bulk gates for a standalone boundary region."""
    NT_region = A_region.shape[0] // N
    gates, KL_all, KR_all, is_legit, tf_list, tini_list = brl.compute_bulk_gates(
        A_region, N, NT_region
    )
    return (
        shift_gate_coordinates(gates, du=offset, dv=offset),
        {
            "local_gates": gates,
            "KL_all": KL_all,
            "KR_all": KR_all,
            "is_legit": is_legit,
            "tf_list": tf_list,
            "tini_list": tini_list,
            "offset": offset,
        },
    )


def build_evaporation_bulk_circuit(
    A,
    N,
    tev_idx,
    t_in=None,
    t_f=None,
    f_region_t_f=None,
    rank=None,
    ridge=1e-12,
    tol=1e-10,
    optimize_degenerate_basis=True,
    degeneracy_tol=1e-8,
):
    """
    Construct the five-part past/future bulk circuit and its SVD bridge.

    Parts:
      1. P: ordinary bulk circuit from A restricted to [t_in, tev_idx).
      2. F: ordinary bulk circuit from A restricted to
         [tev_idx, f_region_t_f).  By default this extends to the end of A.
         The separate t_f argument is the
         future endpoint used in the SVD bridge window.
      3. V1: isometry circuit for the past singular vectors, reflected
         vertically relative to bulk_reconstruction_lib.isometry_to_diamond_gates.
      4. V2: isometry circuit for the future singular vectors.
      5. bridge: two swap gates and two nontrivial diagonal-coupler D gates.
    """
    NT = A.shape[0] // N
    if f_region_t_f is None:
        f_region_t_f = NT
    f_region_t_f = int(f_region_t_f)
    if not (tev_idx < f_region_t_f <= NT):
        raise ValueError(
            f"Need tev_idx < f_region_t_f <= NT, got "
            f"tev_idx={tev_idx}, f_region_t_f={f_region_t_f}, NT={NT}"
        )

    t_in, t_f, idx_p, idx_f, A_p, A_f, A_pf = extract_past_future_blocks(
        A, N, tev_idx, t_in=t_in, t_f=t_f
    )
    U_pf = normalized_past_future_overlap(A_p, A_pf, A_f, ridge=ridge)
    V1, singular_values, V2 = svd_isometries(U_pf, N, rank=rank, tol=tol)
    basis_rotation_info = {
        "applied": False,
        "reason": "optimize_degenerate_basis is disabled",
        "pairs": [],
    }
    if optimize_degenerate_basis:
        V1, singular_values, V2, basis_rotation_info = (
            rotate_degenerate_pairs_by_flavor_polarization(
                V1,
                V2,
                singular_values,
                N=N,
                degeneracy_tol=degeneracy_tol,
            )
        )

    ell_p = A_p.shape[0] // N
    ell_f = A_f.shape[0] // N

    p_gates, p_meta = compute_region_gates(A_p, N, offset=t_in)
    A_f_region = A[tev_idx * N : f_region_t_f * N, tev_idx * N : f_region_t_f * N]
    f_gates, f_meta = compute_region_gates(A_f_region, N, offset=tev_idx+2)
    f_meta["svd_t_f"] = int(t_f)

    v1_local = brl.isometry_to_diamond_gates(V1, N=N, tol=tol)
    v1_gates, v1_map = place_v1_gates(v1_local, tev_idx=tev_idx)

    v2_local = brl.isometry_to_diamond_gates(V2, N=N, tol=tol)
    v2_gates, v2_map = place_v2_gates(v2_local, tev_idx=tev_idx+2)

    bridge_gates, bridge_meta = build_bridge_gates(singular_values, N=N, tev_idx=tev_idx)

    regions = {
        "P": BulkCircuitRegion("P", p_gates, metadata=p_meta),
        "F": BulkCircuitRegion("F", f_gates, metadata=f_meta),
        "V1": BulkCircuitRegion(
            "V1",
            v1_gates,
            local_to_global=v1_map,
            metadata={"local_gates": v1_local, "ell": ell_p},
        ),
        "V2": BulkCircuitRegion(
            "V2",
            v2_gates,
            local_to_global=v2_map,
            metadata={"local_gates": v2_local, "ell": ell_f},
        ),
        "bridge": BulkCircuitRegion("bridge", bridge_gates, metadata=bridge_meta),
    }

    return EvaporationBulkCircuit(
        N=N,
        tev_idx=int(tev_idx),
        t_in=int(t_in),
        t_f=int(t_f),
        f_region_t_f=int(f_region_t_f),
        rank=len(singular_values),
        singular_values=singular_values,
        U_pf=U_pf,
        V1=V1,
        V2=V2,
        A_p=A_p,
        A_f=A_f,
        A_pf=A_pf,
        regions=regions,
        basis_rotation_info=basis_rotation_info,
    )


def validate_evaporation_bulk_circuit(circuit, atol=1e-8):
    """
    Numerical checks for the five-part circuit construction.

    Returns a dictionary of errors and boolean geometry checks.
    """
    N = circuit.N
    errors = {}

    errors["V1_isometry"] = np.linalg.norm(circuit.V1.conj().T @ circuit.V1 - np.eye(circuit.rank))
    errors["V2_isometry"] = np.linalg.norm(circuit.V2.conj().T @ circuit.V2 - np.eye(circuit.rank))
    D = np.diag(circuit.singular_values)
    errors["svd_reconstruction"] = np.linalg.norm(circuit.U_pf - circuit.V1 @ D @ circuit.V2.conj().T)

    for name, region in circuit.regions.items():
        max_err = 0.0
        for gate in region.gates.values():
            max_err = max(max_err, np.linalg.norm(gate.conj().T @ gate - np.eye(2 * N)))
        errors[f"{name}_max_gate_unitarity"] = max_err

    bridge_z = circuit.regions["bridge"].metadata["z_values"]
    errors["bridge_z_values"] = bridge_z
    errors["bridge_z_check"] = sorted(bridge_z.values()) == [-1, 0, 0, 1]
    errors["V1_positive_z"] = all(v - u > 0 for u, v in circuit.regions["V1"].gates)
    errors["V2_positive_z"] = all(v - u > 0 for u, v in circuit.regions["V2"].gates)
    errors["passed"] = (
        errors["V1_isometry"] < atol
        and errors["V2_isometry"] < atol
        and errors["svd_reconstruction"] < atol
        and errors["bridge_z_check"]
        and errors["V1_positive_z"]
        and errors["V2_positive_z"]
    )
    return errors


def gate_transfer_strength(U, N):
    """Return the upper-left gate-block Frobenius norm divided by N."""
    U11 = U[:N, :N]
    svals = np.linalg.svd(U11, compute_uv=False)
    return np.sqrt(np.sum(svals**2)) / N


def plot_evaporation_circuit(circuit, size=2, draw_links=True, linewidth=1.0):
    """Plot the five-region evaporation circuit in (z,t) coordinates."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    region_styles = {
        "P": {"color": "#f4b000", "marker": "D", "label": "P"},
        "F": {"color": "#5bc0de", "marker": "D", "label": "F"},
        "V1": {"color": "#d62728", "marker": "D", "label": "V1"},
        "V2": {"color": "#7b2cbf", "marker": "D", "label": "V2"},
        "bridge": {"color": "#111111", "marker": "D", "label": "bridge"},
    }

    link_segments = []
    for name, region in circuit.regions.items():
        coords = np.array(
            [(0.5 * (v - u), 0.5 * (u + v), u, v) for (u, v) in region.gates],
            dtype=float,
        )
        if coords.size == 0:
            continue

        zs = coords[:, 0]
        ts = coords[:, 1]
        us = coords[:, 2]
        vs = coords[:, 3]

        if draw_links:
            centers = coords[:, :2]
            endpoints = np.stack(
                [
                    np.column_stack((0.5 * (vs - (us + 0.5)), 0.5 * (us + 0.5 + vs))),
                    np.column_stack((0.5 * (vs - (us - 0.5)), 0.5 * (us - 0.5 + vs))),
                    np.column_stack((0.5 * (vs + 0.5 - us), 0.5 * (us + vs + 0.5))),
                    np.column_stack((0.5 * (vs - 0.5 - us), 0.5 * (us + vs - 0.5))),
                ],
                axis=1,
            )
            region_segments = np.stack(
                [
                    np.repeat(centers[:, None, :], 4, axis=1),
                    endpoints,
                ],
                axis=2,
            ).reshape(-1, 2, 2)
            link_segments.extend(region_segments)

        style = region_styles.get(
            name, {"color": "0.4", "marker": "D", "label": name}
        )
        ax.scatter(
            zs,
            ts,
            s=size,
            color=style["color"],
            marker=style["marker"],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.88,
            label=f"{style['label']} ({len(region.gates)})",
            zorder=3,
        )

    if draw_links and link_segments:
        link_segments = np.asarray(link_segments, dtype=float)
        lc = LineCollection(link_segments, colors="0.55", linewidths=linewidth, alpha=0.7, zorder=1)
        ax.add_collection(lc)
        ax.update_datalim(link_segments.reshape(-1, 2))
        ax.autoscale_view()

    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"bulk depth $z=(v-u)/2$")
    ax.set_ylabel(r"time $t=(u+v)/2$")
    ax.set_title("Five-part evaporation bulk circuit")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def collect_region_gates(circuit, regions):
    """Return gates from selected regions using their stored coordinates."""
    selected_gates = {}
    for name in regions:
        for key, U in circuit.regions[name].gates.items():
            if key in selected_gates:
                raise ValueError(f"Coordinate overlap at {key}")
            selected_gates[key] = U
    return selected_gates


def _plot_gate_strength_grid(
    gate_dict,
    N,
    dt,
    max_index,
    ax,
    vmin=None,
    vmax=None,
    title="",
    mask_negative_z=True,
):
    """Plot gate values using a pcolormesh on the circuit coordinates."""
    norms_mat = np.full((max_index, max_index), np.nan)
    for (u, v), U in gate_dict.items():
        if 0 <= u < max_index and 0 <= v < max_index:
            norms_mat[u, v] = gate_transfer_strength(U, N)

    idx = np.arange(max_index)
    UU, VV = np.meshgrid(idx, idx, indexing="ij")
    TT = dt * (UU + VV) / 2.0
    ZZ = dt * (VV - UU) / 2.0
    display = np.where((ZZ >= 0) | (not mask_negative_z), norms_mat, np.nan)

    pc = ax.pcolormesh(
        ZZ,
        TT,
        display,
        cmap="viridis",
        shading="auto",
        edgecolors="none",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(r"Bulk depth $z=(v-u)/2$")
    ax.set_ylabel(r"Time $t=(u+v)/2$")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.3)
    return pc


def plot_evaporation_gate_transfer_strength(
    circuit,
    dt=1.0,
    reference_gates=None,
    regions=("P", "V1", "bridge", "V2", "F"),
    mask_negative_z=False,
):
    """Color gates by the upper-left block Frobenius norm divided by N."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2 if reference_gates is not None else 1,
        1,
        figsize=(8, 12 if reference_gates is not None else 6),
        constrained_layout=True,
        squeeze=False,
    )

    selected_gates = collect_region_gates(circuit, regions)
    if not selected_gates:
        raise ValueError("No gates selected for plotting")

    color_source = reference_gates if reference_gates is not None else selected_gates
    color_vals = [gate_transfer_strength(U, circuit.N) for U in color_source.values()]
    vmin = min(color_vals)
    vmax = max(color_vals)
    base_t_f = getattr(circuit, "f_region_t_f", getattr(circuit, "t_f", 0))
    max_index = (
        max(
            base_t_f,
            max(max(u, v) for u, v in selected_gates),
            max(max(u, v) for u, v in color_source),
        )
        + 1
    )

    pc = _plot_gate_strength_grid(
        selected_gates,
        circuit.N,
        dt,
        max_index,
        axes[0, 0],
        vmin=vmin,
        vmax=vmax,
        title="five-part circuit",
        mask_negative_z=mask_negative_z,
    )
    fig.colorbar(pc, ax=axes[0, 0], label=r"2-norm of $U_{11}$ / $N$")

    if reference_gates is not None:
        pc_ref = _plot_gate_strength_grid(
            reference_gates,
            circuit.N,
            dt,
            max_index,
            axes[1, 0],
            vmin=vmin,
            vmax=vmax,
            title="ordinary full reconstruction reference",
            mask_negative_z=True,
        )
        fig.colorbar(pc_ref, ax=axes[1, 0], label=r"2-norm of $U_{11}$ / $N$")

    return fig, axes.ravel()


def simulate_evaporation_wavepacket(
    circuit,
    start_u,
    start_v,
    initial_state,
    is_left=False,
    regions=("P", "V1", "bridge", "V2", "F"),
):
    """
    Simulate a wavepacket through the five-part evaporation circuit.

    This is the same link convention as bulk_reconstruction_lib.simulate_wavepacket:
    each gate maps incoming [R(u, v-1), L(u-1, v)] to outgoing [L(u, v), R(u, v)].
    Unlike the ordinary wedge simulator, this version allows z=(v-u)/2<0.
    """
    N = circuit.N
    initial_state = np.asarray(initial_state, dtype=complex)
    if initial_state.shape != (N,):
        raise ValueError(f"initial_state must have shape ({N},)")

    gates = collect_region_gates(circuit, regions)
    if not gates:
        raise ValueError("No gates selected for wavepacket simulation")
    left_reflection_boundaries = set()
    for name in ("P", "F"):
        if name not in regions:
            continue
        metadata = circuit.regions[name].metadata or {}
        if "KR_all" not in metadata:
            continue
        offset = int(metadata.get("offset", 0))
        region_length = metadata["KR_all"].shape[0]
        left_reflection_boundaries.update(
            (offset + i, offset + i) for i in range(region_length)
        )

    right_reflection_boundaries = set()
    for name in ("V1", "V2"):
        if name not in regions:
            continue
        region_gates = circuit.regions[name].gates
        if not region_gates:
            continue
        max_depth = max(v - u for u, v in region_gates)
        right_reflection_boundaries.update(
            (u, v + 1) for (u, v) in region_gates if v - u == max_depth
        )

    R_links = {}
    L_links = {}
    if is_left:
        L_links[(start_u, start_v)] = initial_state
    else:
        R_links[(start_u, start_v)] = initial_state

    evolution_sites = set(gates) | left_reflection_boundaries | right_reflection_boundaries
    min_u = min([start_u] + [u for u, _ in evolution_sites]) - 1
    max_u = max([start_u] + [u for u, _ in evolution_sites]) + 1
    min_v = min([start_v] + [v for _, v in evolution_sites]) - 1
    max_v = max([start_v] + [v for _, v in evolution_sites]) + 1
    max_t_sum = max(max(u, v) * 2 for u, v in evolution_sites)
    max_t_sum = max(max_t_sum, start_u + start_v)
    zero = np.zeros(N, dtype=complex)

    for t_sum in range(start_u + start_v + 1, max_t_sum + 2):
        for u in range(min_u, max_u + 1):
            v = t_sum - u
            if v < min_v or v > max_v:
                continue

            r_known = (u, v - 1) in R_links
            l_known = (u - 1, v) in L_links
            if not (r_known or l_known):
                continue

            r_in = R_links.get((u, v - 1), zero)
            l_in = L_links.get((u - 1, v), zero)

            if (u, v) in gates:
                out = gates[(u, v)] @ np.concatenate([r_in, l_in])
                L_links[(u, v)] = out[:N]
                R_links[(u, v)] = out[N:]
            elif (u, v) in right_reflection_boundaries:
                if r_known:
                    L_links[(u, v)] = r_in
            elif (u, v) in left_reflection_boundaries:
                R_links[(u, v)] = l_in

    return R_links, L_links


def _wavepacket_link_matrices(R_links, L_links, N, dt):
    """Convert sparse wavepacket links to fine-grid matrices for pcolormesh."""
    all_items = [(key, "R", val) for key, val in R_links.items()]
    all_items += [(key, "L", val) for key, val in L_links.items()]
    if not all_items:
        raise ValueError("No wavepacket links to plot")

    Z_op = np.zeros(N)
    Z_op[0::2] = 1.0
    Z_op[1::2] = -1.0

    t_indices = []
    z_indices = []
    for (u, v), chirality, _ in all_items:
        t_indices.append(u + v)
        z_indices.append(v - u if chirality == "R" else v - u - 1)

    t_min = min(t_indices)
    t_max = max(t_indices)
    z_min = min(z_indices)
    z_max = max(z_indices)

    wp_mat = np.full((t_max - t_min + 1, z_max - z_min + 1), np.nan)
    wp_z_mat = np.full_like(wp_mat, np.nan)

    for (u, v), chirality, vec in all_items:
        t_idx = u + v - t_min
        z_raw = v - u if chirality == "R" else v - u - 1
        z_idx = z_raw - z_min
        intensity = np.sum(np.abs(vec) ** 2)
        z_value = np.sum(Z_op * np.abs(vec) ** 2)
        if np.isnan(wp_mat[t_idx, z_idx]):
            wp_mat[t_idx, z_idx] = 0.0
            wp_z_mat[t_idx, z_idx] = 0.0
        wp_mat[t_idx, z_idx] += intensity
        wp_z_mat[t_idx, z_idx] += z_value

    t_fine = (np.arange(t_min, t_max + 1) + 0.5) * dt / 2.0
    z_fine = (np.arange(z_min, z_max + 1) + 0.5) * dt / 2.0
    return wp_mat, wp_z_mat, z_fine, t_fine


def plot_evaporation_wavepacket_propagation(
    R_links,
    L_links,
    N,
    dt=1.0,
    intensity_vmax=1.0,
    z_limit=1.0,
):
    """Plot wavepacket propagation through the five-part circuit."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    wp_mat, wp_z_mat, z_fine, t_fine = _wavepacket_link_matrices(R_links, L_links, N, dt)

    fig_wp_int = plt.figure(figsize=(10, 8))
    ax1 = fig_wp_int.add_subplot(111)
    colors = [(0, 1, 1), (1, 0, 0)]
    cmap_cr = LinearSegmentedColormap.from_list("cyan_red", colors)
    pc_wp = ax1.pcolormesh(
        z_fine,
        t_fine,
        wp_mat,
        cmap=cmap_cr,
        shading="auto",
        edgecolors="none",
        vmin=0,
        vmax=intensity_vmax,
    )
    plt.colorbar(pc_wp, label=r"Intensity $|\psi|^2$")
    ax1.set_ylabel(r"Time $t$")
    ax1.set_xlabel(r"Bulk depth $z$")
    ax1.set_title("Five-part circuit wavepacket propagation (intensity)")
    ax1.grid(True, linestyle=":", alpha=0.3)

    fig_wp_z = plt.figure(figsize=(10, 8))
    ax2 = fig_wp_z.add_subplot(111)
    pc_z = ax2.pcolormesh(
        z_fine,
        t_fine,
        wp_z_mat,
        cmap="RdBu_r",
        shading="auto",
        edgecolors="none",
        vmin=-z_limit,
        vmax=z_limit,
    )
    plt.colorbar(pc_z, label=r"$\langle \psi | Z | \psi \rangle$")
    ax2.set_xlabel(r"Bulk depth $z$")
    ax2.set_ylabel(r"Time $t$")
    ax2.set_title("Five-part circuit wavepacket propagation (Z-component)")
    ax2.grid(True, linestyle=":", alpha=0.3)

    return fig_wp_int, fig_wp_z, wp_mat, wp_z_mat, z_fine, t_fine
