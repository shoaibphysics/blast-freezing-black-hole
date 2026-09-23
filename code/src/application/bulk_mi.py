import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from bulk_reconstruction.reconstruction import compute_bulk_gates
from finite_p.replica import (
    get_two_replica_eta_block,
    two_replica_eta_contraction,
    eta_mutual_information_at_tb,
)

def _maybe_dataframe(rows):
    """Return a pandas DataFrame when pandas is available, otherwise raw rows."""

    if pd is None:
        return rows
    return pd.DataFrame(rows)


def real_time_grid_from_meta(meta):
    """
    Physical real-time grid used by the contour.

    The contour real-time points are

        dt, 2 dt, ..., t_f.

    There is no t=0 real-time point in this discretization.
    """

    return (np.arange(meta.N_t) + 1) * meta.dt


def upper_real_time_indices(meta):
    """Upper real-time contour indices in physical-time order."""

    return meta.N_beta + np.arange(meta.N_t)


def branch_indices_for_all_real_times(meta, branch):
    """
    Return contour indices for all real-time points on one branch.

    The returned ordering matches real_time_grid_from_meta(meta).
    """

    if branch in {"upper", "+", "forward"}:
        return meta.N_beta + np.arange(meta.N_t)

    if branch in {"lower", "-", "backward"}:
        n_t = np.arange(1, meta.N_t + 1)
        return meta.N_fold + (meta.N_t - n_t)

    raise ValueError("branch must be upper/+ or lower/-")


def kernel_for_branch(K, branch):
    """
    Return branch-dependent kernel coefficients.

    Upper branch means ket insertion, so use K.
    Lower branch means bra insertion, so use complex conjugate K.
    """

    K = np.asarray(K, dtype=np.complex128)

    if branch in {"upper", "+", "forward"}:
        return K

    if branch in {"lower", "-", "backward"}:
        return np.conj(K)

    raise ValueError("branch must be upper/+ or lower/-")


def eta_reconstruction_GRR_GRL_from_two_replica_disconnected(
    G_disc,
    meta,
    *,
    replica_index=1,
):
    """
    Build the normalized GRR and GRL matrices used by HKLL.

    We use one diagonal replica block of the disconnected saddle. This is the
    untwisted background anticommutator data used to construct HKLL kernels.
    """

    NT = meta.N_t
    idx = upper_real_time_indices(meta)

    raw_RR = get_two_replica_eta_block(
        G_disc,
        meta,
        replica_index,
        replica_index,
        "R",
        "R",
    )[np.ix_(idx, idx)]

    raw_RL = get_two_replica_eta_block(
        G_disc,
        meta,
        replica_index,
        replica_index,
        "R",
        "L",
    )[np.ix_(idx, idx)]

    GRR = np.eye(NT, dtype=np.complex128)
    GRL = np.full((NT, NT), np.nan + 0.0j, dtype=np.complex128)

    for j in range(NT):
        rows = np.arange(NT)

        # Avoid the equal-time contact point from the numerical contour.
        # The normalized anticommutator has diagonal 1.
        after_contact = rows > j
        GRR[rows[after_contact], j] = 2j * raw_RR[rows[after_contact], j]

        # RL has no same UV sign-contact issue, so fill the full column.
        GRL[:, j] = -2.0 * raw_RL[:, j]

    # Fill the opposite triangle by Hermiticity.
    lower = np.tril_indices(NT, k=-1)

    GRR[(lower[1], lower[0])] = np.conj(GRR[lower])
    GRL[(lower[1], lower[0])] = np.conj(GRL[lower])
    GRL = (GRL + GRL.conj().T) / 2.0

    if not (np.all(np.isfinite(GRR)) and np.all(np.isfinite(GRL))):
        raise RuntimeError("failed to assemble finite GRR/GRL matrices")

    return GRR, GRL


def anticommutator_matrix_from_GRR_GRL(GRR, GRL):
    """Build CorM and A in the same convention as the HKLL code."""

    I2 = np.eye(2, dtype=np.complex128)
    J2 = np.array([[0, 1], [-1, 0]], dtype=np.complex128)

    CorM = np.kron(GRR, I2) + np.kron(GRL, -1j * J2)
    CorM = (CorM + CorM.conj().T) / 2.0

    A = np.real(CorM)
    A = (A + A.T) / 2.0

    return CorM, A


def build_hkll_kernels_from_disconnected_eta(
    G_disc,
    meta,
    *,
    replica_index=1,
    N_bulk=2,
):
    """
    Build the HKLL reconstruction input and kernels from disconnected eta data.

    Returns a dictionary containing GRR, GRL, CorM, A, gates, KL_all, KR_all,
    is_legit, tf_list, tini_list, and tgrid.
    """

    GRR, GRL = eta_reconstruction_GRR_GRL_from_two_replica_disconnected(
        G_disc,
        meta,
        replica_index=replica_index,
    )

    CorM, A = anticommutator_matrix_from_GRR_GRL(GRR, GRL)

    gates, KL_all, KR_all, is_legit, tf_list, tini_list = compute_bulk_gates(
        A,
        N_bulk,
        meta.N_t,
    )

    return {
        "GRR": GRR,
        "GRL": GRL,
        "CorM": CorM,
        "A": A,
        "gates": gates,
        "KL_all": KL_all,
        "KR_all": KR_all,
        "is_legit": is_legit,
        "tf_list": tf_list,
        "tini_list": tini_list,
        "tgrid": real_time_grid_from_meta(meta),
        "N_bulk": int(N_bulk),
    }


def two_replica_reconstruction_matrix(
    G_two,
    meta,
    *,
    replica_1,
    replica_2,
    branch_1,
    branch_2,
):
    """
    Build the 2N_t x 2N_t real-time reconstruction-basis matrix.

    The HKLL kernel acts on the flattened boundary vector ordered as

        time 1 component 1,
        time 1 component 2,
        time 2 component 1,
        time 2 component 2,
        ...

    The conversion matches the bulk reconstruction convention:

        GRR =  2 i G_RR
        GRL = -2   G_RL

    and then

        CorM = kron(GRR, I2) + kron(GRL, -i J2).
    """

    idx_1 = branch_indices_for_all_real_times(meta, branch_1)
    idx_2 = branch_indices_for_all_real_times(meta, branch_2)

    raw_RR = get_two_replica_eta_block(
        G_two,
        meta,
        replica_1,
        replica_2,
        "R",
        "R",
    )[np.ix_(idx_1, idx_2)]

    raw_RL = get_two_replica_eta_block(
        G_two,
        meta,
        replica_1,
        replica_2,
        "R",
        "L",
    )[np.ix_(idx_1, idx_2)]

    GRR = 2j * raw_RR
    GRL = -2.0 * raw_RL

    I2 = np.eye(2, dtype=np.complex128)
    J2 = np.array([[0, 1], [-1, 0]], dtype=np.complex128)

    return np.kron(GRR, I2) + np.kron(GRL, -1j * J2)


def smeared_two_replica_contraction(
    G_two,
    meta,
    K1,
    K2,
    *,
    replica_1,
    replica_2,
    branch_1,
    branch_2,
):
    """
    Compute the scalar contraction

        K1(branch_1) M K2(branch_2)^T

    where M is the real-time two-replica eta Green's function in the
    reconstruction basis.
    """

    M = two_replica_reconstruction_matrix(
        G_two,
        meta,
        replica_1=replica_1,
        replica_2=replica_2,
        branch_1=branch_1,
        branch_2=branch_2,
    )

    K1b = kernel_for_branch(K1, branch_1)
    K2b = kernel_for_branch(K2, branch_2)

    expected = 2 * meta.N_t

    if K1b.shape != (expected,):
        raise ValueError(f"K1 has shape {K1b.shape}, expected {(expected,)}")

    if K2b.shape != (expected,):
        raise ValueError(f"K2 has shape {K2b.shape}, expected {(expected,)}")

    return complex(K1b @ M @ K2b.T)


def boundary_delta_kernel(meta, t_b, *, component=0, N_bulk=2):
    """
    Boundary delta kernel at physical time t_b.

    component=0 selects the first reconstruction-basis component, whose
    self-correlator corresponds to the RR channel.
    """

    n = int(round(t_b / meta.dt)) - 1

    if abs(t_b / meta.dt - round(t_b / meta.dt)) > 1e-9:
        raise ValueError(f"t_b={t_b} is not aligned with dt={meta.dt}")

    if n < 0 or n >= meta.N_t:
        raise ValueError(f"t_b={t_b} is outside the real-time grid")

    if component < 0 or component >= N_bulk:
        raise ValueError(f"component={component} is outside 0,...,{N_bulk - 1}")

    K = np.zeros(N_bulk * meta.N_t, dtype=np.complex128)
    K[N_bulk * n + component] = 1.0

    return K


def component_delta_kernel(meta, t_b, component, *, N_bulk=2):
    """Alias for a boundary delta kernel with an explicit component."""

    return boundary_delta_kernel(meta, t_b, component=component, N_bulk=N_bulk)


def real_part_with_warning(z, name, imag_tol=1e-7):
    """Return real part, while warning if the imaginary part is large."""

    z = complex(z)

    if abs(z.imag) > imag_tol:
        print(f"warning: {name} has imaginary part {z.imag:g}")

    return float(z.real)


def bulk_mutual_information_from_kernel(
    G_twisted,
    meta,
    K,
    *,
    imag_tol=1e-7,
):
    """
    Compute F2[K], F4[K], K2[K], and the second Renyi mutual informations.

    This function does not know where K came from. K can be a boundary delta
    kernel, a right-moving HKLL row, or a left-moving HKLL row.
    """

    C11_mp = smeared_two_replica_contraction(
        G_twisted,
        meta,
        K,
        K,
        replica_1=1,
        replica_2=1,
        branch_1="lower",
        branch_2="upper",
    )

    C22_mp = smeared_two_replica_contraction(
        G_twisted,
        meta,
        K,
        K,
        replica_1=2,
        replica_2=2,
        branch_1="lower",
        branch_2="upper",
    )

    C12_mp = smeared_two_replica_contraction(
        G_twisted,
        meta,
        K,
        K,
        replica_1=1,
        replica_2=2,
        branch_1="lower",
        branch_2="upper",
    )

    C21_mp = smeared_two_replica_contraction(
        G_twisted,
        meta,
        K,
        K,
        replica_1=2,
        replica_2=1,
        branch_1="lower",
        branch_2="upper",
    )

    C12_mm = smeared_two_replica_contraction(
        G_twisted,
        meta,
        K,
        K,
        replica_1=1,
        replica_2=2,
        branch_1="lower",
        branch_2="lower",
    )

    C12_pp = smeared_two_replica_contraction(
        G_twisted,
        meta,
        K,
        K,
        replica_1=1,
        replica_2=2,
        branch_1="upper",
        branch_2="upper",
    )

    F2 = C12_mp
    K2 = C11_mp
    F4 = C11_mp * C22_mp - C12_mp * C21_mp - C12_mm * C12_pp

    F2_real = real_part_with_warning(F2, "F2", imag_tol=imag_tol)
    F4_real = real_part_with_warning(F4, "F4", imag_tol=imag_tol)
    K2_real = real_part_with_warning(K2, "K2", imag_tol=imag_tol)

    if F4_real > 0:
        exp_minus_I2_classical = (1.0 + 2.0 * F2_real + F4_real) / (
            4.0 * np.sqrt(F4_real)
        )
        I2_classical = -np.log(exp_minus_I2_classical)
    else:
        exp_minus_I2_classical = np.nan
        I2_classical = np.nan

    quantum_denominator = 1.0 + 2.0 * K2_real + F4_real

    if quantum_denominator > 0:
        exp_minus_I2_quantum = 0.5 * (1.0 + 2.0 * F2_real + F4_real) / (
            quantum_denominator
        )
        I2_quantum = -np.log(exp_minus_I2_quantum)
    else:
        exp_minus_I2_quantum = np.nan
        I2_quantum = np.nan

    return {
        "t_f": float(meta.t_f),
        "F2": F2,
        "F2_real": F2_real,
        "F2_imag": float(np.imag(F2)),
        "F4": F4,
        "F4_real": F4_real,
        "F4_imag": float(np.imag(F4)),
        "K2": K2,
        "K2_real": K2_real,
        "K2_imag": float(np.imag(K2)),
        "exp_minus_I2_classical": exp_minus_I2_classical,
        "I2_classical": I2_classical,
        "exp_minus_I2_quantum": exp_minus_I2_quantum,
        "I2_quantum": I2_quantum,
        "C11_minus_plus": C11_mp,
        "C22_minus_plus": C22_mp,
        "C12_minus_plus": C12_mp,
        "C21_minus_plus": C21_mp,
        "C12_minus_minus": C12_mm,
        "C12_plus_plus": C12_pp,
    }


def uv_from_bulk_tz(meta, t_bulk, z_bulk):
    """
    Convert physical bulk coordinates to HKLL array indices.

    The real-time grid is dt, 2dt, ..., t_f, so physical time t=(n+1)dt
    corresponds to array index n.
    """

    u_float = (t_bulk - z_bulk) / meta.dt - 1.0
    v_float = (t_bulk + z_bulk) / meta.dt - 1.0

    u = int(round(u_float))
    v = int(round(v_float))

    if abs(u - u_float) > 1e-9 or abs(v - v_float) > 1e-9:
        raise ValueError(
            f"(t,z)=({t_bulk},{z_bulk}) is not aligned with dt={meta.dt}"
        )

    if u < 0 or v < 0 or u >= meta.N_t or v >= meta.N_t or v < u:
        raise ValueError(f"invalid bulk indices u={u}, v={v}")

    return u, v


def select_bulk_kernel(
    KR_all,
    KL_all,
    is_legit,
    meta,
    *,
    t_bulk,
    z_bulk,
    direction="R",
    r=0,
    use_boundary_delta_at_z0=True,
    boundary_component=None,
):
    """
    Select one kernel vector K.

    If z_bulk=0 and use_boundary_delta_at_z0=True, return the exact boundary
    delta kernel. Otherwise return an HKLL row from KR_all or KL_all.
    """

    u, v = uv_from_bulk_tz(meta, t_bulk, z_bulk)

    if boundary_component is None:
        boundary_component = r

    if abs(z_bulk) < 1e-12 and use_boundary_delta_at_z0:
        K = boundary_delta_kernel(meta, t_bulk, component=boundary_component)
        return {
            "K": K,
            "u": u,
            "v": v,
            "r": int(r),
            "direction": "boundary-delta",
            "t": float(t_bulk),
            "z": 0.0,
            "is_legit": True,
            "K_norm": float(np.linalg.norm(K)),
            "K_max": float(np.max(np.abs(K))),
        }

    if not is_legit[u, v]:
        raise ValueError(f"bulk point u={u}, v={v} is not marked legitimate")

    direction = direction.upper()

    if direction == "R":
        K = KR_all[u, v, r, :]
    elif direction == "L":
        K = KL_all[u, v, r, :]
    else:
        raise ValueError("direction must be 'R' or 'L'")

    K = np.asarray(K, dtype=np.complex128)
    tgrid = real_time_grid_from_meta(meta)

    return {
        "K": K,
        "u": u,
        "v": v,
        "r": int(r),
        "direction": direction,
        "t": float(0.5 * (tgrid[u] + tgrid[v])),
        "z": float(0.5 * (tgrid[v] - tgrid[u])),
        "is_legit": bool(is_legit[u, v]),
        "K_norm": float(np.linalg.norm(K)),
        "K_max": float(np.max(np.abs(K))),
    }


def bulk_mi_scan_in_z(
    G_twisted,
    meta,
    KR_all,
    KL_all,
    is_legit,
    *,
    t_bulk,
    z_values,
    direction="R",
    r=0,
    A=None,
    max_kernel_norm_warning=50.0,
    use_boundary_delta_at_z0=True,
    imag_tol=1e-7,
):
    """Compute bulk mutual information at fixed t as z is varied."""

    rows = []

    for z_bulk in z_values:
        try:
            K_info = select_bulk_kernel(
                KR_all,
                KL_all,
                is_legit,
                meta,
                t_bulk=float(t_bulk),
                z_bulk=float(z_bulk),
                direction=direction,
                r=r,
                use_boundary_delta_at_z0=use_boundary_delta_at_z0,
            )

            result = bulk_mutual_information_from_kernel(
                G_twisted,
                meta,
                K_info["K"],
                imag_tol=imag_tol,
            )

            K = K_info["K"]

            row = {
                "bulk_t": float(t_bulk),
                "bulk_z": float(K_info["z"]),
                "u": K_info["u"],
                "v": K_info["v"],
                "direction": K_info["direction"],
                "r": K_info["r"],
                "is_legit": K_info["is_legit"],
                "K_norm": K_info["K_norm"],
                "K_max": K_info["K_max"],
                "kernel_warning": bool(K_info["K_norm"] > max_kernel_norm_warning),
                "F2_real": result["F2_real"],
                "F4_real": result["F4_real"],
                "K2_real": result["K2_real"],
                "I2_classical": result["I2_classical"],
                "I2_quantum": result["I2_quantum"],
                "F2_imag": result["F2_imag"],
                "F4_imag": result["F4_imag"],
                "K2_imag": result["K2_imag"],
            }

            if A is not None:
                KAK = np.conj(K) @ A @ K
                row["KAK_real"] = float(np.real(KAK))
                row["KAK_imag"] = float(np.imag(KAK))

            rows.append(row)

        except Exception as exc:
            rows.append(
                {
                    "bulk_t": float(t_bulk),
                    "bulk_z": float(z_bulk),
                    "direction": direction,
                    "r": int(r),
                    "is_legit": False,
                    "error": str(exc),
                }
            )

    return _maybe_dataframe(rows)


def bulk_mi_scan_points(
    G_twisted,
    meta,
    KR_all,
    KL_all,
    is_legit,
    points,
    *,
    A=None,
    default_direction="R",
    default_r=0,
    max_kernel_norm_warning=50.0,
    use_boundary_delta_at_z0=True,
    imag_tol=1e-7,
):
    """
    Compute bulk mutual information on a list of points.

    Each point can be a dictionary with keys t_bulk, z_bulk, direction, and r.
    """

    rows = []

    for point in points:
        t_bulk = float(point["t_bulk"])
        z_bulk = float(point["z_bulk"])
        direction = point.get("direction", default_direction)
        r = int(point.get("r", default_r))

        table = bulk_mi_scan_in_z(
            G_twisted,
            meta,
            KR_all,
            KL_all,
            is_legit,
            t_bulk=t_bulk,
            z_values=[z_bulk],
            direction=direction,
            r=r,
            A=A,
            max_kernel_norm_warning=max_kernel_norm_warning,
            use_boundary_delta_at_z0=use_boundary_delta_at_z0,
            imag_tol=imag_tol,
        )

        if pd is None:
            rows.extend(table)
        else:
            rows.extend(table.to_dict("records"))

    return _maybe_dataframe(rows)


def check_boundary_delta_matches_boundary_mi(
    G_twisted,
    meta,
    *,
    times=None,
    component=0,
    imag_tol=1e-7,
):
    """
    Check that the z=0 delta-kernel calculation matches eta_mutual_information_at_tb.

    This is the boundary consistency check for component=0/R.
    """

    if times is None:
        times = real_time_grid_from_meta(meta)
        times = times[times < meta.t_f - 1e-12]

    rows = []

    for t_b in times:
        direct = eta_mutual_information_at_tb(
            G_twisted,
            meta,
            float(t_b),
            imag_tol=imag_tol,
        )

        K = boundary_delta_kernel(meta, float(t_b), component=component)
        smeared = bulk_mutual_information_from_kernel(
            G_twisted,
            meta,
            K,
            imag_tol=imag_tol,
        )

        rows.append(
            {
                "t_b": float(t_b),
                "F2_direct": direct["F2_real"],
                "F2_kernel": smeared["F2_real"],
                "F2_diff": smeared["F2_real"] - direct["F2_real"],
                "F4_direct": direct["F4_real"],
                "F4_kernel": smeared["F4_real"],
                "F4_diff": smeared["F4_real"] - direct["F4_real"],
                "K2_direct": direct["K2_real"],
                "K2_kernel": smeared["K2_real"],
                "K2_diff": smeared["K2_real"] - direct["K2_real"],
                "I2_classical_direct": direct["I2_classical"],
                "I2_classical_kernel": smeared["I2_classical"],
                "I2_classical_diff": smeared["I2_classical"] - direct["I2_classical"],
                "I2_quantum_direct": direct["I2_quantum"],
                "I2_quantum_kernel": smeared["I2_quantum"],
                "I2_quantum_diff": smeared["I2_quantum"] - direct["I2_quantum"],
            }
        )

    return _maybe_dataframe(rows)


def check_reconstruction_basis_against_direct(
    G_two,
    meta,
    *,
    sample_times=None,
    branches=("upper", "lower"),
    replicas=(1, 2),
    components=(0, 1),
    N_bulk=2,
):
    """
    Check that the reconstruction-basis matrix reproduces direct component values.

    This tests RR, RL, LR, and LL by inserting component delta kernels.
    """

    if sample_times is None:
        sample_times = real_time_grid_from_meta(meta)

    component_to_species = {
        0: "R",
        1: "L",
    }

    rows = []

    for replica_1 in replicas:
        for replica_2 in replicas:
            for branch_1 in branches:
                for branch_2 in branches:
                    for t1 in sample_times:
                        for t2 in sample_times:
                            for component_1 in components:
                                for component_2 in components:
                                    K1 = component_delta_kernel(
                                        meta,
                                        float(t1),
                                        component_1,
                                        N_bulk=N_bulk,
                                    )
                                    K2 = component_delta_kernel(
                                        meta,
                                        float(t2),
                                        component_2,
                                        N_bulk=N_bulk,
                                    )

                                    from_matrix = smeared_two_replica_contraction(
                                        G_two,
                                        meta,
                                        K1,
                                        K2,
                                        replica_1=replica_1,
                                        replica_2=replica_2,
                                        branch_1=branch_1,
                                        branch_2=branch_2,
                                    )

                                    direct = two_replica_eta_contraction(
                                        G_two,
                                        meta,
                                        replica_1=replica_1,
                                        replica_2=replica_2,
                                        branch_1=branch_1,
                                        branch_2=branch_2,
                                        t1=float(t1),
                                        t2=float(t2),
                                        species_1=component_to_species[component_1],
                                        species_2=component_to_species[component_2],
                                    )

                                    rows.append(
                                        {
                                            "replica_1": replica_1,
                                            "replica_2": replica_2,
                                            "branch_1": branch_1,
                                            "branch_2": branch_2,
                                            "t1": float(t1),
                                            "t2": float(t2),
                                            "component_1": component_1,
                                            "component_2": component_2,
                                            "species_1": component_to_species[component_1],
                                            "species_2": component_to_species[component_2],
                                            "from_matrix": from_matrix,
                                            "direct": direct,
                                            "abs_diff": abs(from_matrix - direct),
                                        }
                                    )

    return _maybe_dataframe(rows)