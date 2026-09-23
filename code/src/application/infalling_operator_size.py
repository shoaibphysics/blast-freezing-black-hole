import numpy as np
import scipy.linalg as la

from large_p.evap_syk import compute_size_operator_DeltaNchi_grid


def complete_isometry_complement(V, tol=1e-12):
    """Return an orthonormal complement V_perp for an isometry V."""
    V = np.asarray(V)
    if V.ndim != 2:
        raise ValueError("V must be a matrix")
    if np.linalg.norm(V.conj().T @ V - np.eye(V.shape[1])) > 100 * tol:
        raise ValueError("V is not an isometry within tolerance")
    return la.null_space(V.conj().T)


def isometry_unitary_from_diamond_gates(gates, ell, k, N):
    """
    Reconstruct the full unitary from the gates made by isometry_to_diamond_gates.

    The gate constructor peels an isometry V to [I; 0] by applying the adjoint
    circuit in the order j=0,1,... and decreasing row within each layer.  This
    function multiplies those same adjoint gates into P and returns U=P^dagger,
    whose first k*N columns are V and whose remaining columns are the circuit
    ordered complement.
    """
    dim = ell * N
    peeled_unitary = np.eye(dim, dtype=complex)

    for j in range(k):
        for row in range(ell - 2, j - 1, -1):
            key = (-j, row - j)
            if key not in gates:
                raise KeyError(f"Missing V1 local gate {key}")
            row_slice = slice(row * N, (row + 2) * N)
            adjoint_gate = gates[key].conj().T
            peeled_unitary[row_slice, :] = adjoint_gate @ peeled_unitary[row_slice, :]

    return peeled_unitary.conj().T


def v1_circuit_complement(circuit, tol=1e-9):
    """
    Return V1_perp from the actual V1 gate circuit, not an arbitrary null space.

    The columns are ordered by the completed input legs of the isometry circuit,
    i.e. by the columns k*N:(ell_p*N) of the full circuit unitary.
    """
    N = circuit.N
    ell_p = circuit.A_p.shape[0] // N
    k = circuit.rank // N
    local_gates = circuit.regions["V1"].metadata["local_gates"]
    U_v1 = isometry_unitary_from_diamond_gates(local_gates, ell=ell_p, k=k, N=N)

    first_columns_err = np.linalg.norm(U_v1[:, : circuit.rank] - circuit.V1)
    if first_columns_err > 1000 * tol:
        raise ValueError(
            "The V1 gate product does not reproduce circuit.V1 within tolerance: "
            f"error={first_columns_err:g}"
        )
    return U_v1[:, circuit.rank :]


def past_horizon_right_kernel(circuit):
    """
    Stack K_{P,R}(u, v=ell_p-1) for all past-region outputs.

    The returned matrix maps boundary fermions chi_P to the right movers
    psi_P at the horizon edge, ordered by the V1 circuit output edge.  Since
    V1 is embedded with the up-down reflection

        local output site s -> global u = tev_idx - 1 - s,

    the matching P-horizon rows are ordered by decreasing local P coordinate u.
    """
    p_meta = circuit.regions["P"].metadata
    KR_all = p_meta["KR_all"]
    ell_p = circuit.A_p.shape[0] // circuit.N
    horizon_v = ell_p - 1
    rows = [KR_all[u, horizon_v] for u in range(ell_p - 1, -1, -1)]
    return np.vstack(rows)


def infalling_boundary_kernel(circuit, tol=1e-12):
    """
    Boundary kernel for infalling modes psi_I = V1_perp^dag K_{P,R} chi_P.

    Returns an array of shape ((ell_p-k)*N, ell_p*N).
    """
    V1_perp = v1_circuit_complement(circuit, tol=tol)
    K_PR = past_horizon_right_kernel(circuit)
    return V1_perp.conj().T @ K_PR


def contract_kernel_with_DeltaNchi(K, DeltaNchi_time, N, tol=1e-14):
    """
    Contract rows of K against DeltaNchi_time \\otimes I_N.

    K has shape (num_modes, NT*N).  Returns one real size per row.
    """
    K = np.asarray(K)
    NT = DeltaNchi_time.shape[0]
    if K.shape[1] != NT * N:
        raise ValueError(f"K has {K.shape[1]} columns, expected {NT * N}")

    V = K.reshape(K.shape[0], NT, N)
    row_sizes = np.full(K.shape[0], np.nan)
    for r in range(K.shape[0]):
        support = np.any(np.abs(V[r]) > tol, axis=1)
        if not np.any(support):
            continue
        V_sub = V[r, support, :]
        D_sub = DeltaNchi_time[np.ix_(support, support)]
        row_sizes[r] = np.real(np.einsum("ta,ts,sa->", np.conj(V_sub), D_sub, V_sub))
    return row_sizes


def group_flavor_sizes(row_sizes, N):
    """Average row sizes over each N-flavor infalling site."""
    row_sizes = np.asarray(row_sizes)
    if row_sizes.size % N != 0:
        raise ValueError("Number of row sizes must be a multiple of N")
    grouped = row_sizes.reshape(row_sizes.size // N, N)
    return np.nanmean(grouped, axis=1), grouped


def compute_infalling_operator_size(circuit, tgrid, params, tf=None, tol=1e-12):
    """
    Compute Delta N_chi^{in}(n; tf) for the infalling modes of V1.

    The past boundary time grid is tgrid[t_in:tev_idx], matching A_p.
    """
    if tf is None:
        tf = float(tgrid[-1])

    past_tgrid = np.asarray(tgrid[circuit.t_in : circuit.tev_idx])
    DeltaNchi = compute_size_operator_DeltaNchi_grid(
        past_tgrid, tf, params, invalid_value=0.0 + 0.0j
    )
    DeltaNchi_time = np.real(DeltaNchi)
    K_in = infalling_boundary_kernel(circuit, tol=tol)
    row_sizes = contract_kernel_with_DeltaNchi(K_in, DeltaNchi_time, circuit.N)
    site_sizes, flavor_sizes = group_flavor_sizes(row_sizes, circuit.N)
    k = circuit.rank // circuit.N
    completed_input_site = np.arange(k, k + site_sizes.size)

    return {
        "tf": float(tf),
        "past_tgrid": past_tgrid,
        "DeltaNchi_time": DeltaNchi_time,
        "K_in": K_in,
        "row_sizes": row_sizes,
        "site_sizes": site_sizes,
        "flavor_sizes": flavor_sizes,
        "mode_index": np.arange(site_sizes.size),
        "completed_input_site": completed_input_site,
        "horizon_u_index": circuit.tev_idx - 1 - completed_input_site,
    }


def compute_infalling_operator_size_scan(circuit, tgrid, params, tf_grid, tol=1e-12):
    """Compute infalling operator sizes for a sequence of final times."""
    tf_grid = np.asarray(tf_grid, dtype=float)
    scans = [
        compute_infalling_operator_size(circuit, tgrid, params, tf=tf, tol=tol)
        for tf in tf_grid
    ]
    size_mat = np.vstack([scan["site_sizes"] for scan in scans])
    return {
        "tf_grid": tf_grid,
        "mode_index": scans[0]["mode_index"] if scans else np.array([]),
        "size_mat": size_mat,
        "scans": scans,
    }


def plot_infalling_operator_size(infall_size, ax=None, marker="o", lw=1.5):
    """Plot Delta N_chi for one final time as a function of infalling mode index."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    else:
        fig = ax.figure

    x = infall_size.get("horizon_u_index", infall_size["mode_index"])
    xlabel = r"horizon coordinate $u$" if "horizon_u_index" in infall_size else r"infalling mode index $n$"
    ax.plot(
        x,
        infall_size["site_sizes"],
        marker=marker,
        lw=lw,
    )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta N_\chi^{\mathrm{in}}(n;t_f)$")
    ax.set_title(rf"Infalling operator size at $t_f={infall_size['tf']:.3g}$")
    ax.grid(True, linestyle=":", alpha=0.35)
    return fig, ax


def plot_infalling_operator_size_scan(
    infall_scan,
    tev=None,
    ax=None,
    cmap="coolwarm",
    symmetric=True,
):
    """Plot a heatmap of infalling operator size versus mode index and final time."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    else:
        fig = ax.figure

    S = infall_scan["size_mat"]
    vmin = vmax = None
    if symmetric and np.any(np.isfinite(S)):
        finite_abs_max = np.nanmax(np.abs(S))
        vmin = -finite_abs_max
        vmax = finite_abs_max

    im = ax.pcolormesh(
        infall_scan["mode_index"],
        infall_scan["tf_grid"],
        S,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=r"$\Delta N_\chi^{\mathrm{in}}(n;t_f)$")
    if tev is not None:
        ax.axhline(tev, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"infalling mode index $n$")
    ax.set_ylabel(r"final time $t_f$")
    ax.set_title(r"Infalling operator size vs final time")
    ax.grid(True, linestyle=":", alpha=0.25)
    return fig, ax, im
