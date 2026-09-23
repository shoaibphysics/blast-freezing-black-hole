import numpy as np
import scipy.linalg as la
import warnings

def light_cone_kernel(G, direction):
    r"""
    Computes the light cone kernel from the correlation matrix G using Cholesky decomposition.
    For the right-moving direction 'R', we want an upper triangular factor R of G such that R @ R^H = G
    For the left-moving direction 'L', we want a lower triangular factor R of G such that R @ R^H = G
    In both cases, K = R^-1 is upper triangular and K @ G @ K^H = I
    Returns the kernel K and the Cholesky factor R.
    The bulk operators are defined by $\psi_R(u,v)=\sum_{t=u}^vK^f(u,v|t)\chi(t)$ and $\psi_{L}(u,v)=\sum_{t=u}^vK^{p}(u,v|t)\chi(t)$
    """
    # Ensure G is exactly Hermitian/Symmetric
    G_sym = (G + np.conj(G).T) / 2
    
    try:
        if direction == 'L':
            
            # Use upper Cholesky factors to match MATLAB QR(reversed_isG)
            # Standard Cholesky: G_sym = L @ L^H
            R = la.cholesky(G_sym, lower=True)
            
            # Check for near-singularity
            # If diag elements are too small, kernels blow up
            diag_R = np.abs(np.diag(R))
            if np.any(diag_R < 1e-8):
                raise ValueError("Matrix G is nearly singular or not positive definite.")
            
            # K_S is lower triangular: K_S @ G_rev @ K_S^H = I
            K = la.solve_triangular(R, np.eye(G.shape[0]), lower=True)
            
        elif direction == 'R':
            # Forward direction
            # We want an upper triangular factor R of G such that R @ R^H = G
            # Then K_tri = R^-1 is upper triangular and K_tri @ G @ K_tri^H = I
            # We get R from the reversed Cholesky: P @ G @ P = L @ L^H => G = (P @ L) @ (L^H @ P)
            # R = P @ L @ P is upper triangular
            G_rev = G_sym[::-1, ::-1]
            L = la.cholesky(G_rev, lower=True)
            
            diag_L = np.abs(np.diag(L))
            if np.any(diag_L < 1e-8):
                raise ValueError("Matrix G is nearly singular or not positive definite.")
            # R is upper triangular: R @ G @ R^H = I
            R = L[::-1, ::-1]
            K = la.solve_triangular(R, np.eye(G.shape[0]), lower=False)
        
        else:
            raise ValueError("Direction must be 'L' or 'R'.")
            
    except la.LinAlgError:
        raise ValueError("Cholesky decomposition failed. Matrix G must be positive definite.")
    except Exception as e:
        raise ValueError(f"Error in light_cone_kernel: {str(e)}")
        
    return K, R

def compute_all_kernels(A, N, NT):
    """
    Computes K^L and K^R for all sub-intervals [u, v] while enforcing the inclusion property.
    If an interval [u, v] is numerically unstable, all containing intervals [u', v']
    (where u' <= u and v' >= v) are also marked invalid.
    
    Returns:
    - KL_all: np.ndarray of shape (NT, NT, N, N*NT)
    - KR_all: np.ndarray of shape (NT, NT, N, N*NT)
    - is_legit: np.ndarray of shape (NT, NT) boolean mask.
    - tf_list: np.ndarray of shape (NT,). tf_list[u] is the end index (exclusive) for KL at start u.
    - tini_list: np.ndarray of shape (NT,). tini_list[v] is the start index (inclusive) for KR at end v.
    """
    if A.shape[0] != N * NT:
        raise ValueError(f"Matrix size {A.shape[0]} does not match N*NT = {N}*{NT}")
    
    dtype = np.complex128 if np.iscomplexobj(A) else np.float64
    KL_all = np.zeros((NT, NT, N, N * NT), dtype=dtype)
    KR_all = np.zeros((NT, NT, N, N * NT), dtype=dtype)
    is_legit = np.zeros((NT, NT), dtype=bool)
    
    # 1. Identify legitimate intervals and compute K^R (endpoint fixed v, varying u)
    # This automatically enforces: if [u, v] fails, [u', v'] fails for u' <= u, v' >= v.
    tini_list = np.zeros(NT, dtype=int)
    tini = 0
    for v in range(NT):
        success = False
        K_sub_R = None
        while not success and tini <= v:
            try:
                A_sub = A[tini*N:(v+1)*N, tini*N:(v+1)*N]
                # Check BOTH directions for stability
                _, _ = light_cone_kernel(A_sub, 'L')
                K_sub_R, _ = light_cone_kernel(A_sub, 'R')
                success = True
            except ValueError:
                tini += 1
        
        tini_list[v] = tini
        if success and K_sub_R is not None:
            # Mark legitimate region: [u, v] is legit for u in [tini, v]
            for u in range(tini, v + 1):
                is_legit[u, v] = True
                # Store KR_all: Row group (u-tini) corresponds to K^R(u, v | t)
                KR_all[u, v, :, tini*N:(v+1)*N] = K_sub_R[(u-tini)*N:(u-tini+1)*N, :]

    # 2. Compute K^L (startpoint fixed u, varying v) and enforce consistency
    for u in range(NT):
        # Identify the largest candidate v from Phase 1 is_legit
        candidate_tf = u
        for v in range(u, NT):
            if is_legit[u, v]:
                candidate_tf = v + 1
        
        success_L = False
        tf = candidate_tf
        while not success_L and tf > u:
            try:
                A_sub = A[u*N:tf*N, u*N:tf*N]
                K_sub, _ = light_cone_kernel(A_sub, 'L')
                success_L = True
            except ValueError:
                # Failure! The interval [u, tf-1] is not numerically stable for KL.
                # Invalidate it and propagate via inclusion property:
                # Any interval [u', v'] that contains [u, tf-1] is also invalid.
                bad_v = tf - 1
                is_legit[:u+1, bad_v:] = False
                tf -= 1 # Try a smaller interval
        
        if success_L:
            for v in range(u, tf):
                KL_all[u, v, :, u*N:tf*N] = K_sub[(v-u)*N:(v-u+1)*N, :]
    
    # 3. Finalize tini_list and tf_list based on the potentially updated is_legit
    tini_list = np.zeros(NT, dtype=int)
    tf_list = np.zeros(NT, dtype=int)
    for v in range(NT):
        u_min = v + 1
        for u in range(v + 1):
            if is_legit[u, v]:
                u_min = u
                break
        tini_list[v] = u_min
        
    for u in range(NT):
        v_max = u - 1
        for v in range(u, NT):
            if is_legit[u, v]:
                v_max = v
        tf_list[u] = v_max + 1
        
    return KL_all, KR_all, is_legit, tf_list, tini_list

def enforce_gate_inclusion(gates, is_legit):
    """
    Keep a bulk gate only if every contained bulk gate is also legitimate.

    The condition is: U(u, v) is legitimate only when all proper gates
    U(u', v') with u <= u' < v' <= v are legitimate. The strict u' < v'
    excludes diagonal intervals, which have kernels but no bulk gate.
    """
    NT = is_legit.shape[0]
    is_legit_raw = np.asarray(is_legit, dtype=bool)
    is_legit_included = is_legit_raw.copy()

    valid_gate_mask = np.triu(is_legit_raw, k=1).astype(int)
    valid_prefix = valid_gate_mask.cumsum(axis=0).cumsum(axis=1)

    def rect_sum(row0, row1, col0, col1):
        if row0 >= row1 or col0 >= col1:
            return 0
        total = valid_prefix[row1 - 1, col1 - 1]
        if row0 > 0:
            total -= valid_prefix[row0 - 1, col1 - 1]
        if col0 > 0:
            total -= valid_prefix[row1 - 1, col0 - 1]
        if row0 > 0 and col0 > 0:
            total += valid_prefix[row0 - 1, col0 - 1]
        return int(total)

    for v in range(1, NT):
        for u in range(v):
            if not is_legit_raw[u, v]:
                continue
            n_contained_points = v - u + 1
            n_contained_gates = n_contained_points * (n_contained_points - 1) // 2
            n_valid_contained_gates = rect_sum(u, v + 1, u, v + 1)
            if n_valid_contained_gates != n_contained_gates:
                is_legit_included[u, v] = False

    included_gates = {
        key: gate for key, gate in gates.items()
        if is_legit_included[key]
    }
    return included_gates, is_legit_included

def compute_bulk_gates(A, N, NT):
    """
    Computes the bulk unitary gates U(u, v) for the reconstruction circuit.
    Following the formula in bulk_reconstruction_algorithm.md:
    U(u,v) = [K^L(u+1, v); K^R(u, v)] * A * [K^R(u, v-1)^H, K^L(u, v)^H]
    
    Returns:
    - gates: Dictionary mapping (u, v) to (2N, 2N) unitary matrices.
    - KL_all, KR_all, is_legit_refined, tf_list, tini_list
    """
    KL_all, KR_all, is_legit_kernels, tf_list, tini_list = compute_all_kernels(A, N, NT)
    gates = {}
    is_legit_refined = np.zeros((NT, NT), dtype=bool)
    
    # Iterate over causal diamonds in the bulk
    for v in range(1, NT):
        for u in range(v):
            # Check if all FOUR required kernels are within the legitimate region
            if (is_legit_kernels[u, v] and is_legit_kernels[u, v-1] and 
                is_legit_kernels[u+1, v] and (u+1 <= v)): # u+1, v-1 must be inside
                
                # Double check KL(u+1, v) logic: is_legit[u+1, v] handles it
                # Indices for slicing: kernels are non-zero between u and v
                idx_start = u * N
                idx_end = (v + 1) * N
                
                # Output kernels: L from (u+1, v), R from (u, v)
                KL_out = KL_all[u+1, v, :, idx_start:idx_end]
                KR_out = KR_all[u, v, :, idx_start:idx_end]
                O_sub = np.vstack([KL_out, KR_out]) # (2N, active_NT*N)
                
                # Input kernels: R from (u, v-1), L from (u, v)
                KR_in = KR_all[u, v-1, :, idx_start:idx_end]
                KL_in = KL_all[u, v, :, idx_start:idx_end]
                # H conjugate for the input part of the overlap
                I_sub_H = np.vstack([KR_in, KL_in]).conj().T # (active_NT*N, 2N)
                
                # Compute overlap using the boundary anti-commutator sub-block
                A_sub = A[idx_start:idx_end, idx_start:idx_end]
                U = O_sub @ A_sub @ I_sub_H
                
                # Check for unitarity breakdown
                unitarity_err = np.linalg.norm(U @ U.conj().T - np.eye(2*N))
                if unitarity_err < 1e-2:
                    gates[(u, v)] = U
                    is_legit_refined[u, v] = True
                else:
                    # Skip gates where reconstruction fails numerically
                    pass
            
    gates, is_legit_refined = enforce_gate_inclusion(gates, is_legit_refined)

    # Recompute tf_list and tini_list based on the refined is_legit mask
    tf_list_refined = np.zeros(NT, dtype=int)
    tini_list_refined = np.zeros(NT, dtype=int)
    
    for u in range(NT):
        tf = u
        for v in range(u, NT):
            if is_legit_refined[u, v]:
                tf = v + 1
        tf_list_refined[u] = tf
        
    for v in range(NT):
        tini = v
        for u in range(v, -1, -1):
            if is_legit_refined[u, v]:
                tini = u
        tini_list_refined[v] = tini
            
    return gates, KL_all, KR_all, is_legit_refined, tf_list_refined, tini_list_refined

def compute_bulk_entropy(C, KL_all, is_legit, N, NT):
    """
    Computes the bulk entanglement entropy for each legitimate interval [u, v].
    
    Args:
        C: Boundary commutator matrix (N*NT, N*NT).
           For Majorana fermions, C_{ab} = i <[chi_a, chi_b]>.
        KL_all: Past kernels (NT, NT, N, N*NT).
        is_legit: Boolean mask (NT, NT).
        N: Number of flavors per site.
        NT: Number of time points.
        
    Returns:
        entropy_mat: (NT, NT) matrix of entropies (NaN where not legit).
    """
    entropy_mat = np.full((NT, NT), np.nan)
    
    for v in range(NT):
        for u in range(v + 1):
            if is_legit[u, v]:
                # Extract commutator sub-block for the interval [u, v]
                idx_start = u * N
                idx_end = (v + 1) * N
                C_sub = C[idx_start:idx_end, idx_start:idx_end]
                
                # Extract full stacked kernels for all v_prime in [u, v]
                # KL_all[u, u:v+1, :, idx_start:idx_end] has shape (v-u+1, N, (v-u+1)*N)
                K_stacked = KL_all[u, u:v+1, :, idx_start:idx_end]
                # Reshape to ((v-u+1)*N, (v-u+1)*N) with Order: Time, Flavor
                K_full = K_stacked.reshape((v - u + 1) * N, (v - u + 1) * N)
                
                # Bulk commutator: C_bulk = K_full * C_sub * K_full^H
                C_bulk = K_full @ C_sub @ K_full.conj().T
                
                # Compute entropy for Gaussian Majorana state:
                # S = -1/2 * tr[ ((I+C)/2)*log((I+C)/2) + ((I-C)/2)*log((I-C)/2) ]
                # This corresponds to summing the binary entropy of eigenvalues of iC.
                # C_bulk is imaginary antisymmetric (if C is). Eigenvalues of 1j*C_bulk are real.
                try:
                    eigs = np.linalg.eigvalsh(1j * C_bulk)
                    # Clip to [-1, 1] for numerical stability
                    x = np.clip(np.real(eigs), -1.0, 1.0)
                    
                    # Binary entropy for each eigenvalue
                    p = (1.0 + x) / 2.0
                    with np.errstate(divide='ignore', invalid='ignore'):
                        h = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
                        h[np.isnan(h)] = 0.0
                        
                    # S = 1/2 * sum over all N eigenvalues
                    entropy_mat[u, v] = 0.5 * np.sum(h)
                except np.linalg.LinAlgError:
                    entropy_mat[u, v] = np.nan
                
    return entropy_mat

def compute_bulk_commutators(C, KL_all, KR_all, is_legit, u, v, N):
    """
    Computes the bulk commutators for the past (L) and future (R) light cones
    for a given valid region [u, v].
    
    Args:
        C: Boundary commutator matrix (N*NT, N*NT).
        KL_all: Past kernels (NT, NT, N, N*NT).
        KR_all: Future kernels (NT, NT, N, N*NT).
        is_legit: Boolean mask (NT, NT).
        u, v: Region indices.
        N: Number of flavors per site.
        
    Returns:
        CL: Bulk commutator matrix for left-movers ( (v-u+1)*N, (v-u+1)*N ).
        CR: Bulk commutator matrix for right-movers ( (v-u+1)*N, (v-u+1)*N ).
    """
    if not is_legit[u, v]:
        return None, None
        
    idx_start = u * N
    idx_end = (v + 1) * N
    C_sub = C[idx_start:idx_end, idx_start:idx_end]
    
    # 1. Past Light Cone (Left-movers): Fixed u, varying t in [u, v]
    # KL_stacked has shape (v-u+1, N, (v-u+1)*N)
    KL_stacked = KL_all[u, u:v+1, :, idx_start:idx_end]
    # Reshape to ((v-u+1)*N, (v-u+1)*N) with Order: Time, Flavor
    KL_full = KL_stacked.reshape((v - u + 1) * N, (v - u + 1) * N)
    CL = KL_full @ C_sub @ KL_full.conj().T
    
    # 2. Future Light Cone (Right-movers): Fixed v, varying t in [u, v]
    # KR_stacked has shape (v-u+1, N, (v-u+1)*N)
    KR_stacked = KR_all[u:v+1, v, :, idx_start:idx_end]
    # Reshape to ((v-u+1)*N, (v-u+1)*N) with Order: Time, Flavor
    KR_full = KR_stacked.reshape((v - u + 1) * N, (v - u + 1) * N)
    CR = KR_full @ C_sub @ KR_full.conj().T
    
    return CL, CR

def compute_past_future_mi(A, tini_list, tf_list, tev_idx, N):
    """
    Computes the past-future mutual information I_pf for a given evaporation time tev.
    
    Args:
        A: Boundary anti-commutator matrix (N*NT, N*NT).
        tini_list: List where tini_list[v] is the start of the maximal stable interval [u, v].
        tf_list: List where tf_list[u] is the end (exclusive) of the maximal stable interval [u, v].
        tev_idx: Index of the current time t_ev.
        N: Number of flavors per site.
        
    Returns:
        mi: Mutual information value.
    """
    NT = A.shape[0] // N
    
    # 1. Define intervals
    # Ip = [t_in(tev), tev)
    # If = [tev, t_f(tev))
    t_in = tini_list[tev_idx]
    t_f = tf_list[tev_idx] # This is exclusive index (v_max + 1)
    
    if t_in >= tev_idx or t_f <= tev_idx:
        return 0.0
        
    idx_p = np.arange(t_in * N, tev_idx * N)
    idx_f = np.arange(tev_idx * N, t_f * N)
    
    # 2. Extract sub-matrices
    A_p = A[np.ix_(idx_p, idx_p)]
    A_f = A[np.ix_(idx_f, idx_f)]
    A_pf = A[np.ix_(idx_p, idx_f)]
    
    # 3. Compute U = A_p^{-1/2} A_pf A_f^{-1/2}
    def inv_sqrt(M):
        from scipy import linalg
        # Ensure symmetry
        M = (M + M.conj().T) / 2
        # Adding a small ridge for stability if near singular
        evals, evecs = linalg.eigh(M)
        evals = np.clip(evals, 1e-12, None)
        return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.conj().T

    try:
        Ap_inv_sqrt = inv_sqrt(A_p)
        Af_inv_sqrt = inv_sqrt(A_f)
        U = Ap_inv_sqrt @ A_pf @ Af_inv_sqrt
        
        # 4. Compute singular values
        sigma = np.linalg.svd(U, compute_uv=False)
        # Clip sigma to [0, 1] for Choi state validity
        sigma = np.clip(sigma, 0.0, 1.0 - 1e-14)
        
        # 5. Compute MI = sum over sigma_k of h_mi(sigma_k)
        # h_mi(sigma) = (1+sigma)/2 log(1+sigma) + (1-sigma)/2 log(1-sigma)
        p1 = (1.0 + sigma) / 2.0
        p2 = (1.0 - sigma) / 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            mi_terms = p1 * np.log(1.0 + sigma) + p2 * np.log(1.0 - sigma)
            mi_terms[np.isnan(mi_terms)] = 0.0
        
        return np.sum(mi_terms)
    except Exception:
        return 0.0

def build_diamond_circuit_unitary(gates, N, u0, u1, v0, v1, include_boundary_reflection=True):
    """
    Build the finite circuit map for gates in [u0, u1) x [v0, v1).

    The convention matches simulate_wavepacket: each gate maps incoming
    [R(u, v-1), L(u-1, v)] to outgoing [L(u, v), R(u, v)].

    Input order:
        [L(u0-1, v0:v1), R(u0:u1, v0-1)]
    Output order:
        [R(u0:u1, v1-1), L(u1-1, v0:v1)]
    """
    u0 = int(u0)
    u1 = int(u1)
    v0 = int(v0)
    v1 = int(v1)
    du = u1 - u0
    dv = v1 - v0
    if du <= 0 or dv <= 0:
        raise ValueError("Need u1 > u0 and v1 > v0")

    dim = N * (du + dv)
    zero = np.zeros((N, dim), dtype=complex)
    R_maps = {}
    L_maps = {}

    # Input columns: first the L edge ordered by increasing v, then the R edge
    # ordered by increasing u.
    for j, v in enumerate(range(v0, v1)):
        mat = np.zeros((N, dim), dtype=complex)
        mat[:, j * N:(j + 1) * N] = np.eye(N)
        L_maps[(u0 - 1, v)] = mat

    r_offset = dv * N
    for i, u in enumerate(range(u0, u1)):
        mat = np.zeros((N, dim), dtype=complex)
        mat[:, r_offset + i * N:r_offset + (i + 1) * N] = np.eye(N)
        R_maps[(u, v0 - 1)] = mat

    # Apply gates in quantum-circuit order, i.e. increasing physical time u+v.
    for t_sum in range(u0 + v0, (u1 - 1) + (v1 - 1) + 1):
        for u in range(u0, u1):
            v = t_sum - u
            if v < v0 or v >= v1:
                continue

            r_in = R_maps.get((u, v - 1), zero)
            l_in = L_maps.get((u - 1, v), zero)

            if (u, v) in gates:
                U_gate = gates[(u, v)]
                in_map = np.vstack([r_in, l_in])
                out_map = U_gate @ in_map
                L_maps[(u, v)] = out_map[:N, :]
                R_maps[(u, v)] = out_map[N:, :]
            elif include_boundary_reflection and u == v:
                R_maps[(u, v)] = l_in
                L_maps[(u, v)] = zero.copy()
            else:
                raise KeyError(f"Missing gate ({u}, {v}) inside the requested region")

    out_rows = []
    for u in range(u0, u1):
        out_rows.append(R_maps.get((u, v1 - 1), zero))
    for v in range(v0, v1):
        out_rows.append(L_maps.get((u1 - 1, v), zero))

    return np.vstack(out_rows)

def isometry_to_diamond_gates(V, N, tol=1e-10):
    """
    Decompose a single-particle isometry into Gaussian diamond-circuit gates.

    Args:
        V: Single-particle isometry of shape (ell*N, k*N), satisfying
           V.conj().T @ V = I_{k*N}.
        N: Number of flavors per lattice site.
        tol: Numerical tolerance for isometry and residual-rank checks.

    Returns:
        gates: Dictionary mapping (u, v) to (2N, 2N) unitary matrices.
            The gate coordinates are:
                u = -j,                j = 0, ..., k-1
                v = row_index - j,     v = 0, ..., ell-j-2
            Thus the j-th peeling layer has gates
                (-j, 0), ..., (-j, ell-j-2).

    Notes:
        The returned gates are forward circuit gates.  Their adjoints, applied
        layer-by-layer from j=0 upward and from large v to small v within each
        layer, reduce V to the trivial isometry [I; 0].  Equivalently, the
        forward circuit maps [I; 0] to V.

        This implementation assumes the intermediate residual Gram matrices are
        nonsingular.  That is the generic case for the numerical isometries used
        in the evaporation-circuit analysis.  Singular residuals require the
        separate polar-decomposition kernel convention.
    """
    V = np.asarray(V)
    if V.ndim != 2:
        raise ValueError("V must be a matrix")
    if V.shape[0] % N != 0 or V.shape[1] % N != 0:
        raise ValueError("V dimensions must be multiples of N")

    ell = V.shape[0] // N
    k = V.shape[1] // N
    if ell < k:
        raise ValueError(f"Need ell >= k, got ell={ell}, k={k}")

    gram_err = np.linalg.norm(V.conj().T @ V - np.eye(k * N))
    if gram_err > 100 * tol:
        raise ValueError(f"Input is not an isometry within tolerance: error={gram_err:g}")

    work = V.astype(np.complex128 if np.iscomplexobj(V) else np.float64, copy=True)
    gates = {}

    for j in range(k):
        col_slice = slice(j * N, (j + 1) * N)

        for row in range(ell - 2, j - 1, -1):
            row_slice = slice(row * N, (row + 2) * N)
            X = work[row_slice, col_slice]

            # Build Q such that Q @ X = [(X^H X)^{1/2}; 0].
            gram = (X.conj().T @ X + (X.conj().T @ X).conj().T) / 2
            evals, evecs = la.eigh(gram)
            if np.min(evals) < tol:
                raise ValueError(
                    "Encountered a singular residual block while constructing "
                    f"layer j={j}, row={row}; min eigenvalue={np.min(evals):g}"
                )

            sqrt_gram = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
            inv_sqrt_gram = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.conj().T
            Y = X @ inv_sqrt_gram

            # Complete the N orthonormal columns Y to a 2N x 2N unitary.
            Y_complement = la.null_space(Y.conj().T)
            if Y_complement.shape[1] != N:
                raise ValueError(
                    "Failed to complete the local isometry to a two-leg unitary "
                    f"at layer j={j}, row={row}"
                )
            Q_basis = np.hstack([Y, Y_complement])
            Q = Q_basis.conj().T

            # Clean the intended top block to the positive square root.  This
            # avoids accumulating tiny phase/roundoff errors from null_space.
            transformed = Q @ work[row_slice, :]
            transformed[:N, col_slice] = sqrt_gram
            transformed[N:, col_slice] = 0.0
            work[row_slice, :] = transformed

            # Store the forward gate.  Its adjoint is the peeling operation Q.
            gates[(-j, row - j)] = Q.conj().T

    target = np.zeros_like(work)
    target[:k * N, :] = np.eye(k * N)
    final_err = np.linalg.norm(work - target)
    if final_err > 1000 * tol:
        warnings.warn(
            "Constructed gates, but the adjoint circuit does not reduce the "
            f"isometry to [I; 0] within tolerance: error={final_err:g}",
            RuntimeWarning,
        )

    return gates

def simulate_wavepacket(gates, N, NT, start_u, start_v, initial_state, is_left):
    """
    Simulates the propagation of a wavepacket in the bulk circuit.
    Follows physical time evolution: t = (u+v)/2 increases.
    
    Args:
        gates: Dict of (u, v) -> (2N, 2N) matrices.
        N: Flavors per site.
        NT: Total time points.
        start_u, start_v: Starting interval indices.
        initial_state: (N,) vector.
        is_left: True if the initial packet is a left-mover.
        
    Returns:
        R_links, L_links: Full complex vectors for each link (NT, NT, N).
    """
    R_links = np.full((NT, NT, N), np.nan, dtype=complex)
    L_links = np.full((NT, NT, N), np.nan, dtype=complex)
    
    # Set initial state (outputs of gate start_u, start_v)
    if is_left:
        L_links[start_u, start_v] = initial_state
    else:
        R_links[start_u, start_v] = initial_state
        
    # Evolve forward in physical time: t_sum = u + v increases
    for t_sum in range(start_u + start_v + 1, 2 * NT - 1):
        for u in range(NT):
            v = t_sum - u
            if v < 0 or v >= NT or v < u:
                continue
            if (u, v) == (start_u, start_v):
                continue
                
            # Incoming links to gate (u, v):
            # R comes from (u, v-1) at t - 0.5
            # L comes from (u-1, v) at t - 0.5
            r_in = R_links[u, v-1] if (v > 0 and v-1 >= u) else np.full(N, np.nan, dtype=complex)
            l_in = L_links[u-1, v] if (u > 0 and v >= u-1) else np.full(N, np.nan, dtype=complex)
            
            if not (np.any(np.isnan(r_in)) and np.any(np.isnan(l_in))):
                # At least one input is known. Treat others as zero.
                r_calc = np.where(np.isnan(r_in), 0.0, r_in)
                l_calc = np.where(np.isnan(l_in), 0.0, l_in)
                
                if (u, v) in gates:
                    U = gates[(u, v)]
                    vec_in = np.concatenate([r_calc, l_calc])
                    vec_out = U @ vec_in
                    L_links[u, v] = vec_out[:N]
                    R_links[u, v] = vec_out[N:]
                elif u == v:
                    # Boundary reflection
                    R_links[u, v] = l_calc
                    L_links[u, v] = np.full(N, np.nan, dtype=complex)
                else:
                    pass
            else:
                # Both inputs unknown
                pass
                
    return R_links, L_links
