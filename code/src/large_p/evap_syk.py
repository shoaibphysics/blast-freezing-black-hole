# This library contains functions to compute the two-point function of 
# the SYK+bath model, with an SYK island of N fermions $\eta_i$ coupled with a much 
# bigger island of M fermions $\psi_j$.
# Code in this library studies the probe limit $p\rightarrow \infty$, $M\gg Np $. 
# For derivations see paper/supplementary.tex, Sec. S4.

import numpy as np

def zArcTan(x):
    """Exponential representation of 2 ArcTan[x]: exp(2i ArcTan[x])"""
    x = np.asarray(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = (1 + 1j * x) / (1 - 1j * x)

    inf_mask = np.isinf(x)
    if result.shape == ():
        return complex(-1.0 if inf_mask else result)

    result = np.asarray(result, dtype=np.complex128)
    result[inf_mask] = -1.0
    return result

def compute_phiG_VG(mu):
    phiG = -0.5 * np.log(mu * (np.sqrt((mu / 2)**2 + 1) - mu / 2))
    VG = np.sqrt(mu * (mu / 2 + np.sqrt(1 + (mu / 2)**2)))
    return phiG, VG

def RI(t1, t2, a, eps):
    denom = 1 + (np.sin(eps)**(-2)) * (np.sinh(a * (t1 - t2) / 2)**2)
    return (1.0 / denom) * np.exp(-1j * np.pi) * zArcTan(np.tan(eps) / np.tanh(a * (t1 - t2) / 2))

def RII(t1, t2, a, tev, mu, nu, eps, phiG, VG):
    term1 = 1 / ((1 + (1/np.sin(eps)**2) * np.sinh(a * (tev - t2) / 2)**2) * (1 + (1/VG**2) * np.sin(VG * (t1 - tev) / 2)**2))
    term2 = np.exp(-1j * np.pi) * zArcTan(np.tan(eps) / np.tanh(a * (tev - t2) / 2)) / zArcTan(np.exp(-phiG) * np.tan(VG * (t1 - tev) / 2))
    term3 = np.exp(1j * mu * (1 - nu) * (t1 - tev))
    return term1 * term2 * term3

def RIII(t1, t2, tev, mu, nu, phiG, VG):
    term1 = 1 / (1 + (1/VG**2) * np.sin(VG * (t1 - t2) / 2)**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cot_t12 = np.exp(-phiG) / np.tan(VG * (t1 - t2) / 2)
        cot_t1ev = np.exp(-phiG) / np.tan(VG * (t1 - tev) / 2)
        cot_tevt2 = np.exp(-phiG) / np.tan(VG * (tev - t2) / 2)
    term2 = np.exp(1j * np.pi) * zArcTan(cot_t12)
    term3 = 1 / zArcTan(cot_t1ev)
    term4 = 1 / zArcTan(cot_tevt2)
    term5 = 1 / zArcTan(np.exp(-phiG) * np.tan(VG * (t1 - tev) / 2))
    term6 = 1 / zArcTan(np.exp(-phiG) * np.tan(VG * (tev - t2) / 2))
    term7 = np.exp(1j * mu * (1 - nu) * (t1 - t2))
    return term1 * term2 * term3 * term4 * term5 * term6 * term7

def LI(t1, t2, a, eps):
    return (np.sin(eps) / np.cosh(a * (t1 + t2) / 2))**2

def LII(t1, t2, a, tev, mu, nu, eps, phiG, VG):
    term1 = (np.sin(eps) / np.cosh(a * (tev + t2) / 2))**2 / (1 + (1/VG**2) * np.sin(VG * (t1 - tev) / 2)**2)
    term2 = 1 / zArcTan(np.exp(-phiG) * np.tan(VG * (t1 - tev) / 2))
    term3 = np.exp(1j * mu * (1 - nu) * (t1 - tev))
    return term1 * term2 * term3

def LIII(t1, t2, a, tev, mu, nu, eps, phiG, VG):
    prefactor = (1 / (1 + VG**(-2))) * (np.sin(eps) / np.cosh(a * tev))**2
    denom = (1 + (1/VG**2) * np.cos(VG * (t1 - t2) / 2)**2)
    numerator_terms = (1 + (1/VG**2) * np.cos(VG * (t1 - tev) / 2)**2) * (1 + (1/VG**2) * np.cos(VG * (tev - t2) / 2)**2)
    denominator_terms = (1 + (1/VG**2) * np.sin(VG * (t1 - tev) / 2)**2) * (1 + (1/VG**2) * np.sin(VG * (tev - t2) / 2)**2)
    term1 = prefactor / denom * (numerator_terms / denominator_terms)
    term2 = 1 / zArcTan(np.exp(-phiG) * np.tan(VG * (t1 - t2) / 2))
    term3 = np.exp(1j * mu * (1 - nu) * (t1 - t2))
    return term1 * term2 * term3

def compute_self_energy(t1grid, t2grid, type_str, params):
    """
    Computes egRR or egRL on a 2D grid.
    type_str: 'RR' or 'RL'
    params: dict with a, tev, mu, nu, eps, etc.
    """
    
    tev = params['tev']
    mu = params['mu']
    nu = params['nu']
    eps = params['eps']
    a = params['a'] * np.sin(eps) # The coefficient of t in two-point function, which is a*sin(eps)
    phiG, VG = compute_phiG_VG(mu)

    T1, T2 = np.meshgrid(t1grid, t2grid)
    
    # Avoid singularities where t1 == t2 or other sensitive points
    eps_safe = 1e-8
    T1 = np.where(np.abs(T1 - T2) < eps_safe, T1 + eps_safe, T1)
    T1 = np.where(np.abs(T1 - tev) < eps_safe, T1 + eps_safe, T1)
    T2 = np.where(np.abs(T2 - tev) < eps_safe, T2 + eps_safe, T2)

    result = np.zeros(T1.shape, dtype=complex)

    # Masks for regions
    mask11 = (T1 < tev) & (T2 < tev)
    mask21 = (T1 >= tev) & (T2 < tev)
    mask12 = (T1 < tev) & (T2 >= tev)
    mask22 = (T1 >= tev) & (T2 >= tev)

    if type_str == 'RR':
        result[mask11] = RI(T1[mask11], T2[mask11], a, eps)
        result[mask21] = RII(T1[mask21], T2[mask21], a, tev, mu, nu, eps, phiG, VG)
        # Symmetry: egRR(t1, t2) = Conjugate[RII(t2, t1)] if t1 < tev and t2 >= tev
        result[mask12] = np.conj(RII(T2[mask12], T1[mask12], a, tev, mu, nu, eps, phiG, VG))
        result[mask22] = RIII(T1[mask22], T2[mask22], tev, mu, nu, phiG, VG)
    elif type_str == 'RL':
        result[mask11] = LI(T1[mask11], T2[mask11], a, eps)
        result[mask21] = LII(T1[mask21], T2[mask21], a, tev, mu, nu, eps, phiG, VG)
        result[mask12] = np.conj(LII(T2[mask12], T1[mask12], a, tev, mu, nu, eps, phiG, VG))
        result[mask22] = LIII(T1[mask22], T2[mask22], a, tev, mu, nu, eps, phiG, VG)

    return result


def _compute_self_energy_pairwise(t1, t2, type_str, params):
    """
    Pairwise version of compute_self_energy without meshgrid expansion.
    """
    tev = params['tev']
    mu = params['mu']
    nu = params['nu']
    eps = params['eps']
    a = params['a'] * np.sin(eps)
    phiG, VG = compute_phiG_VG(mu)

    t1, t2 = np.broadcast_arrays(
        np.asarray(t1, dtype=float),
        np.asarray(t2, dtype=float),
    )

    eps_safe = 1e-8
    T1 = np.where(np.abs(t1 - t2) < eps_safe, t1 + eps_safe, t1)
    T1 = np.where(np.abs(T1 - tev) < eps_safe, T1 + eps_safe, T1)
    T2 = np.where(np.abs(t2 - tev) < eps_safe, t2 + eps_safe, t2)

    result = np.zeros(T1.shape, dtype=complex)

    mask11 = (T1 < tev) & (T2 < tev)
    mask21 = (T1 >= tev) & (T2 < tev)
    mask12 = (T1 < tev) & (T2 >= tev)
    mask22 = (T1 >= tev) & (T2 >= tev)

    if type_str == 'RR':
        result[mask11] = RI(T1[mask11], T2[mask11], a, eps)
        result[mask21] = RII(T1[mask21], T2[mask21], a, tev, mu, nu, eps, phiG, VG)
        result[mask12] = np.conj(RII(T2[mask12], T1[mask12], a, tev, mu, nu, eps, phiG, VG))
        result[mask22] = RIII(T1[mask22], T2[mask22], tev, mu, nu, phiG, VG)
    elif type_str == 'RL':
        result[mask11] = LI(T1[mask11], T2[mask11], a, eps)
        result[mask21] = LII(T1[mask21], T2[mask21], a, tev, mu, nu, eps, phiG, VG)
        result[mask12] = np.conj(LII(T2[mask12], T1[mask12], a, tev, mu, nu, eps, phiG, VG))
        result[mask22] = LIII(T1[mask22], T2[mask22], a, tev, mu, nu, eps, phiG, VG)
    else:
        raise ValueError("type_str must be 'RR' or 'RL'")

    if result.shape == ():
        return complex(result)
    return result

def phase_unwrap_2d(phi):
    """
    Unwrap phase in 2D by rows then columns.
    """
    row_unwrapped = np.unwrap(phi, axis=1)
    col_unwrapped = np.unwrap(row_unwrapped, axis=0)
    return col_unwrapped

def get_unwrapped_field(z_field, p):
    """
    Computes (z_field)^(1/p) using phase unwrapping.
    """
    amp = np.abs(z_field)
    phase = np.angle(z_field)
    unwrapped_phase = phase_unwrap_2d(phase)
    return (amp**(1.0/p)) * np.exp(1j * unwrapped_phase / p)


def _get_unwrapped_factor(z_field, p):
    """
    Compute z_field^(1/p) with phase unwrapping.

    For 2D inputs this matches get_unwrapped_field exactly.
    For 1D inputs we unwrap along the single axis.
    For scalars we fall back to the principal value since there is no neighbor
    information available for branch selection.
    """
    z_field = np.asarray(z_field, dtype=complex)

    if z_field.ndim == 0:
        return complex(np.abs(z_field)**(1.0 / p) * np.exp(1j * np.angle(z_field) / p))

    if z_field.ndim == 1:
        amp = np.abs(z_field)
        phase = np.unwrap(np.angle(z_field))
        return (amp**(1.0 / p)) * np.exp(1j * phase / p)

    return get_unwrapped_field(z_field, p)

def evapSYK2pt(tgrid, params):
    """
    Computes the general 2pt function in the evaporating SYK model.

    Convention: GRR and GRL are both Hermitian
    """
    RzField = compute_self_energy(tgrid, tgrid, 'RR', params)
    GRR = get_unwrapped_field(RzField, params['p'])
    
    LzField = compute_self_energy(tgrid, tgrid, 'RL', params)
    GRL = get_unwrapped_field(LzField, params['p'])
    
    I2 = np.eye(2)
    J2 = np.array([[0, 1], [-1, 0]])
    CorM = np.kron(GRR, I2) + np.kron(GRL, -1j * J2)
    CorM = (CorM + CorM.conj().T) / 2
    
    
    
    return CorM, GRR, GRL, RzField, LzField


def _size_operator_common_params(params):
    mu = params['mu']
    eps = params['eps']
    a = params['a']

    x = mu * (np.sqrt((mu / 2.0)**2 + 1.0) - mu / 2.0)
    VG = np.sqrt(x / (1.0 - x))
    omega = np.sqrt(x * (2.0 - x) / (1.0 - x))
    alpha = a * np.sin(eps)

    return {
        'a': a,
        'eps': eps,
        'x': x,
        'VG': VG,
        'omega': omega,
        'alpha': alpha,
    }


def _chi_psi0(t, tf, cparams):
    t = np.asarray(t, dtype=float)
    imag_part = -1j * np.arctanh(np.sqrt(cparams['x']))
    return np.where(
        t < tf,
        cparams['VG'] * t + imag_part,
        cparams['VG'] * (2.0 * tf - t) + imag_part,
    )


def _chi_psi0_prime(t, tf, cparams):
    t = np.asarray(t, dtype=float)
    return np.where(t < tf, cparams['VG'], -cparams['VG'])


def _chi_delta_psi(t, tf, cparams):
    dt = np.asarray(t, dtype=float) - tf
    return (
        1j * cparams['VG'] * np.sin(cparams['omega'] * dt) / cparams['omega']
        + cparams['VG']
        * (1.0 - np.cos(cparams['omega'] * dt))
        / (cparams['omega']**2 * np.sqrt(1.0 - cparams['x']))
    )


def _chi_delta_psi_prime(t, tf, cparams):
    dt = np.asarray(t, dtype=float) - tf
    return (
        1j * cparams['VG'] * np.cos(cparams['omega'] * dt)
        + cparams['VG']
        * np.sin(cparams['omega'] * dt)
        / (cparams['omega'] * np.sqrt(1.0 - cparams['x']))
    )


def _chi_dgR_dlambda(t1, t2, tf, cparams):
    psi01 = _chi_psi0(t1, tf, cparams)
    psi02 = _chi_psi0(t2, tf, cparams)
    return (
        _chi_delta_psi_prime(t1, tf, cparams) / _chi_psi0_prime(t1, tf, cparams)
        - _chi_delta_psi(t1, tf, cparams) * np.cos((psi01 - np.conj(psi02)) / 2.0)
        / np.sin((psi01 - np.conj(psi02)) / 2.0)
    )


def _chi_dgL_dlambda(t1, t2, tf, cparams):
    psi01 = _chi_psi0(t1, tf, cparams)
    psi02 = _chi_psi0(t2, tf, cparams)
    return (
        _chi_delta_psi_prime(t1, tf, cparams) / _chi_psi0_prime(t1, tf, cparams)
        + _chi_delta_psi(t1, tf, cparams) * np.sin((psi01 - np.conj(psi02)) / 2.0)
        / np.cos((psi01 - np.conj(psi02)) / 2.0)
    )


def _eta_pre_psi0(t, cparams):
    u = (cparams['alpha'] * np.asarray(t, dtype=float) - 1j * cparams['eps']) / 2.0
    return (2.0 / cparams['a']) * np.arctan(np.tanh(u))


def _eta_pre_psi0_prime(t, cparams):
    u = cparams['alpha'] * np.asarray(t, dtype=float) - 1j * cparams['eps']
    return np.sin(cparams['eps']) / np.cosh(u)


def _eta_pre_psi0_double_prime(t, cparams):
    u = cparams['alpha'] * np.asarray(t, dtype=float) - 1j * cparams['eps']
    return -cparams['alpha'] * np.sin(cparams['eps']) * np.tanh(u) / np.cosh(u)


def _eta_pre_delta_p_ev(tf, tev, cparams):
    return (
        (1.0 - cparams['x'])
        * (1.0 - np.cos(cparams['omega'] * (np.asarray(tf, dtype=float) - tev)))
        / (2.0 - cparams['x'])
    )


def analytic_DeltaNchi_tb_tf(tb, tf, params, invalid_value=np.nan):
    """
    Analytic formula from paper/supplementary.tex, Sec. S4 (chi size response).

    Valid for tb < tev < tf.
    """

    tev = params['tev']
    cparams = _size_operator_common_params(params)

    tb, tf = np.broadcast_arrays(
        np.asarray(tb, dtype=float),
        np.asarray(tf, dtype=float),
    )

    result = np.full(tb.shape, invalid_value, dtype=float)
    valid = (tb < tev) & (tf > tev)
    if np.any(valid):
        tbv = tb[valid]
        tfv = tf[valid]
        eps = cparams['eps']
        s_eps = np.sin(eps)
        delta_nev = -_eta_pre_delta_p_ev(tfv, tev, cparams)
        prefactor = (
            1.0
            - (1.0 / s_eps**2) * (1.0 - np.cosh(tbv * s_eps) / np.cosh(tev * s_eps))
            + (tev - tbv) * (np.cos(eps)**2 / s_eps) * np.tanh(tev * s_eps)
        )
        result[valid] = delta_nev * prefactor

    if result.shape == ():
        return float(result)
    return result


def _eta_pre_delta_coeffs(tf, tev, cparams):
    delta_p_ev = _eta_pre_delta_p_ev(tf, tev, cparams)
    eps = cparams['eps']
    s_eps = np.sin(eps)
    delta_eps = delta_p_ev * np.tanh(tev * s_eps)
    delta_t0 = delta_p_ev * (np.cos(eps) / s_eps) * (
        tev * np.tanh(tev * s_eps) - 1.0 / s_eps
    )
    delta_c = -delta_p_ev / (s_eps * np.cosh(tev * s_eps))
    return delta_c, delta_eps, delta_t0


def _eta_pre_delta_psi(t, tf, tev, cparams):
    delta_c, delta_eps, delta_t0 = _eta_pre_delta_coeffs(tf, tev, cparams)
    t = np.asarray(t, dtype=float)
    shift = -delta_t0 + delta_eps * (t * np.cos(cparams['eps']) - 1j) / np.sin(cparams['eps'])
    return delta_c + _eta_pre_psi0_prime(t, cparams) * shift


def _eta_pre_delta_psi_prime(t, tf, tev, cparams):
    _, delta_eps, delta_t0 = _eta_pre_delta_coeffs(tf, tev, cparams)
    t = np.asarray(t, dtype=float)
    shift = -delta_t0 + delta_eps * (t * np.cos(cparams['eps']) - 1j) / np.sin(cparams['eps'])
    return (
        _eta_pre_psi0_double_prime(t, cparams) * shift
        + _eta_pre_psi0_prime(t, cparams) * delta_eps * np.cos(cparams['eps']) / np.sin(cparams['eps'])
    )


def compute_size_operator_Dchi(tilde_t1, t2, tf, params, invalid_value=np.nan + 0j):
    """
    Compute the size-operator kernel mathfrak{D}_chi(tilde_t1, t2; tf) from the
    size-operator derivation in paper/supplementary.tex, Sec. S4.

    This implementation follows the note in the domain
    tf > tev and tilde_t1, t2 < tf, where tilde_t1 = 2 tf - t1 is the
    physical time variable used in the note. Inputs may be scalars or
    broadcastable numpy arrays. Entries outside that domain are filled with
    invalid_value.
    """

    tev = params['tev']
    cparams = _size_operator_common_params(params)

    tilde_t1, t2, tf = np.broadcast_arrays(
        np.asarray(tilde_t1, dtype=float),
        np.asarray(t2, dtype=float),
        np.asarray(tf, dtype=float),
    )
    t1 = 2.0 * tf - tilde_t1
    tilde_tev = 2.0 * tf - tev

    result = np.full(tilde_t1.shape, invalid_value, dtype=complex)
    valid = (tf > tev) & (tilde_t1 < tf) & (t2 < tf)

    mask1 = valid & (t2 <= tev) & (tev <= tilde_t1) & (tilde_t1 < tf)
    if np.any(mask1):
        dgR = _chi_dgR_dlambda(t1[mask1], tev, tf[mask1], cparams)
        dgL = _chi_dgL_dlambda(t1[mask1], tev, tf[mask1], cparams)
        result[mask1] = -1.0 + 1j * (np.real(dgR) + 1j * np.imag(dgL))

    mask2 = valid & (tilde_t1 < tev) & (t2 <= tev)
    if np.any(mask2):
        psi01 = _eta_pre_psi0(tilde_t1[mask2], cparams)
        psi02 = _eta_pre_psi0(t2[mask2], cparams)
        delta_psi = _eta_pre_delta_psi(tilde_t1[mask2], tf[mask2], tev, cparams)
        delta_psi_prime = _eta_pre_delta_psi_prime(tilde_t1[mask2], tf[mask2], tev, cparams)
        psi0_prime = _eta_pre_psi0_prime(tilde_t1[mask2], cparams)
        result[mask2] = 1j * (
            delta_psi_prime / psi0_prime
            - delta_psi * np.cos((psi01 - np.conj(psi02)) / 2.0) / np.sin((psi01 - np.conj(psi02)) / 2.0)
        )

    mask3 = valid & (tev <= tilde_t1) & (tilde_t1 < tf) & (tev < t2) & (t2 < tf)
    if np.any(mask3):
        dg_main = _chi_dgR_dlambda(t1[mask3], t2[mask3], tf[mask3], cparams)
        dgR_ev = _chi_dgR_dlambda(t1[mask3], tev, tf[mask3], cparams)
        dgL_ev = _chi_dgL_dlambda(t1[mask3], tev, tf[mask3], cparams)
        result[mask3] = -1.0 + 1j * dg_main + np.imag(dgR_ev - dgL_ev)

    mask4 = valid & (tilde_t1 < tev) & (tev < t2) & (t2 < tf)
    if np.any(mask4):
        # S4, eq:S4-bilocal-region-IV-result: start from the pre-evaporation
        # eta response at t2=tev, not a continuation of the bath-only region.
        boundary_term = -_eta_pre_delta_p_ev(tf[mask4], tev, cparams) * _compute_size_operator_Deta_pre(
            tilde_t1[mask4], np.full_like(tilde_t1[mask4], tev), tev, cparams
        )
        dgR_match_t2 = _chi_dgR_dlambda(tilde_tev[mask4], t2[mask4], tf[mask4], cparams)
        dgR_match_ev = _chi_dgR_dlambda(tilde_tev[mask4], tev, tf[mask4], cparams)
        result[mask4] = boundary_term + 1j * (dgR_match_t2 - dgR_match_ev)

    if result.shape == ():
        return complex(result)
    return result


def compute_size_operator_DeltaNchi(tilde_t1, t2, tf, params, invalid_value=np.nan + 0j):
    """
    Compute Delta N_chi(tilde_t1, t2; tf) using
    Delta N_chi = exp[g_R^eta(tilde_t1, t2; 0)/p] * D_chi.
    """
    Dchi = compute_size_operator_Dchi(tilde_t1, t2, tf, params, invalid_value=invalid_value)
    z_rr = _compute_self_energy_pairwise(tilde_t1, t2, 'RR', params)
    GRR_factor = _get_unwrapped_factor(z_rr, params['p'])
    DeltaNchi = GRR_factor * Dchi
    if np.shape(DeltaNchi) == ():
        return complex(DeltaNchi)
    return DeltaNchi


def compute_size_operator_DeltaNchi_grid(time_grid, tf, params, invalid_value=np.nan + 0j):
    """
    Compute Delta N_chi on a square grid built from a 1D physical-time grid.
    Uses phase-unwrapped GRR to match the large-p two-point function branch.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    TILDE_T1, T2 = np.meshgrid(time_grid, time_grid, indexing='ij')
    Dchi = compute_size_operator_Dchi(TILDE_T1, T2, tf, params, invalid_value=invalid_value)
    z_rr = _compute_self_energy_pairwise(TILDE_T1, T2, 'RR', params)
    GRR_factor = _get_unwrapped_factor(z_rr, params['p'])
    DeltaNchi = GRR_factor * Dchi
    return DeltaNchi





##################################
##### Eta size operator ##########
##################################

def _compute_size_operator_Deta_pre(tilde_t1, t2, tev, cparams):
    """
    Pre-evaporation eta-size kernel.

    This is the eta-size analogue of the pre-evaporation part of
    compute_size_operator_Dchi, but with the eta-size boundary data

        delta p_eta(tilde t_ev) = -1.

    We keep the same a=1 convention used by the existing size-operator code.
    """
    tilde_t1, t2 = np.broadcast_arrays(
        np.asarray(tilde_t1, dtype=float),
        np.asarray(t2, dtype=float),
    )

    eps = cparams['eps']
    s_eps = np.sin(eps)

    delta_p_ev = -np.ones_like(tilde_t1)

    delta_eps = delta_p_ev * np.tanh(tev * s_eps)
    delta_t0 = delta_p_ev * (np.cos(eps) / s_eps) * (
        tev * np.tanh(tev * s_eps) - 1.0 / s_eps
    )
    delta_c = -delta_p_ev / (s_eps * np.cosh(tev * s_eps))

    shift = (
        -delta_t0
        + delta_eps * (tilde_t1 * np.cos(eps) - 1j) / s_eps
    )

    delta_psi = delta_c + _eta_pre_psi0_prime(tilde_t1, cparams) * shift
    delta_psi_prime = (
        _eta_pre_psi0_double_prime(tilde_t1, cparams) * shift
        + _eta_pre_psi0_prime(tilde_t1, cparams)
        * delta_eps
        * np.cos(eps)
        / s_eps
    )

    psi01 = _eta_pre_psi0(tilde_t1, cparams)
    psi02 = _eta_pre_psi0(t2, cparams)
    psi0_prime = _eta_pre_psi0_prime(tilde_t1, cparams)

    return 1j * (
        delta_psi_prime / psi0_prime
        - delta_psi
        * np.cos((psi01 - np.conj(psi02)) / 2.0)
        / np.sin((psi01 - np.conj(psi02)) / 2.0)
    )


def compute_size_operator_Deta(tilde_t1, t2, tf, params, invalid_value=np.nan + 0j):
    """
    Compute the eta-size kernel mathfrak{D}_eta(tilde_t1, t2; tf).

    Here tilde_t1 is the reflected physical time used in the note.  The
    unfolded contour point is 2*tf - tilde_t1.
    """
    tev = params['tev']
    cparams = _size_operator_common_params(params)

    tilde_t1, t2, tf = np.broadcast_arrays(
        np.asarray(tilde_t1, dtype=float),
        np.asarray(t2, dtype=float),
        np.asarray(tf, dtype=float),
    )

    result = np.full(tilde_t1.shape, invalid_value, dtype=complex)
    valid = (tf > tev) & (tilde_t1 < tf) & (t2 < tf)

    # Region I: tilde_t1, t2 < tev.
    mask_pre = valid & (tilde_t1 < tev) & (t2 <= tev)
    if np.any(mask_pre):
        result[mask_pre] = _compute_size_operator_Deta_pre(
            tilde_t1[mask_pre],
            t2[mask_pre],
            tev,
            cparams,
        )

    # Regions II and III: the eta-size kick is direct, so D_eta = 1.
    mask_direct = valid & (tev <= tilde_t1) & (tilde_t1 < tf)
    if np.any(mask_direct):
        result[mask_direct] = 1.0

    # Region IV: tilde_t1 < tev < t2.  The chi bath is unperturbed for an
    # eta-size insertion, so the answer is the boundary value at t2 = tev.
    mask_cross = valid & (tilde_t1 < tev) & (tev < t2) & (t2 < tf)
    if np.any(mask_cross):
        result[mask_cross] = _compute_size_operator_Deta_pre(
            tilde_t1[mask_cross],
            np.full_like(tilde_t1[mask_cross], tev),
            tev,
            cparams,
        )

    if result.shape == ():
        return complex(result)
    return result


def compute_size_operator_DeltaNeta(tilde_t1, t2, tf, params, invalid_value=np.nan + 0j):
    """
    Compute Delta N_eta(tilde_t1, t2; tf).

    Delta N_eta = exp[g_R^eta(tilde_t1,t2;0)/p] * mathfrak{D}_eta.
    """
    Deta = compute_size_operator_Deta(
        tilde_t1, t2, tf, params, invalid_value=invalid_value
    )

    z_rr = _compute_self_energy_pairwise(tilde_t1, t2, 'RR', params)
    GRR_factor = _get_unwrapped_factor(z_rr, params['p'])

    DeltaNeta = GRR_factor * Deta

    if np.shape(DeltaNeta) == ():
        return complex(DeltaNeta)
    return DeltaNeta


def compute_size_operator_DeltaNeta_grid(time_grid, tf, params, invalid_value=np.nan + 0j):
    """
    Compute Delta N_eta on a square grid built from a 1D physical-time grid.
    Uses phase-unwrapped GRR to match the large-p two-point function branch.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    TILDE_T1, T2 = np.meshgrid(time_grid, time_grid, indexing='ij')

    Deta = compute_size_operator_Deta(
        TILDE_T1, T2, tf, params, invalid_value=invalid_value
    )

    z_rr = _compute_self_energy_pairwise(TILDE_T1, T2, 'RR', params)
    GRR_factor = _get_unwrapped_factor(z_rr, params['p'])

    DeltaNeta = GRR_factor * Deta
    return DeltaNeta









# Quick self-test for the eta size-operator implementation.
#
# This checks the physical equal-time boundary insertion t1=t2=tb with
# tb < tev < tf.  In this limit the finite-p prefactor should be one, and
# the note predicts
#
#     Delta N_eta(tb,tb;tf) = D_eta(tb,tb;tf) = E(tb,tev).

if __name__ == "__main__":
    params = {
        "mu": 0.5,
        "nu": 0.0,
        "eps": 0.3,
        "a": 1.0,
        "tev": 1.0,
        "p": 24,
    }

    tb = 0.3
    tf = 2.0

    tev = params["tev"]
    eps = params["eps"]
    s = np.sin(eps)

    E = (
        1.0
        - (1.0 / s**2) * (1.0 - np.cosh(tb * s) / np.cosh(tev * s))
        + (tev - tb) * (np.cos(eps)**2 / s) * np.tanh(tev * s)
    )

    Deta = compute_size_operator_Deta(tb, tb, tf, params)
    DeltaNeta = compute_size_operator_DeltaNeta(tb, tb, tf, params)

    print("E =", E)
    print("Deta =", Deta)
    print("DeltaNeta =", DeltaNeta)
    print("Deta - E =", Deta - E)
    print("DeltaNeta - E =", DeltaNeta - E)
