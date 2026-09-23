import numpy as np
from scipy.optimize import fsolve

def phase_unwrap_2d(phi):
    """Unwrap phase in 2D by rows then columns."""
    row_unwrapped = np.unwrap(phi, axis=1)
    col_unwrapped = np.unwrap(row_unwrapped, axis=0)
    return col_unwrapped

def get_unwrapped_power(z_field, q):
    """Computes (z_field)^(1/q) using phase unwrapping."""
    amp = np.abs(z_field)
    phase = np.angle(z_field)
    unwrapped_phase = phase_unwrap_2d(phase)
    return (amp**(1.0/q)) * np.exp(1j * unwrapped_phase / q)

def coupledSYK2pt(psi, Dpsi, q):
    """
    Computes the general 2pt function in large q coupled SYK model.
    Translated from coupledSYK2pt.m.
    """
    psiX, psiY = np.meshgrid(psi, psi)
    DpsiX, DpsiY = np.meshgrid(Dpsi, Dpsi)
    
    # SigLL = (-DpsiX.*conj(DpsiY)./(sin((psiX-conj(psiY))/2)).^2);
    SigLL = -DpsiX * np.conj(DpsiY) / (np.sin((psiX - np.conj(psiY)) / 2)**2)
    
    # SigLR = (DpsiX.*conj(DpsiY)./(cos((psiX-conj(psiY))/2)).^2);
    SigLR = DpsiX * np.conj(DpsiY) / (np.cos((psiX - np.conj(psiY)) / 2)**2)
    
    CLL = get_unwrapped_power(SigLL, q)
    CLR = 1j * get_unwrapped_power(SigLR, q)
    
    # CorM in MATLAB: interleaved L and R flavors at each time point
    # CorM = kron(CLL, [[1,0],[0,0]]) + kron(CLR, [[0,1],[0,0]]) + 
    #        kron(CLR.H, [[0,0],[1,0]]) + kron(CLL, [[0,0],[0,1]])
    # This is equivalent to putting [CLL_ij, CLR_ij; CLR_ji*, CLL_ij] in each 2x2 block
    
    I11 = np.array([[1, 0], [0, 0]])
    I12 = np.array([[0, 1], [0, 0]])
    I21 = np.array([[0, 0], [1, 0]])
    I22 = np.array([[0, 0], [0, 1]])
    
    CorM = (np.kron(CLL, I11) + np.kron(CLR, I12) + 
            np.kron(CLR.conj().T, I21) + np.kron(CLL, I22))
    
    return CorM, CLL, CLR, SigLL, SigLR

def generate_rescued_bh_profile(beta, tin, tmax, tWH, dt):
    """
    Generates the psi and Dpsi profile for the 'rescued black hole' case.
     tin: number of initial time steps
     tmax: number of black hole duration steps
     tWH: number of wormhole duration steps
    """
    tins = np.arange(-tin, 0) * dt
    ts = np.arange(0, tmax + 1) * dt
    # tWH indices: 1 to tWH
    twh_steps = np.arange(1, tWH + 1) * dt
    
    # Solve for pv = pi*v
    pv_func = lambda x: x - beta * np.cos(x / 2)
    pv = fsolve(pv_func, 0.1)[0]
    eps = (np.pi - pv) / 2
    
    # BH Region (ts)
    sin_eps = np.sin(eps)
    tan_eps = np.tan(eps)
    ephi = np.cosh(sin_eps * ts) / sin_eps
    p = np.arctan(tan_eps * np.tanh(sin_eps * ts))
    
    # Dpsi = exp(i*p)/sqrt(ephi^2 - 1)
    Dpsi_bh = np.exp(1j * p) / np.sqrt(ephi**2 - 1)
    # Correcting for t=0 where ephi^2 - 1 = 1/sin^2 - 1 = cos^2/sin^2 = cot^2
    # At t=0, p=0, ephi=1/sin_eps. ephi^2-1 = cos^2/sin^2. sqrt = cot.
    # Actually at t=0, p=0, ephi=1/sin_eps, Dpsi should be sin_eps/cos_eps = tan_eps.
    # Let's handle t=0 carefully to avoid division by zero or inf
    Dpsi_bh[0] = tan_eps
    
    # psi = 2*atan(tanh((sin_eps * ts - i*eps)/2))
    psi_bh = 2 * np.arctan(np.tanh((sin_eps * ts - 1j * eps) / 2))
    
    # Initial Region (tins)
    psiin = psi_bh[0] + tins * Dpsi_bh[0]
    Dpsiin = Dpsi_bh[0] * np.ones(tin)
    
    # Wormhole Region (tWH)
    Dpsif = np.abs(Dpsi_bh[-1])
    psif = psi_bh[-1]
    
    psiWH = psif + Dpsif * twh_steps
    DpsiWH = Dpsif * np.ones(tWH)
    
    psiev = np.concatenate([psiin, psi_bh, psiWH])
    Dpsiev = np.concatenate([Dpsiin, Dpsi_bh, DpsiWH])
    
    return psiev, Dpsiev
