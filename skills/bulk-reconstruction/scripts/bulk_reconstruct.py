#!/usr/bin/env python3
"""Validated arbitrary-flavor adapter to the repository's bulk circuit engine."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


KINDS = (
    "majorana-anticommutator", "majorana-wightman", "majorana-greater",
    "majorana-normalized", "complex-anticommutator", "complex-greater-lesser",
)


def backend(repo_root=None):
    roots = [Path(repo_root)] if repo_root else [*Path(__file__).resolve().parents,
                                               Path.cwd(), *Path.cwd().parents]
    for root in roots:
        path = root / "code/src/bulk_reconstruction/reconstruction.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("skill_bulk_backend", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module, path.resolve(), hashlib.sha256(path.read_bytes()).hexdigest()
    raise ValueError("Reconstruction checkout not found; supply --repo-root PATH")


def check_times(times):
    times = np.asarray(times)
    if (times.ndim != 1 or len(times) < 2 or np.iscomplexobj(times)
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be finite, real, strictly increasing, length >= 2")
    return times.astype(float)


def matrix(value, T, N):
    value = np.asarray(value)
    if value.shape not in ((T * N, T * N), (T, N, T, N)):
        raise ValueError(f"Expected {(T*N, T*N)} or {(T, N, T, N)}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ValueError("Correlator must be a finite numeric array")
    return value.reshape(T * N, T * N).astype(np.complex128)


def validate_input(two_point, times, N, kind, lesser=None, atol=1e-10, rtol=1e-8):
    """Return canonical anticommutator A and measured validation diagnostics."""
    times = check_times(times)
    if not isinstance(N, (int, np.integer)) or N < 1 or kind not in KINDS:
        raise ValueError("Specify a positive integer N and a supported input kind")
    if not np.isfinite(atol + rtol) or atol < 0 or rtol < 0:
        raise ValueError("Validation tolerances must be finite and nonnegative")
    T = len(times)
    X = matrix(two_point, T, N)
    report = {"kind": kind, "N": int(N), "T": T, "atol": atol, "rtol": rtol}

    def symmetry(name, residual, source):
        err = float(np.max(np.abs(residual)))
        limit = float(atol + rtol * np.max(np.abs(source)))
        report[name] = {"residual": err, "limit": limit}
        if err > limit:
            raise ValueError(f"{name}: residual {err:g} exceeds {limit:g}")

    def psd(name, value):
        vals = np.linalg.eigvalsh((value + value.conj().T) / 2)
        limit = float(atol + rtol * np.max(np.abs(vals)))
        report[name] = {"minimum_eigenvalue": float(vals[0]), "tolerance": limit,
                        "numerical_rank": int(np.count_nonzero(vals > limit))}
        if vals[0] < -limit:
            raise ValueError(f"{name}: negative eigenvalue {vals[0]:g} < {-limit:g}")

    if kind == "complex-greater-lesser":
        if lesser is None:
            raise ValueError("Complex greater data require the lesser matrix as well")
        Y = matrix(lesser, T, N)
        symmetry("greater_antihermiticity", X + X.conj().T, X)
        symmetry("lesser_antihermiticity", Y + Y.conj().T, Y)
        psd("greater_covariance", 1j * X)
        psd("lesser_covariance", -1j * Y)
        A = 1j * (X - Y)
    else:
        if lesser is not None:
            raise ValueError("lesser is only used with complex-greater-lesser")
        if kind == "majorana-greater":
            symmetry("greater_antihermiticity", X + X.conj().T, X)
            psd("wightman", 1j * X)
            A = 1j * (X + X.T)
        elif kind in ("majorana-wightman", "majorana-normalized"):
            symmetry("correlator_hermiticity", X - X.conj().T, X)
            psd("wightman", X)
            A = (X + X.T) / (2 if kind == "majorana-normalized" else 1)
        else:
            A = X.copy()

    raw_A = A.copy()
    symmetry("anticommutator_hermiticity", A - A.conj().T, A)
    if kind.startswith("majorana"):
        symmetry("anticommutator_reality", A.imag, A)
        A = A.real
    blocks = np.stack([A[t*N:(t+1)*N, t*N:(t+1)*N] for t in range(T)])
    symmetry("equal_time_canonical", blocks - np.eye(N), np.eye(N))
    psd("anticommutator", A)
    A = (A + A.conj().T) / 2  # Only cleanup after every validation passed.
    report["cleanup_max_abs"] = float(np.max(np.abs(A - raw_A)))
    return A, report


def reconstruct(A, N, engine, gate_tol=1e-6):
    """Apply stricter gate checks without repairing or regularizing the engine."""
    if not np.isfinite(gate_tol) or gate_tol <= 0 or gate_tol > 1e-2:
        raise ValueError("gate_tol must be positive and no looser than backend cutoff 1e-2")
    T = len(A) // N
    gates, KL, KR, raw_mask, _, _ = engine.compute_bulk_gates(A, N, T)
    mask = raw_mask.copy()
    identity = np.eye(2*N)
    max_unitarity = max_whitening = 0.0
    accepted_errors = {}
    for (u, v), U in gates.items():
        sl = slice(u*N, (v+1)*N)
        gram = A[sl, sl]
        O = np.vstack((KL[u+1, v, :, sl], KR[u, v, :, sl]))
        I = np.vstack((KR[u, v-1, :, sl], KL[u, v, :, sl]))
        unitary = float(np.linalg.norm(U @ U.conj().T - identity))
        whitening = float(max(np.linalg.norm(O @ gram @ O.conj().T - identity),
                              np.linalg.norm(I @ gram @ I.conj().T - identity)))
        max_unitarity = max(max_unitarity, unitary)
        max_whitening = max(max_whitening, whitening)
        if not np.isfinite(unitary + whitening) or max(unitary, whitening) > gate_tol:
            mask[u, v] = False
        else:
            accepted_errors[u, v] = max(unitary, whitening)
    gates, mask = engine.enforce_gate_inclusion(gates, mask)
    diagnostics = {
        "backend_initial_gate_count": int(raw_mask.sum()),
        "retained_gate_count": len(gates),
        "possible_gate_count": T * (T - 1) // 2,
        "gate_tolerance": gate_tol,
        "raw_max_unitarity_error": max_unitarity,
        "raw_max_whitening_error": max_whitening,
        "retained_max_error": max((accepted_errors[k] for k in gates), default=None),
        "backend_cholesky_pivot_floor": 1e-8,
        "backend_initial_gate_tolerance": 1e-2,
    }
    return gates, KL, KR, mask, diagnostics


def output_paths(path):
    path = Path(path)
    data = path.with_suffix(".npz")
    if path.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ValueError("Plot output must have .png, .pdf, or .svg extension")
    if path.exists() or data.exists():
        raise ValueError("Output exists; choose a new plot filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, data


def load_reconstruction(path):
    with np.load(path, allow_pickle=False) as archive:
        times = check_times(archive["times"])
        N = int(archive["N"])
        indices, matrices = archive["gate_indices"], archive["gates"]
        mask = archive["is_legit"]
        gate_tol = float(archive["gate_tolerance"])
        source_hash = str(archive["backend_sha256"])
    if N < 1 or indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("Invalid flavor count or gate indices")
    if not np.isfinite(gate_tol) or not 0 < gate_tol <= 1e-2:
        raise ValueError("Invalid saved gate tolerance")
    if (not np.issubdtype(indices.dtype, np.integer)
            or matrices.shape != (len(indices), 2*N, 2*N)
            or not np.all(np.isfinite(matrices))):
        raise ValueError("Invalid gate array")
    T = len(times)
    expected = np.zeros((T, T), dtype=bool)
    gates = {}
    for (u, v), U in zip(indices, matrices):
        if not (0 <= u < v < T) or (u, v) in gates:
            raise ValueError("Duplicate or out-of-range gate")
        if np.linalg.norm(U @ U.conj().T - np.eye(2*N)) > gate_tol:
            raise ValueError("Saved gate fails unitarity check")
        gates[int(u), int(v)] = U
        expected[u, v] = True
    if not np.array_equal(expected, mask):
        raise ValueError("Saved gate mask and gate indices disagree")
    return times, N, gates, source_hash


def gate_block_norms(gates, N):
    """Preserve the paper's gate 'angle' map: Frobenius block norm divided by N."""
    indices = np.array(sorted(gates), dtype=int).reshape(-1, 2)
    norms = np.empty(len(indices))
    for k, ij in enumerate(indices):
        norms[k] = np.linalg.norm(gates[tuple(ij)][:N, :N], ord="fro")
    return indices, norms, norms / N


def observable(links, P=None):
    """Link intensity or full Hermitian quadratic form; NaNs retain masks."""
    if P is None:
        return np.sum(np.abs(links)**2, axis=-1)
    values = np.einsum("...a,ab,...b->...", links.conj(), P, links)
    finite = np.isfinite(values)
    if np.any(np.abs(values.imag[finite]) > 1e-9 * (1 + np.abs(values.real[finite]))):
        raise ValueError("Projection has a non-roundoff imaginary component")
    return values.real


def check_projection(P, N):
    P = np.asarray(P, dtype=complex)
    if P.shape != (N, N) or not np.all(np.isfinite(P)):
        raise ValueError(f"Projection must be a finite {N} by {N} matrix")
    if np.max(np.abs(P - P.conj().T)) > 1e-10 + 1e-8 * np.max(np.abs(P)):
        raise ValueError("Projection must be Hermitian")
    return (P + P.conj().T) / 2


def propagate(gates, N, T, start_u, start_v, initial_state, mover, engine):
    psi = np.asarray(initial_state, dtype=complex)
    if psi.shape != (N,) or not np.all(np.isfinite(psi)):
        raise ValueError(f"Initial state must be a finite {N}-component vector")
    norm = float(np.vdot(psi, psi).real)
    if not np.isclose(norm, 1, atol=1e-10, rtol=1e-8):
        raise ValueError(f"Initial state norm squared is {norm:g}, expected 1")
    if not (0 <= start_u <= start_v < T) or mover not in ("R", "L"):
        raise ValueError("Invalid injection indices or mover")
    if start_u == start_v:
        if mover != "R":
            raise ValueError("Boundary injection is on an outgoing R link")
    elif (start_u, start_v) not in gates:
        raise ValueError("Injection gate is outside the valid reconstructed geometry")
    return engine.simulate_wavepacket(gates, N, T, start_u, start_v, psi, mover == "L")


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def coordinates(times, u, v):
    return ((times[v] - times[u]) / 2, (times[v] + times[u]) / 2)


def finish_plot(fig, ax, artist, times, label, path):
    ax.set(xlabel="Bulk depth z", ylabel="Bulk time t", xlim=(0, np.ptp(times)/2),
           ylim=(times[0], times[-1]))
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(artist, ax=ax, label=label)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plotting().close(fig)


def run_reconstruction(args):
    target = Path(args.output)
    if any((target / name).exists() for name in ("reconstruction.npz", "diagnostics.json")):
        raise ValueError("Reconstruction output exists; choose a new output directory")
    engine, source, source_hash = backend(args.repo_root)
    with np.load(args.input, allow_pickle=False) as data:
        times = check_times(data["times"])
        if args.flavors < 1:
            raise ValueError("flavors must be positive")
        gib = 2 * len(times)**3 * args.flavors**2 * (8 if args.kind.startswith("majorana") else 16) / 2**30
        if not np.isfinite(args.max_kernel_gib) or args.max_kernel_gib <= 0:
            raise ValueError("max-kernel-gib must be finite and positive")
        if gib > args.max_kernel_gib:
            raise ValueError(f"Kernel estimate {gib:.3g} GiB exceeds limit {args.max_kernel_gib:g}; use fewer samples or an explicit larger budget")
        A, report = validate_input(data["two_point"], times, args.flavors, args.kind,
                                   data["lesser"] if "lesser" in data else None,
                                   args.atol, args.rtol)
    labels = args.flavor_labels or [str(i) for i in range(args.flavors)]
    if len(labels) != args.flavors or len(set(labels)) != args.flavors:
        raise ValueError("Supply exactly N distinct flavor labels in array order")
    gates, KL, KR, mask, checks = reconstruct(A, args.flavors, engine, args.gate_tol)
    report.update(checks)
    report.update(kernel_estimate_gib=gib, backend_path=str(source),
                  backend_sha256=source_hash, flavor_labels=labels,
                  input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    indices = np.array(sorted(gates), dtype=int).reshape(-1, 2)
    matrices = np.array([gates[tuple(k)] for k in indices], dtype=A.dtype).reshape(-1, 2*args.flavors, 2*args.flavors)
    payload = dict(times=times, N=args.flavors, A=A, gate_indices=indices, gates=matrices,
                   is_legit=mask, gate_tolerance=args.gate_tol, backend_path=str(source),
                   backend_sha256=source_hash, flavor_labels=np.array(labels))
    if args.save_kernels:
        payload.update(KL_all=KL, KR_all=KR)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "reconstruction.npz", **payload)
    (target / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(gates)} / {len(times)*(len(times)-1)//2} valid gates to {target}")


def run_gate_map(args):
    from matplotlib.collections import PolyCollection
    times, N, gates, _ = load_reconstruction(args.reconstruction)
    if not gates:
        raise ValueError("No valid bulk gates to plot; inspect reconstruction diagnostics")
    indices, norms, values = gate_block_norms(gates, N)
    label = r"$\|U_{11}\|_F/N$"
    path, data_path = output_paths(args.output)
    edges = np.r_[times[0] - (times[1]-times[0])/2,
                  (times[:-1]+times[1:])/2, times[-1] + (times[-1]-times[-2])/2]
    polygons = []
    for u, v in indices:
        polygons.append([((b-a)/2, (b+a)/2) for a, b in
                         [(edges[u], edges[v]), (edges[u+1], edges[v]),
                          (edges[u+1], edges[v+1]), (edges[u], edges[v+1])]])
    plt = plotting()
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    artist = PolyCollection(polygons, array=values, cmap="viridis", edgecolors="none")
    ax.add_collection(artist)
    ax.set_title("Bulk gate strength")
    finish_plot(fig, ax, artist, times, label, path)
    z, t = coordinates(times, indices[:, 0], indices[:, 1])
    np.savez_compressed(data_path, gate_indices=indices, block_norms=norms, values=values,
                        normalization="frobenius_block_norm / N", times=times, z=z, t=t, N=N)


def run_wavepacket(args):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize, TwoSlopeNorm
    times, N, gates, source_hash = load_reconstruction(args.reconstruction)
    engine, _, actual_hash = backend(args.repo_root)
    if source_hash != actual_hash:
        raise ValueError("Backend changed since reconstruction; reconstruct with this checkout first")
    P = None
    if args.quantity == "projection":
        if args.projection is None:
            raise ValueError("Projection plot requires --projection with an explicit N by N matrix")
        P = check_projection(np.load(args.projection, allow_pickle=False), N)
    elif args.projection is not None:
        raise ValueError("Use --quantity projection when supplying --projection")
    psi = np.load(args.initial_state, allow_pickle=False)
    R, L = propagate(gates, N, len(times), args.start_u, args.start_v, psi, args.mover, engine)
    r_intensity, l_intensity = observable(R), observable(L)
    rv, lv = (observable(R, P), observable(L, P)) if P is not None else (r_intensity, l_intensity)
    if args.quantity == "amplitude":
        rv, lv = np.sqrt(rv), np.sqrt(lv)
    segments, colors = [], []
    for values, du, dv in ((rv, 0, 1), (lv, 1, 0)):
        for u, v in np.argwhere(np.isfinite(values)):
            if u+du < len(times) and v+dv < len(times) and u+du <= v+dv:
                segments.append([coordinates(times, u, v), coordinates(times, u+du, v+dv)])
                colors.append(values[u, v])
    if not segments:
        raise ValueError("No outgoing links within the sampled window to plot")
    path, data_path = output_paths(args.output)
    cmap, norm = "viridis", Normalize(0, 1)
    label = "Amplitude magnitude (flavor norm)" if args.quantity == "amplitude" else "Intensity (norm squared)"
    if P is not None:
        eig = np.linalg.eigvalsh(P)
        lo, hi = min(0., eig[0]), max(0., eig[-1])
        if lo < 0 < hi:
            cmap, norm = "RdBu_r", TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
        else:
            norm = Normalize(lo, hi if hi > lo else lo+1)
        label = r"Projected density $\mathrm{Re}(\psi^\dagger P\psi)$"
    plt = plotting()
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    artist = LineCollection(segments, array=np.asarray(colors), cmap=cmap, norm=norm, linewidths=2)
    ax.add_collection(artist)
    ax.plot(*coordinates(times, args.start_u, args.start_v), "o", color="black", markersize=4)
    ax.set_title(f"Wavepacket {args.quantity}, initial {args.mover} mover")
    finish_plot(fig, ax, artist, times, label, path)
    payload = dict(times=times, N=N, R_links=R, L_links=L, R_intensity=r_intensity,
                   L_intensity=l_intensity, R_observable=rv, L_observable=lv,
                   quantity=args.quantity, initial_state=psi, start_u=args.start_u,
                   start_v=args.start_v, mover=args.mover)
    if P is not None:
        payload.update(projection=P, projection_eigenvalues=np.linalg.eigvalsh(P))
    np.savez_compressed(data_path, **payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("reconstruct", help="Validate two-point data and reconstruct gates")
    rec.add_argument("--input", required=True)
    rec.add_argument("--kind", choices=KINDS, required=True)
    rec.add_argument("--flavors", type=int, required=True)
    rec.add_argument("--flavor-labels", nargs="+")
    rec.add_argument("--output", required=True)
    rec.add_argument("--repo-root")
    rec.add_argument("--save-kernels", action="store_true")
    rec.add_argument("--atol", type=float, default=1e-10)
    rec.add_argument("--rtol", type=float, default=1e-8)
    rec.add_argument("--gate-tol", type=float, default=1e-6)
    rec.add_argument("--max-kernel-gib", type=float, default=2.)
    rec.set_defaults(run=run_reconstruction)
    gate = sub.add_parser("gate-map", help="Plot the existing Frobenius block norm / N")
    gate.add_argument("--reconstruction", required=True)
    gate.add_argument("--output", required=True)
    gate.set_defaults(run=run_gate_map)
    wave = sub.add_parser("wavepacket", help="Plot amplitude, intensity or a supplied projection")
    wave.add_argument("--reconstruction", required=True)
    wave.add_argument("--initial-state", required=True)
    wave.add_argument("--start-u", type=int, required=True)
    wave.add_argument("--start-v", type=int, required=True)
    wave.add_argument("--mover", choices=("R", "L"), required=True)
    wave.add_argument("--quantity", choices=("amplitude", "intensity", "projection"), required=True)
    wave.add_argument("--projection")
    wave.add_argument("--repo-root")
    wave.add_argument("--output", required=True)
    wave.set_defaults(run=run_wavepacket)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
