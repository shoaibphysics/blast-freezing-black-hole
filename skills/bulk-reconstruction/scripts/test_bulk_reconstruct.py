"""Numerical contract tests. Run from any cwd: python /path/to/this_file.py."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import bulk_reconstruct as br


def fixture(N, T=5, complex_basis=False):
    """Canonical equal-time frames of independent ambient fermion modes."""
    rng = np.random.default_rng(140 + N)
    D = T*N + 4
    rows = []
    for _ in range(T):
        values = rng.normal(size=(D, N))
        if complex_basis:
            values = values + 1j*rng.normal(size=(D, N))
        q, _ = np.linalg.qr(values)
        rows.append(q.conj().T)
    frame = np.vstack(rows)
    A = frame @ frame.conj().T
    # A nonzero imaginary antisymmetric covariance distinguishes W from Re W.
    K = rng.normal(size=(D, D))
    K -= K.T
    K /= 4*np.linalg.norm(K, 2)
    W = frame @ (0.5*np.eye(D) + 1j*K) @ frame.conj().T
    times = np.array([-2., -1.4, 0., 0.8, 2.5])[:T]
    return times, A, W


class ReconstructionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine, _, _ = br.backend()

    def test_majorana_convention_equivalence_and_wrong_sign(self):
        times, A, W = fixture(3)
        for kind, data in [("majorana-anticommutator", A),
                           ("majorana-wightman", W),
                           ("majorana-greater", -1j*W),
                           ("majorana-normalized", 2*W)]:
            actual, report = br.validate_input(data.reshape(5, 3, 5, 3), times, 3, kind)
            np.testing.assert_allclose(actual, A, atol=1e-14)
            self.assertLess(report["equal_time_canonical"]["residual"], 1e-14)
        with self.assertRaisesRegex(ValueError, "negative eigenvalue"):
            br.validate_input(1j*W, times, 3, "majorana-greater")
        with self.assertRaisesRegex(ValueError, "equal_time_canonical"):
            br.validate_input(W, times, 3, "majorana-normalized")

    def test_complex_basis_keeps_imaginary_anticommutator(self):
        times, A, _ = fixture(4, complex_basis=True)
        actual, _ = br.validate_input(-0.7j*A, times, 4, "complex-greater-lesser", 0.3j*A)
        np.testing.assert_allclose(actual, A, atol=1e-14)
        self.assertGreater(np.max(np.abs(actual.imag)), 0.1)
        with self.assertRaisesRegex(ValueError, "require the lesser"):
            br.validate_input(-0.7j*A, times, 4, "complex-greater-lesser")
        with self.assertRaisesRegex(ValueError, "reality"):
            br.validate_input(A, times, 4, "majorana-anticommutator")

    def test_invalid_symmetry_and_indefinite_gram_rejected(self):
        times, A, _ = fixture(3)
        B = A.copy()
        B[0, 3] += 0.1
        with self.assertRaisesRegex(ValueError, "hermiticity"):
            br.validate_input(B, times, 3, "majorana-anticommutator")
        B = np.eye(15)
        B[0, 3] = B[3, 0] = 2
        with self.assertRaisesRegex(ValueError, "negative eigenvalue"):
            br.validate_input(B, times, 3, "majorana-anticommutator")
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            br.validate_input(A, times[::-1], 3, "majorana-anticommutator")

    def test_gates_whiten_and_preserve_complete_cut_norm_for_arbitrary_N(self):
        for N in (1, 2, 3, 4):
            for use_complex in (False, True):
                with self.subTest(N=N, complex=use_complex):
                    _, A, _ = fixture(N, complex_basis=use_complex)
                    gates, _, _, mask, checks = br.reconstruct(A, N, self.engine)
                    self.assertEqual(len(gates), 10)
                    self.assertEqual(mask.sum(), 10)
                    self.assertLess(checks["retained_max_error"], 1e-10)
                    psi = np.zeros(N, complex)
                    psi[0] = 1
                    R, L = br.propagate(gates, N, 5, 0, 0, psi, "R", self.engine)
                    # Before the packet hits the sampled window's far edge, each
                    # complete anti-diagonal is a circuit cut with norm one.
                    r, l = br.observable(R), br.observable(L)
                    for layer in range(5):
                        total = sum(np.nansum([r[u, layer-u], l[u, layer-u]])
                                    for u in range(layer+1))
                        self.assertAlmostEqual(total, 1, places=10)

    def test_rank_loss_is_not_regularized(self):
        A = np.kron(np.ones((5, 5)), np.eye(3))
        A, report = br.validate_input(A, np.arange(5.), 3, "majorana-anticommutator")
        self.assertEqual(report["anticommutator"]["numerical_rank"], 3)
        gates, _, _, mask, _ = br.reconstruct(A, 3, self.engine)
        self.assertFalse(gates)
        self.assertFalse(mask.any())

    def test_full_complex_projection_and_mask(self):
        psi = np.array([1, 1j, 0]) / np.sqrt(2)
        P = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 2]])
        P = br.check_projection(P, 3)
        self.assertAlmostEqual(br.observable(psi, P), 1)
        self.assertAlmostEqual(br.observable(0.5*psi, P), 0.25)
        self.assertAlmostEqual(br.observable(psi), 1)
        self.assertTrue(np.isnan(br.observable(np.full(3, np.nan, complex), P)))
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            br.check_projection([[0, 1], [0, 0]], 2)
        with self.assertRaisesRegex(ValueError, "3 by 3"):
            br.check_projection(np.eye(2), 3)

    def test_existing_block_norm_definition(self):
        theta = np.array([0.1, 0.7, 1.2])
        s, c = np.diag(np.sin(theta)), np.diag(np.cos(theta))
        U = np.block([[s, c], [c, -s]])
        _, norms, displayed = br.gate_block_norms({(0, 1): U}, 3)
        expected = np.sqrt(np.sum(np.sin(theta)**2))
        self.assertAlmostEqual(norms[0], expected)
        self.assertAlmostEqual(displayed[0], expected/3)
        for N in (1, 2, 3, 5):
            _, _, values = br.gate_block_norms({(0, 1): np.eye(2*N)}, N)
            self.assertAlmostEqual(values[0], 1/np.sqrt(N))

    def test_cli_end_to_end_all_optional_plots(self):
        with tempfile.TemporaryDirectory(prefix="bulk-skill-test-") as directory:
            d = Path(directory)
            times, _, W = fixture(3)
            np.savez(d / "input.npz", times=times, two_point=-1j*W)
            np.save(d / "psi.npy", np.array([1., 1j, 0])/np.sqrt(2))
            np.save(d / "P.npy", np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 2]]))
            command = [sys.executable, str(Path(br.__file__).resolve())]

            def run(*args):
                result = subprocess.run(command + list(args), cwd=d, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run("reconstruct", "--input", "input.npz", "--kind", "majorana-greater",
                "--flavors", "3", "--output", "run")
            report = json.loads((d / "run/diagnostics.json").read_text())
            self.assertEqual(report["retained_gate_count"], 10)
            run("gate-map", "--reconstruction", "run/reconstruction.npz",
                "--output", "gate-map.png")
            for quantity in ("amplitude", "intensity", "projection"):
                extra = ["--projection", "P.npy"] if quantity == "projection" else []
                run("wavepacket", "--reconstruction", "run/reconstruction.npz",
                    "--initial-state", "psi.npy", "--start-u", "0", "--start-v", "1",
                    "--mover", "R", "--quantity", quantity, "--output", f"{quantity}.png", *extra)
                self.assertGreater((d / f"{quantity}.png").stat().st_size, 1000)
            with np.load(d / "projection.npz", allow_pickle=False) as z:
                self.assertAlmostEqual(z["R_observable"][0, 1], 1)
                self.assertEqual(z["projection"].shape, (3, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
