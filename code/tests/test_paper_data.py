"""Check published input completeness, selective tracking, and data rebuilding."""
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from scripts import generate_paper_figures as gpf


class PaperDataTests(unittest.TestCase):
    def test_published_snapshot_set_is_complete_and_unchanged(self):
        run, manifest = gpf.find_two_replica_run(gpf.manuscript_parameters()["finite_p"])
        self.assertEqual(run.parents[3], ROOT / "data")
        with manifest.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["t_f"]) for row in rows], list(np.arange(.5, 25.5, .5)))
        expected = {row["file"] for row in rows} | {"manifest.csv"}
        self.assertEqual({p.name for p in manifest.parent.iterdir()}, expected)
        checksums = (run / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(checksums), 51)
        for line in checksums:
            digest, relative = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((run / relative).read_bytes()).hexdigest(), digest)
        for row in rows:
            for name in ("chi", "disc", "twisted"):
                self.assertEqual(row[f"{name}_converged"], "True")

    @unittest.skipUnless((ROOT / ".git").exists(), "ignore rules require a Git checkout")
    def test_only_published_data_files_are_allowlisted(self):
        run, manifest = gpf.find_two_replica_run(gpf.manuscript_parameters()["finite_p"])
        allowed = [ROOT / "data/README.md", run / "README.md", run / "SHA256SUMS"]
        allowed += list(manifest.parent.iterdir())
        ignored = [ROOT / "data/random.pt", ROOT / "data/generated/numerics/new.pt",
                   ROOT / "data/numerics/eta_two_replica/runs/another/two_point_functions/manifest.csv",
                   manifest.parent / "extra_snapshot.pt", manifest.parent / "extra.csv"]
        paths = [str(p.relative_to(ROOT)) for p in allowed + ignored]
        result = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=ROOT,
                                input="\n".join(paths) + "\n", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()), {str(p.relative_to(ROOT)) for p in ignored})

    def test_short_numerical_rebuild_matches_saved_correlators(self):
        with tempfile.TemporaryDirectory() as destination:
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            result = subprocess.run(
                [sys.executable, str(ROOT / "code/scripts/generate_figure4_data.py"),
                 "--tf-max", "1", "--data-root", destination], cwd=destination,
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rebuilt = list(Path(destination).glob("numerics/**/two_point_functions/*.pt"))
            self.assertEqual(len(rebuilt), 2)
            _, manifest = gpf.find_two_replica_run(gpf.manuscript_parameters()["finite_p"])
            for path in rebuilt:
                actual = torch.load(path, map_location="cpu", weights_only=False)
                expected = torch.load(manifest.parent / path.name, map_location="cpu", weights_only=False)
                for name in ("G_chi", "G_disc", "G_twisted", "w"):
                    np.testing.assert_allclose(actual[name], expected[name], rtol=1e-7, atol=1e-8)
                for name in ("chi", "disc", "twisted"):
                    self.assertTrue(actual[f"{name}_info"]["converged"])

    def test_explicit_missing_data_root_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as destination:
            env = dict(os.environ, BH_DATA_ROOT=str(Path(destination) / "missing"))
            script = (f"import sys; sys.path.insert(0, {str(ROOT / 'code')!r}); "
                      "from scripts import generate_paper_figures as g; "
                      "g.find_two_replica_run(g.manuscript_parameters()['finite_p'])")
            result = subprocess.run([sys.executable, "-c", script], cwd=destination,
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileNotFoundError: no two-replica run", result.stderr)


if __name__ == "__main__":
    unittest.main()
