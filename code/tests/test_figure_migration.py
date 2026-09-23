"""Regression checks for canonical plotting imports and manuscript annotations."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
from scripts import generate_paper_figures as gpf


class FigureMigrationTests(unittest.TestCase):
    def test_plotting_modules_use_canonical_code_and_relocated_inputs(self):
        for name in ("generate_paper_figures", "generate_figure2", "generate_figure4"):
            canonical = importlib.import_module("scripts." + name)
            self.assertEqual(Path(canonical.__file__).resolve(),
                             REPO_ROOT / "code/scripts" / (name + ".py"))
        from util.paths import FIGURE_INPUT_ROOT, FIGURE_ROOT
        self.assertEqual(gpf.PAPER_FIG_DIR, FIGURE_INPUT_ROOT)
        for name in ("fig3b.png", "fig3c_formation_gate_color_map.png", "fig3d.png"):
            self.assertTrue((FIGURE_INPUT_ROOT / name).is_file())
        self.assertFalse((REPO_ROOT / "paper/figures").exists())
        if "BH_FIGURE_ROOT" not in os.environ:
            self.assertEqual(FIGURE_ROOT, REPO_ROOT / "code/figure-reproduction/generated/figures")
        self.assertEqual(Path(gpf.brl.__file__).resolve(),
                         REPO_ROOT / "code/src/bulk_reconstruction/reconstruction.py")
        self.assertTrue(hasattr(gpf, "generate_fig7_p4_figures"))

    def test_direct_entry_points_work_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as cwd:
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            for folder in ("code/scripts",):
                for name in ("generate_paper_figures", "generate_figure2", "generate_figure4"):
                    result = subprocess.run(
                        [sys.executable, str(REPO_ROOT / folder / (name + ".py")), "--help"],
                        cwd=cwd, env=env, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_manuscript_presets_keep_distinct_figure_parameters(self):
        presets = gpf.manuscript_parameters()
        self.assertEqual((presets["geometry"].mu, presets["geometry"].tev), (0.1, 15))
        self.assertEqual((presets["operator_size"].mu, presets["operator_size"].tev), (0.5, 6))
        self.assertEqual(presets["operator_size"].size_fixed_v_time, 8)
        self.assertEqual(presets["finite_p"].mi_tf, 25)
        presets["geometry"].tev = 99
        self.assertEqual(presets["operator_size"].tev, 6)

    def test_region_boundaries_and_labels_follow_null_coordinates(self):
        params = gpf.FigureParams(tmax=40)
        times = params.tgrid()
        u, v = np.meshgrid(times, times, indexing="ij")
        # A finite reconstructed domain, so the lines must also stop at its edge.
        norms = np.where((u <= v) & ((v-u)/2 <= 12), 1.0, np.nan)
        fig, ax = gpf.plt.subplots()
        try:
            gpf.annotate_evaporation_regions(ax, {"tgrid": times, "norms_mat": norms}, params)
            self.assertEqual(len(ax.lines), 2)
            for index, line in enumerate(ax.lines):
                z, t = map(np.asarray, line.get_data())
                valid = np.isfinite(z) & np.isfinite(t)
                self.assertTrue(np.any(valid))
                np.testing.assert_allclose((t+z if index == 0 else t-z)[valid], params.tev)
                self.assertLessEqual(z[valid].max(), 12)
            labels = {text.get_text(): text.get_position() for text in ax.texts}
            self.assertEqual(set(labels), {"I", "II", "III"})
            z, t = labels["I"]
            self.assertLess(t+z, params.tev)
            z, t = labels["II"]
            self.assertLess(t-z, params.tev)
            self.assertGreater(t+z, params.tev)
            z, t = labels["III"]
            self.assertGreater(t-z, params.tev)
        finally:
            gpf.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
