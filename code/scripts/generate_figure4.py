#!/usr/bin/env python3
"""Assemble manuscript Figure 4 from its saved numerical panels.

Run `.venv/bin/python code/scripts/generate_figure4.py --out-dir code/figure-reproduction/generated/figures` to add labels (a) and (b).
The legacy fig5a/fig5b source images are produced by generate_paper_figures.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
from util.paths import FIGURE_ROOT
PANELS = (
    ("a", "fig5a_renyi2_quantum_mi_heatmap.png"),
    ("b", "fig5b_renyi2_action_difference.png"),
)


def assemble_figure4(out_dir: Path, *, source_dir: Path | None = None) -> dict[str, str]:
    """Preserve both plots and add labels in caption order, as in Figure 2."""
    out_dir = Path(out_dir).resolve()
    source_dir = out_dir if source_dir is None else Path(source_dir).resolve()
    panels = [plt.imread(source_dir / name) for _, name in PANELS]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.5, 4.8), facecolor="white")
    grid = fig.add_gridspec(1, 2, left=0.01, right=0.99, bottom=0.01,
                           top=0.91, wspace=0.05)
    for slot, (label, _), data in zip(grid, PANELS, panels):
        ax = fig.add_subplot(slot)
        ax.imshow(data, interpolation="none")
        ax.set_anchor("N")
        ax.set_axis_off()
        box = slot.get_position(fig)
        fig.text(box.x0 + 0.005, box.y1 + 0.012, f"({label})",
                 fontsize=26, fontweight="bold", ha="left", va="bottom")
    files = {}
    for extension in ("png", "pdf"):
        path = out_dir / f"figure4_finite_p.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        files[extension] = str(path)
        print(f"wrote {path}")
    plt.close(fig)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="Read panels from this directory (default: --out-dir).")
    args = parser.parse_args()
    assemble_figure4(args.out_dir, source_dir=args.source_dir)


if __name__ == "__main__":
    main()
