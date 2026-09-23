#!/usr/bin/env python3
"""Generate manuscript Figure 2, with panels (a)–(f) in caption order.

Run `.venv/bin/python code/scripts/generate_figure2.py --out-dir code/figure-reproduction/generated/figures` to regenerate the annotated gate map and Z-only wavepacket panels
and assemble the figure. Use --assemble-only after generating the panels.
The formation gate map and hand-drawn fig3b.png / fig3d.png are source assets.
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
from util.paths import FIGURE_INPUT_ROOT, FIGURE_ROOT

FIG_DIR = FIGURE_INPUT_ROOT
PANELS = (
    ("a", "fig3a_gate_color_map.png"),
    ("b", "fig3f_wavepacket_late.png"),
    ("c", "fig3b.png"),
    ("d", "fig3c_formation_gate_color_map.png"),
    ("e", "fig3e_wavepacket_early.png"),
    ("f", "fig3d.png"),
)


def assemble_figure2(out_dir: Path, *, diagram_dir: Path = FIG_DIR) -> dict[str, str]:
    """Assemble existing single-panel plots without cropping or distorting data."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.5, 11.8), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=0.01, right=0.99, bottom=0.01,
                           top=0.965, wspace=0.05, hspace=0.11)
    for slot, (label, name) in zip(grid, PANELS):
        path = (diagram_dir if name in {"fig3b.png", "fig3d.png"} else out_dir) / name
        data = plt.imread(path)
        if "wavepacket" in name and data.shape[1] > data.shape[0]:
            raise ValueError(f"{path} still has two panels; regenerate before assembling.")
        ax = fig.add_subplot(slot)
        ax.imshow(data, interpolation="none")
        ax.set_axis_off()
        # Place labels relative to each layout slot, independent of image aspect.
        box = slot.get_position(fig)
        fig.text(box.x0 + 0.005, box.y1 + 0.008, f"({label})",
                 fontsize=14, fontweight="bold", ha="left", va="bottom")
    files = {}
    for extension in ("png", "pdf"):
        path = out_dir / f"figure2_bulk_geometry.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        files[extension] = str(path)
        print(f"wrote {path}")
    plt.close(fig)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()
    if args.assemble_only:
        assemble_figure2(args.out_dir)
    else:
        from scripts.generate_paper_figures import FigureParams, generate_figure2
        generate_figure2(FigureParams(dt=args.dt), args.out_dir.resolve())


if __name__ == "__main__":
    main()
