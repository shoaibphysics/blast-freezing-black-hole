# Scientific entry points

Use [the figure-reproduction guide](../figure-reproduction/README.md) for public
commands, figure mappings, outputs, parameter presets, and known limitations.
Use [environment/README.md](../../environment/README.md) for installation.

- `generate_paper_figures.py --manuscript`: the supported Nature manuscript workflow.
- `generate_figure2.py`: geometry panel generation and composite assembly.
- `generate_figure4.py`: finite-p panel assembly after those panels exist.
- `generate_figure4_data.py`: a new timestamped finite-p data sweep under `data/generated/`.

These are the canonical builders; no duplicate entry points are kept under `paper/`.
The three required Figure 2 image inputs live in [figures/](figures/README.md).
Generated plots default to `code/figure-reproduction/generated/figures/`;
`--out-dir` or `BH_FIGURE_ROOT` can select another output directory.
After generating the numerical panels, `generate_figure4.py` reads them from
its output directory unless `--source-dir` is explicitly supplied.

The maintained notebook is [paper_figures.ipynb](../notebooks/paper_figures.ipynb).
Its calculations and presets are preserved; outputs and private metadata were
removed, cell IDs assigned, and output paths redirected to the ignored generated area.

The historical non-`--manuscript` workflow includes optional p=4 helpers.
Those experiments and their datasets are outside this publication's scope.
`BH_DATA_ROOT` can select another directory containing `numerics/`; missing
inputs raise an error rather than falling back to the development repository.
