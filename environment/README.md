# Recreate the environment

The canonical Python specification is `environment/requirements.txt`.
The tested platform is macOS 27.0 arm64, CPU, Python 3.12.14. Other platforms
have not been validated. The file pins the complete tested Python environment,
including notebook and audit tools. Only `appnope` is macOS-conditional.
No GPU, credentials, external dataset download, or proprietary solver is required.

From this repository's root, with Python 3.12.14 available:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r environment/requirements.txt
```

Alternatively, with uv installed:

```sh
uv venv .venv --python 3.12.14
uv pip install --python .venv/bin/python -r environment/requirements.txt
```

If uv does not discover an existing Python 3.12.14 interpreter, specify its
executable path with `uv venv .venv --python /path/to/python3.12`. This explicit
interpreter selection was needed on the validation host. Package installation
from the pinned file then succeeded from the local uv cache.

The documented runner prefix is `.venv/bin/python`. Keep CPU thread counts at one
for comparisons with the tested run:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python code/figure-reproduction/check_claims.py
.venv/bin/python code/figure-reproduction/reproduce_figures.py
.venv/bin/python -m unittest discover -s code/tests -v
.venv/bin/python skills/bulk-reconstruction/scripts/test_bulk_reconstruct.py
.venv/bin/python code/figure-reproduction/run_notebook.py
```

Generated outputs and environments are ignored by Git. The
[figure guide](../code/figure-reproduction/README.md) is authoritative for
inputs, outputs, parameter presets, runtime estimates, and limitations.

## TeX requirements

Paper compilation was tested with TeX Live 2026, pdfTeX 1.40.29, latexmk 4.87,
and BibTeX. Install a complete TeX Live/MacTeX environment separately; Python's
requirements file does not install TeX. Required packages include AMS packages,
geometry, graphicx, xcolor, natbib, hyperref, xr-hyper, TikZ/PGF, appendix, cuted,
threeparttable, wrapfig, and rotating, plus class dependencies.

```sh
.venv/bin/python code/figure-reproduction/fetch_tex_support.py
.venv/bin/python code/figure-reproduction/build_paper.py
```

The first command explicitly downloads the two exact publisher-owned support
files, verifies their SHA-256 hashes, and preserves existing files. It requires
internet access once; these files are excluded from Git. Their original LPPL
notices apply separately from the authors' CC BY 4.0 and MIT licenses. See the
[paper guide](../paper/README.md) for details. Numerical reproduction does not
require this download.

The second command builds an isolated paper copy and preserves supplied sources
and PDFs.

## Computational requirements and validation scope

Use a CPU workstation or laptop with several GiB of free RAM and disk space.
Geometry reconstruction stores dense kernels; larger time grids increase memory
substantially. Cached checks and full figure generation take seconds to a few
minutes on the tested machine. Notebook execution takes a few minutes. A fresh
50-point finite-p sweep took about 20 minutes in the pre-staging check and writes
another dataset of roughly 112 MiB. Request user approval before starting a full
sweep or increasing grids in an interactive agent session.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python code/scripts/generate_figure4_data.py
```

That command writes `data/generated/numerics/`, never overwriting published
snapshots. `--tf-max 1` is a two-point solver smoke test, not a reproduction of
post-evaporation Figure 4. See the [validation report](../supplementary/validation-report.md)
for setup attempts, actual checks, warnings, and validation scope.
