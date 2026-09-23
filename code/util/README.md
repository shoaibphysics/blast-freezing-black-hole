# Shared utilities

Reusable plotting, output, and path helpers. `paths.py` resolves paths from this
package and honors explicit `BH_DATA_ROOT` and `BH_FIGURE_ROOT` settings. It has
no dependency on a development checkout. Use the
[figure-reproduction wrappers](../figure-reproduction/README.md) to preserve
the supplied publication assets.

`plotting.py` contains optional diagnostic plots. The older `output_utils.py`
helper writes to `code/figures/`, outside the ignored reproduction directory;
the manuscript workflow does not use it. Use `paths.py` and the documented
wrappers for manuscript outputs.
