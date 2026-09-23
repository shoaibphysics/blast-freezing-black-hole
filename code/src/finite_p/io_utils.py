"""Input/output helpers for the evaporation numerics package.

This module contains small bookkeeping utilities used by scripts and notebooks:
finding the project root, constructing filesystem-safe labels, and saving
tables with complex-valued columns in a CSV-friendly form.  It should stay
independent of the physics solvers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the numerics project root by walking upward from ``start``.

    The project root is identified by the presence of both ``src`` and
    ``pyproject.toml``.  This lets notebooks locate the package reliably even
    when they are launched from inside a subfolder.
    """

    current = Path.cwd().resolve() if start is None else Path(start).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "src").exists() and (candidate / "pyproject.toml").exists():
            return candidate

    raise RuntimeError(f"Could not find numerics project root from {current}")


def slug_float(x: float, ndigits: int = 2) -> str:
    """Return a filesystem-safe label for a floating-point value.

    Examples
    --------
    ``0.3`` becomes ``0p30`` and ``-0.05`` becomes ``m0p05``.
    """

    return f"{float(x):.{ndigits}f}".replace("-", "m").replace(".", "p")


def expand_complex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Split complex-valued dataframe columns into real and imaginary columns.

    CSV files do not have a clean native complex-number type.  This function
    converts a column ``z`` containing complex numbers into ``z_real`` and
    ``z_imag`` while leaving ordinary real/string columns unchanged.
    """

    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        vals = df[col]
        sample = vals.dropna().head(10)

        is_complex = False
        if len(sample):
            is_complex = any(isinstance(v, complex) for v in sample) or np.iscomplexobj(
                vals.to_numpy()
            )

        if is_complex:
            arr = vals.to_numpy(dtype=np.complex128)
            out[f"{col}_real"] = np.real(arr)
            out[f"{col}_imag"] = np.imag(arr)
        else:
            out[col] = vals

    return out


def write_dataframe(df: pd.DataFrame, path: str | Path, *, expand_complex: bool = True) -> Path:
    """Write a dataframe to CSV and return the resolved output path."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    output = expand_complex_columns(df) if expand_complex else df
    output.to_csv(path, index=False)

    return path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a resolved path."""

    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path