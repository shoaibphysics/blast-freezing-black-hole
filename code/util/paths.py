from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent

DATA_ROOT = Path(os.environ.get("BH_DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()
NUMERICS_DATA_ROOT = DATA_ROOT / "numerics"

FIGURE_INPUT_ROOT = CODE_ROOT / "scripts" / "figures"
FIGURE_ROOT = Path(os.environ.get(
    "BH_FIGURE_ROOT", CODE_ROOT / "figure-reproduction" / "generated" / "figures"
)).expanduser().resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
