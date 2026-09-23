from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PYTHON_DIR / "figures"


def ensure_figures_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def figure_path(filename):
    return ensure_figures_dir() / filename
