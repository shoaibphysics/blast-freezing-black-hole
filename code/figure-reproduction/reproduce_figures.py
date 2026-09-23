"""Regenerate Nature figures in an ignored directory, preserving supplied assets."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/figure-reproduction/generated/figures"


def main():
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run([
        sys.executable, str(ROOT / "code/scripts/generate_paper_figures.py"),
        "--manuscript", "--out-dir", str(OUT),
    ], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
