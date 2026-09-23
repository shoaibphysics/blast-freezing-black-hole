"""Build both Nature documents in a separate copy; never update supplied PDFs."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper"
OUT = ROOT / "code/figure-reproduction/generated/paper-build"


def main():
    missing = [name for name in ("sn-jnl.cls", "sn-nature.bst")
               if not (SOURCE / name).is_file()]
    if missing:
        raise SystemExit(
            "Publisher TeX support files are missing. First run:\n"
            ".venv/bin/python code/figure-reproduction/fetch_tex_support.py"
        )
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in SOURCE.rglob("*") if p.is_file()}
    target = OUT / "paper"
    if target.exists():
        raise SystemExit("Build output already exists. Preserve or move generated/paper-build before a new build.")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, target, ignore=shutil.ignore_patterns("build", "__pycache__"))
    subprocess.run([sys.executable, "build.py"], cwd=target, check=True)
    for path, digest in protected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Protected input changed: {path.relative_to(ROOT)}")
    print("Built PDFs in code/figure-reproduction/generated/paper-build/paper/")


if __name__ == "__main__":
    main()
