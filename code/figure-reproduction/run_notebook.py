"""Execute the maintained notebook with this interpreter and audit caught failures."""
from pathlib import Path
import json
import os
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/figure-reproduction/generated/notebook"


def main():
    # Set thread limits before the kernel imports numerical libraries.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import sys
    import nbformat
    from nbclient import NotebookClient
    OUT.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(ROOT / "code/notebooks/paper_figures.ipynb", as_version=4)
    with tempfile.TemporaryDirectory(prefix="paper-notebook-") as tmp:
        kernel = Path(tmp) / "kernels/paper-reproduction"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Paper reproduction", "language": "python",
        }))
        os.environ["JUPYTER_PATH"] = tmp
        os.environ["JUPYTER_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        NotebookClient(nb, timeout=600, kernel_name="paper-reproduction",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    failures = []
    for i, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            for key in ("text/plain", "text/markdown"):
                text += output.get("data", {}).get(key, "")
            if "was not generated" in text or "skipping Fig." in text:
                failures.append({"cell": i, "message": text})
    nbformat.write(nb, OUT / "paper_figures.executed.ipynb")
    summary = {"executed_code_cells": sum(c.cell_type == "code" and bool(c.source.strip()) for c in nb.cells),
               "caught_generation_failures": failures}
    (OUT / "notebook-result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError("The notebook caught figure-generation failures; inspect the result JSON.")


if __name__ == "__main__":
    main()
