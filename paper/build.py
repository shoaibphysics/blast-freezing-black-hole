#!/usr/bin/env python3
"""Build both submission PDFs using only files in this directory.

Refresh embedded references from references.bib before compiling. Requires
Python 3, BibTeX, latexmk and a TeX installation with the standard packages.
"""

from pathlib import Path
import os
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
ENV = dict(os.environ, LC_ALL="C", LANG="C")


def run(command, log_name):
    with (BUILD / log_name).open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed; see build/{log_name}")


def embed_references(stem):
    path = ROOT / f"{stem}.tex"
    source = path.read_text()
    before, rest = source.split("% BEGIN REFERENCES\n", 1)
    _, after = rest.split("% END REFERENCES", 1)
    active = re.sub(r"(?<!\\)%[^\n]*", "", before)
    keys = dict.fromkeys(key.strip() for group in re.findall(
        r"\\cite(?:p|t|alp|author|year)?(?:\[[^]]*\]){0,2}\{([^}]+)\}", active
    ) for key in group.split(","))
    job = f"{stem}-references"
    (BUILD / f"{job}.aux").write_text(
        "\\relax\n" + "".join(f"\\citation{{{key}}}\n" for key in keys)
        + "\\bibstyle{sn-nature}\n\\bibdata{references}\n"
    )
    run(["bibtex", f"build/{job}"], f"{job}.txt")
    bbl = (BUILD / f"{job}.bbl").read_text().replace("\\bibcommenthead\n", "")
    # Keep the article-specific source self-contained, with no external .bbl.
    updated = before + "% BEGIN REFERENCES\n" + bbl + "% END REFERENCES" + after
    if updated != source:
        path.write_text(updated)


def main():
    BUILD.mkdir(exist_ok=True)
    for stem in ("manuscript", "supplementary"):
        embed_references(stem)
        run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
             "-outdir=build", f"{stem}.tex"], f"{stem}-build.txt")
        log = (BUILD / f"{stem}.log").read_text()
        if re.search(r"undefined|multiply defined|Overfull \\[hv]box", log):
            raise RuntimeError(f"Unresolved references or overflow: build/{stem}.log")
        shutil.copy2(BUILD / f"{stem}.pdf", ROOT / f"{stem}.pdf")
        print(f"Built {stem}.pdf", flush=True)


if __name__ == "__main__":
    main()
