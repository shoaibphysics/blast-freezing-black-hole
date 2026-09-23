# Paper and scientific supplement

`manuscript.pdf` and `manuscript.tex` are the canonical Nature-format main paper.
`supplementary.pdf` and `supplementary.tex` are the scientific supplement and have
the same scientific standing as the main paper. They are not the optional APP
context in the root `supplementary/` directory. Sources and supplied PDFs are
preserved from the author-approved selection, including their scientific wording.

The six figure assets beside the manuscript are its exact LaTeX inputs.
The three component images needed to regenerate Figure 2 are in
[`code/scripts/figures/`](../code/scripts/figures/README.md); its builders are in
`code/scripts/`. Historical numbers in asset filenames do not change
the current manuscript figure numbers; use the
[figure map](../code/figure-reproduction/README.md).

The paper, scientific supplement, and figure artwork use CC BY 4.0; see
[paper licensing](LICENSE). The build script and this guide use MIT.

## Build without replacing the supplied paper

From the repository root, after [environment setup](../environment/README.md):

```sh
.venv/bin/python code/figure-reproduction/fetch_tex_support.py
.venv/bin/python code/figure-reproduction/build_paper.py
```

The first command needs internet access once to obtain `sn-jnl.cls` and
`sn-nature.bst` from the official Springer Nature template. It checks the archive
and both file hashes, retrieves the exact versions used for this paper, and
refuses to overwrite different existing files. The downloaded files are ignored
by Git and retain their original license notices. They are needed only to rebuild
the PDFs; reading the supplied PDFs and reproducing numerical figures needs no
publisher download. If the publisher archive changes or becomes unavailable,
the checksum check stops rather than substituting another version.

The wrapper copies this directory into the ignored generated area, then builds
the manuscript followed by the supplement. Cross-references use the manuscript's
auxiliary file. Rebuilt PDFs appear in
`code/figure-reproduction/generated/paper-build/paper/`.

Do not run `paper/build.py` directly when preserving the supplied PDFs: that
script refreshes embedded references and replaces PDFs in its own directory.
Bibliography inputs are the bundled `references.bib` and downloaded
`sn-nature.bst`; the downloaded document class is `sn-jnl.cls`.
Their third-party notices are preserved; see
[reuse status](../supplementary/reuse-status.md).
