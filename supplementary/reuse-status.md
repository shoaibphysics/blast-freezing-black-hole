# Reuse and artwork provenance

The main manuscript, scientific supplement, bibliography, supplied paper figures,
and the three Figure 2 component images in `code/scripts/figures/` use
[CC BY 4.0](../LICENSES/CC-BY-4.0.txt). Code, skills, datasets, environment files,
and supporting repository documentation use [MIT](../LICENSES/MIT.txt).
The copyright holders are Shoaib Akhtar and Xiao-Liang Qi. See the
[root license scope](../LICENSE) for exact component boundaries.
Third-party material retains its own copyright and license; neither author
license replaces those terms.

## Artwork

The author confirmed that the setup drawing `paper/fig1.png` and the Penrose
diagrams `code/scripts/figures/fig3b.png` and `code/scripts/figures/fig3d.png` are original work
that the authors have the right to publish. The supplied image assets are included;
separate editable drawing sources were not supplied. These are manually prepared
artwork, not computationally regenerated figures.

The formation map `code/scripts/figures/fig3c_formation_gate_color_map.png` is a numerical
asset; its use in the composite and its direct regeneration have been checked.

## Publisher LaTeX support

The public package does not bundle `sn-jnl.cls` or `sn-nature.bst`. The explicit
setup command in [paper/README.md](../paper/README.md) obtains the two exact
publisher-owned files from the official December 2024 Springer Nature template.
The downloader verifies the archive and individual file SHA-256 hashes and
refuses to overwrite different existing files. Downloaded copies are ignored
by Git and are needed only for rebuilding the PDFs.

The files retain their original notices: `sn-jnl.cls` identifies LPPL 1.3c-or-later
terms, and `sn-nature.bst` identifies LPPL version 1-or-later terms. They are not
covered by the authors' CC BY 4.0 or MIT grants. The class also carries a LaTeX-base distribution
condition; obtaining the files directly from the publisher avoids bundling them
in this publication repository.

Sources: [official Springer Nature template](https://www.springernature.com/gp/authors/campaigns/latex-author-support),
[LPPL 1.3c](https://www.latex-project.org/lppl/lppl-1-3c/).
