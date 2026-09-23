# APP publication validation

Date: 2026-09-23 UTC. Target protocol: APP 1.0.0.

**Local package validation: passed within the scope and limitations below.**
This report covers the curated package as an independent repository. The intended
first version is `1.0.0` (tag `v1.0.0`); version metadata is not evidence of a public
release. Public tagged-release verification has not been performed. Final author
approval, an exact release commit/tree, and the tagged-release manifest remain
separate release steps.

## Package and scientific ground truth

The package contains 141 files. The canonical Nature-format main paper is
`paper/manuscript.tex` and its PDF; the scientific supplement is
`paper/supplementary.tex` and its PDF. Both remain together in `paper/`.
The six final figure assets are the main manuscript's direct LaTeX inputs;
three component images in `code/scripts/figures/` support composite reproduction.
Root `supplementary/` contains secondary validation/provenance context.

The package includes maintained scientific modules, the figure/data entry points,
a notebook, the selected 50-snapshot dataset, pinned environment requirements,
and the bulk-reconstruction skill. It excludes alternative manuscripts, diff PDFs,
private conversations/reviews, the separate follow-up project, development Git
history, installed environments, and generated reproduction outputs.

The [figure guide](../code/figure-reproduction/README.md) is the canonical map
for all five main figures, both supplementary diagrams, commands, parameters,
expected signatures, evidence levels, and limitations. There are no numbered tables.
The [data guide](../data/README.md) identifies the bundled run and provenance.

## Reproduction evidence

The full isolated run used Python 3.12.14 on macOS 27.0 arm64, CPU, with one
OpenBLAS/OMP thread. All 58 pinned dependencies installed in a fresh environment
from the local uv cache. Other platforms and network-only package installation
have not been validated. See [environment setup](../environment/README.md).

| Check from the package root | Recorded result |
|---|---|
| `code/figure-reproduction/reproduce_figures.py` | Passed; 15 PNG outputs matched the earlier source-snapshot reproduction pixel-for-pixel |
| Figure 2 and Figure 4 composite comparison | Both generated PDF composites matched supplied composites at 100 dpi |
| `code/figure-reproduction/check_geometry.py` | Passed; direct replot of the saved formation map matched exactly |
| `code/figure-reproduction/check_claims.py` | Passed; actual cached numerical observations below |
| `code/figure-reproduction/fetch_tex_support.py` | Live publisher download passed archive and individual-file checksums |
| `code/figure-reproduction/build_paper.py` | Main paper and supplement rebuilt to 13 and 30 pages |
| `python -m unittest discover -s code/tests -v` | 15/15 passed, including data allowlisting and a fresh two-time solver check |
| `skills/bulk-reconstruction/scripts/test_bulk_reconstruct.py` | 8/8 passed |
| `code/figure-reproduction/run_notebook.py` | 8 nonempty code cells executed; no caught figure-generation failures |

Commands use `.venv/bin/python`. Tests ran in an isolated copy; data-allowlist tests
used a newly initialized local Git index without development history. A sandbox
local-port restriction initially blocked Jupyter, and a sandbox network restriction
initially blocked publisher setup; approved reruns succeeded. Neither failure was
silently treated as a successful check.

The current preparation refresh changes documentation and Python docstrings only.
After removing docstrings, the executable ASTs of all 44 Python files are identical
to the fully checked package. Parameters, function signatures, control flow,
filenames, numerical inputs, and outputs are unchanged. The full numerical suite
was therefore not repeated for this refresh. A fresh reader session did rerun the
cached claim checker from an isolated copy using the tested pinned environment;
its evidence is recorded in [paper-agent-test.md](paper-agent-test.md).

## Paper build and references

Both documents compile without undefined/multiply-defined references or overfull
boxes. Underfull-vbox notices remain. All 45 main and 253 supplementary labels
resolve; the used citation keys exist in the bibliography and embedded reference
lists. All six external figure inputs are present. The two supplementary diagrams
are defined in TikZ within the supplement.

The earlier author-requested equation-layout changes affected only five main-paper
LaTeX blocks: equations (1), (7), (8), (31), and (32) use one line each; equation
(15) uses two. Normalized mathematical content, label numbers, and all source text
outside those blocks were preserved. All 13 main-paper pages were visually checked.
The paper's automatic pagination, including the Methods start, is intentional.

The current main PDF and its isolated rebuild have identical extracted text.
The unchanged supplied supplement was built with an older TeX environment; a current
rebuild has glyph-position/extraction-order differences, equal non-whitespace
character counts, and the same 30 pages. All 43 rebuilt pages were reviewed in
layout overviews without obvious clipping or overlap. Byte-identical PDF rebuilds
across TeX versions are not claimed. Supplied PDFs were not replaced during this refresh.

Two publisher-owned support files are deliberately excluded. The approved setup
command retrieves the exact `sn-jnl.cls` and `sn-nature.bst` versions, verifies
hashes, preserves differing existing files, and leaves them Git-ignored. Earlier
negative checks rejected wrong content; a repeat setup reused verified files.
The isolated build wrapper is required to preserve supplied PDFs. Its documented
`paper/build.py` helper is intentional code beside the paper, not a duplicate
figure builder. See the [paper guide](../paper/README.md).

## Numerical observations and evidence limits

The cached audit verified 51 hashes, all 50 finite payloads, all 150 saved convergence
flags, zero beta-grid discrepancy, and zero antisymmetry residual. Figure 4(a)'s
maximum MI is 1.3837480740067698 nats, or 99.81632421046106% of 2 ln 2, on 1055 finite
cells of a 50-by-50 grid. This is the displayed final-time-25 map with kernel-norm
cutoff 50, not an MI time sweep or proof of exact saturation.

The second-Renyi entropy-density maximum is 0.18260681636205683. Pre-evaporation
samples are zero; sampled later minima at t=14 and 22 are 0.002668085958073857 and
0.008122848280091466. Both analytic presets give cross-time rank 4 at relative
singular-value threshold 1e-10. Additional size and sampled bound checks are
specified in the figure guide. Sampling is not a proof of all formal derivations.

The pre-staging full 50-time finite-p sweep converged all 150 saddles. Its tensors
passed rtol=1e-7, atol=1e-8 against the bundled inputs. The same strict comparison
on derived grids gave `runs-but-differs`: maximum Iq difference 1.22e-7,
classical-MI difference 4.74e-7, and kernel-norm difference 3.33e-4. The largest
kernel difference was beyond the plotting cutoff. Masks were unchanged and the
Figure 4 PNGs were pixel-identical. This approximately 20-minute sweep was not
repeated after documentation-only changes; tolerances were not relaxed to hide drift.

NumPy `slogdet` warnings occurred despite finite checked results; the earlier
independent LU audit agreed to about 1e-14. Existing docstring-escape and nonfatal
font-cache warnings remain documented. The author retains the manuscript's
physics wording: a gate-block norm is called a reflection probability; a literal
no-color-change statement coexists with negative signed polarization; and exact
entropy-vanishing language is broader than the sampled finite-p evidence. These
interpretation differences are not certified by successful figure reproduction.
They do not by themselves measure boundary transmission or refute the separately
stated supplementary correlator bound.

Figure 1 and the two Penrose panels are `manual-only` artwork supplied by the
authors; assembling them is not a computational derivation. Rights were confirmed,
but separate editable drawing sources are not included. No scientific claim,
calculation, tolerance, or dataset was silently changed.

## Documentation, structure, and preservation

README, AGENTS, and CLAUDE agree on author framing, second-Renyi entropy, ground
truth, canonical guides, and citation. All local Markdown links resolve. The
S4/Figure 3 overview points to the size-response functions in `large_p/evap_syk.py`;
auxiliary circuit routines are clearly distinguished. Historical internal-note
pointers were replaced by descriptive docstrings. The older optional output helper
is explicitly documented as outside the manuscript workflow; its behavior was preserved.

The manuscript entry points use independent figure presets and generated output
locations. All 50 snapshots are needed for the entropy curve; the final snapshot
also supplies the MI plot. There are no duplicate nonempty files, duplicate figure
wrappers, private parent-checkout dependencies, or missing supported-workflow inputs.
Reusable auxiliary helpers and one uncited bibliography entry remain intentionally;
this is a curated package, not an assertion that every helper is used by fixed presets.

The refresh preserves every paper, PDF, bibliography, figure, dataset, environment,
and skill file byte-for-byte. All 180 original development files are unchanged
from the start of the refresh. Relative to the initial pre-packaging snapshot,
only the two previously approved main-manuscript layout files differ; the other
178 remain byte-identical. The approved root README/AGENTS/CLAUDE and license terms
were reviewed and retained rather than rewritten unnecessarily.

A bounded whole-package scan found no private home/temp paths, Overleaf project
URLs, or credential-shaped tokens. Notebook outputs/private metadata remain absent.
Author contact information, acknowledgments, and bibliographic citations are
intentional paper content. Bibliographic-link liveness and every conceivable
sensitive string were not exhaustively verified.

## Licensing and release status

Paper, scientific supplement, bibliography, final figure artwork, and three named
component images use CC BY 4.0. Code, skills, data, environment files, and supporting
documentation use MIT. The root LICENSE defines the boundaries; LICENSES holds
complete texts. The paper build helper is MIT; the images under code remain CC BY.
Publisher support retains its separate LPPL terms. See [reuse-status.md](reuse-status.md).

The selected GitHub repository is https://github.com/shoaibphysics/blast-freezing-black-hole.
This report verifies local preparation only. A public tagged APP release requires
author approval of final contents and an exact commit/tree, followed by the release
and its verifiable `APP_PUBLICATION.json`. No public release or manifest was created
by this refresh.
