# Blast freezing a black hole

**Shoaib Akhtar and Xiao-Liang Qi**

This repository accompanies *Blast freezing a black hole*. The paper introduces
a solvable model of rapid black-hole evaporation built from coupled
Sachdev–Ye–Kitaev (SYK) systems. In the stated large-N and large-p limits,
analytic boundary correlators determine the reconstructed bulk geometry.
Operator size and Rényi-2 mutual information track information carried by an
infalling excitation.
Second Rényi entropy tracks the oscillating entanglement between the evaporating
system and the bath.

The repository contains the main paper, scientific supplement, code, data,
and instructions for reproducing the figures. It also includes paper-agent
instructions and a bulk-reconstruction skill, organized using the Agentic
Publication Protocol (APP).

## Read the paper

- [Main paper PDF](paper/manuscript.pdf) · [LaTeX source](paper/manuscript.tex)
- [Scientific supplement PDF](paper/supplementary.pdf) · [LaTeX source](paper/supplementary.tex)
- [Paper build instructions](paper/README.md)

The scientific supplement stays beside the main paper and is part of its ground
truth. Files in the root `supplementary/` folder provide secondary APP context.

## Use the package

Start with [environment setup](environment/README.md), then use the
[figure-reproduction guide](code/figure-reproduction/README.md). It maps all five
main figures and both supplementary diagrams to inputs, scripts, outputs,
expected numerical signatures, and limitations. [Data documentation](data/README.md)
explains the bundled 50-snapshot dataset. The
[notebook](code/notebooks/paper_figures.ipynb) uses the same maintained code.

After setup, this quick command checks the primary cached numerical result:

```sh
.venv/bin/python code/figure-reproduction/check_claims.py
```

Generated outputs are separate from supplied paper assets. The full solver run
is optional and more expensive; consult the canonical guide before starting it.
The [validation report](supplementary/validation-report.md) records the checks
performed, numerical comparisons, and known limitations.

## Read with an AI coding agent

Open this directory in an AI coding agent that reads [AGENTS.md](AGENTS.md).
Claude-compatible tooling can use [CLAUDE.md](CLAUDE.md).
Try asking:

- What assumptions make this evaporation model solvable?
- Check how close Figure 4's mutual information comes to its stated maximum.
- Which files reproduce Figure 2, and which panels are supplied artwork?
- What does the cross-time rank bound establish, and what does it not establish?

The bundled [bulk-reconstruction skill](skills/bulk-reconstruction/SKILL.md)
provides general-flavor input validation and reconstruction helpers.

## Reuse and citation

The main paper, scientific supplement, and figure artwork are licensed under
[CC BY 4.0](LICENSES/CC-BY-4.0.txt). Code, skills, datasets, and supporting
documentation are licensed under [MIT](LICENSES/MIT.txt).
The [license scope](LICENSE) identifies which terms apply to each component.
Publisher LaTeX support files are obtained separately when rebuilding PDFs
and retain their own terms; see
[reuse and artwork information](supplementary/reuse-status.md).

```bibtex
@unpublished{akhtar_qi_blast_freezing,
  title = {Blast freezing a black hole},
  author = {Akhtar, Shoaib and Qi, Xiao-Liang},
  year = {2026},
  note = {Nature-format manuscript with scientific supplement}
}
```
