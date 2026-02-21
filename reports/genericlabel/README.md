# Technical Report: Differentiable GenericLabel Interpolation

This directory contains the LaTeX source for the technical report on the fused GenericLabel grid sampler in FireANTs. Experiment figures and tables are **placeholders** for now.

## Building the report

```bash
cd reports/genericlabel
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or use your preferred LaTeX workflow (e.g. latexmk).

## Experiments (placeholders)

The report describes experiments (correctness scatter, runtime/memory vs L, optional registration loss curve) but does not yet include generated figures or tables. The scripts \texttt{scripts/benchmark\_genericlabel.py} and \texttt{scripts/benchmark\_genericlabel\_registration.py} are placeholders; when implemented and run on a GPU-enabled machine, they will generate:

- \texttt{figures/correctness\_scatter.pdf} — fused vs baseline labels
- \texttt{figures/scaling\_vs\_L.pdf} — runtime and memory vs number of labels
- \texttt{scaling\_vs\_L.txt}, \texttt{scaling\_vs\_volume.txt} — tables
- \texttt{figures/registration\_loss\_curve.pdf} — optional minimal registration loss curve

## Tests

```bash
pytest tests/test_fusedops_gridsample_genericlabel.py -v
```

Requires CUDA and \texttt{fireants\_fused\_ops}.
