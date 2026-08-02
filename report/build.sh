#!/usr/bin/env bash
# Builds report/latex/paper_arxiv.pdf, the single source of truth for the
# report, from report/latex/body.md. Always run from the repo root so that
# metadata.yaml's `bibliography: references/references.bib` (and any other
# repo-root-relative path) resolves the same way regardless of caller cwd --
# invoking pandoc/pdflatex from report/latex/ instead silently breaks citeproc
# and \includegraphics.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pandoc -s \
  --citeproc \
  --metadata-file=report/latex/metadata.yaml \
  --metadata=geometry:'' \
  --shift-heading-level-by=-1 \
  --wrap=none \
  --include-in-header=report/latex/preamble.tex \
  --include-in-header=report/latex/arxiv-include.tex \
  -o report/latex/paper_arxiv.tex \
  report/latex/body.md

# secnumdepth is disabled (arxiv-include.tex) because appendices use letters
# (A, B, ...) rather than continuing the numeric section counter, so the
# "1", "3.4", "D", etc. in headings are literal text pandoc carried over
# from body.md, not a LaTeX counter -- there's no numbering macro inserting
# its usual tab-width gap after them, just body.md's single literal space.
# Widen that gap so the number visually separates from the title. Requires
# --wrap=none above so each \section{...}/\subsection{...} is one line.
perl -pi -e 's/\\((?:sub)?section)\{([0-9A-Za-z]+(?:\.[0-9]+)*)\s/\\$1\{$2\\quad /' report/latex/paper_arxiv.tex

pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report/latex report/latex/paper_arxiv.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report/latex report/latex/paper_arxiv.tex
