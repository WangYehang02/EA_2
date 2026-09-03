# Paper materials (Stage 5)

**Title:** Historical Path-Residual Guided Candidate Re-ranking for Catalog-Assisted S-Phase Repicking

## Build tables/figures from frozen artifacts (no training)

```bash
conda activate PS
cd ~/yehang/Earthquake
bash scripts/reproduce_paper_tables.sh
bash scripts/reproduce_paper_figures.sh
```

## Compile LaTeX

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
# supplement:
cd supplement && latexmk -pdf -interaction=nonstopmode supplement.tex
```

Or: `make -C paper pdf`

## Evidence policy

See `../reports/paper_evidence_ledger.md` and `../reports/paper_claim_audit.md`.

- Main test: Stage 3 test-only 10k
- Development: Stage 2
- Auxiliary underpowered: Stage 4 (9 events)
