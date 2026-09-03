# Suggested patch (manual merge only)

## related_work.tex
- Insert contents of `ustc_related_work.tex` after the "Deep phase picking" subsection (or as a new subsection).
- Add `\bibliography` entry from `ustc_reference.bib` into `paper/references.bib` **only if** the merge is approved.
- Do **not** change contribution bullets.

## discussion.tex
- Append complementarity + diminishing-returns paragraphs from `ustc_discussion.tex`.
- Keep Stage 3 numbers unchanged.
- Do not cite hierarchical residual as a test result.

## limitations.tex
- Append bullets from `ustc_limitations.tex`.

## Explicit non-goals of this patch
- No new Stage 3/4 tables
- No GNN resurrection
- No USTC models as INSTANCE baselines
- No claim that hierarchical residual is in the main method
