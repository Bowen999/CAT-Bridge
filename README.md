# CAT Bridge

CAT Bridge (Compounds And Transcripts Bridge) ranks genes by how closely their expression time course follows that of a target metabolite, using longitudinal transcriptomics and metabolomics data. It is also available as a web application at [catbridge.work](https://catbridge.work).

Version 2.0.0. If you used 1.0, see [Upgrading from 1.0](#upgrading-from-10).

## Installation

```bash
pip install catbridge
```

Python 3.9 or newer. `pandas<3` is required (pydeseq2, used for the fold change with replicates, does not yet work with pandas 3). The AI assistant helper `catbridge.llm` needs the optional extra `pip install "catbridge[llm]"`.

## Quick start

```python
from catbridge import catbridge as cat

folder = "path/to/demo/pepper/"          # gene.tsv, metabolite.tsv, design.tsv, annotation.tsv

data = cat.compute_corr(
    folder + "gene.tsv",
    folder + "metabolite.tsv",
    folder + "design.tsv",               # None if there are no replicates
    folder + "annotation.tsv",           # None if there is no annotation
    target="Capsaicin",                  # a row name of the metabolite table
    cluster_count=None,                  # number of clusters, or None to skip clustering
    metrics=["Pearson", "CCM"],          # default: all seven metrics
)
print(data.attrs["fc_contrast"])         # the two time points of the fold change: (end, start)

ranked = cat.compute_score(data, "CCM")  # best gene first; the index is the rank
print(ranked.head(10))
```

`compute_corr` returns one row per gene. `pipeline(...)` runs `compute_corr` and `compute_score` in one call. Further options: `lag`, `E`, `tau` (see the docstrings), `pseudocount`, `shrink`, `n_cpus`, `random_state`, `verbose=True`.

## Input files

All files are tables whose first column holds the row names. Format by extension: `.csv` (comma), `.tsv`/`.txt` (tab), `.xls`, `.xlsx`; compressed `.gz` also works.

- **gene**: rows = genes (first column, e.g. `Name`), columns = samples (with a design file) or time points (without). **Time order = column order**: the time points are used in the order of the columns, never sorted by name. With a design file, DESeq2 is used for the fold change and needs **raw integer counts** (not TPM/FPKM). Without a design file any non-negative expression values are accepted.
- **metabolite**: rows = metabolites (first column, e.g. `Name`), same columns as the gene table.
- **design** (optional, needed for replicates): one row per sample; first column = sample name (the column names of the two tables above), column `group` = time point. Replicates share a group; each compared group needs at least 2 samples for DESeq2.
- **annotation** (optional): first column = gene name (as in the gene table), column `Description` = free text. Descriptions that contain enzyme words (names ending in `-ase`, `enzyme`, `synthase`) add a small bonus to the score.

## What is computed

**Score.** Metrics between each gene and the target series (Granger causality, CCM, CCA, DTW, CCF, Spearman, Pearson) are computed on the time series (replicates averaged). `compute_score(data, col)` scales the chosen metric (so that a better value is always larger; for Granger p-values and DTW distances this means -log10(p) and -distance), the fold change, and the annotation score to 0-1 and combines them as `Score = sqrt(norm_metric^2 + norm_fc^2 + Description_Score^2)`, rescaled to 0-1. A missing value of a component counts as 0.

**Fold change.** `log2FoldChange` is defined over one interval: the step between two adjacent time points in which the target metabolite rises fastest (largest increase of the group mean; the earliest step wins a tie; steps with missing values are skipped). It is log2(gene at the later time point / gene at the earlier one): with a design file by DESeq2 with apeGLM shrinkage, without one as `log2((x_end + pseudocount) / (x_start + pseudocount))`. The two time points are in `data.attrs["fc_contrast"]`. If the target never rises, a warning is issued and the fold change is NaN for all genes.

## Upgrading from 1.0

Results from 2.0 cannot be reproduced digit for digit with 1.0. Several algorithm errors were fixed and the fold change now uses the steepest-rise interval of the target (1.0: the time point of the highest target value against the next one). The arguments `fc_window` / `fc_offset` (and `window` / `offset` of `find_noontide`, `df_for_fc`, `compute_fc`) were removed. See [CHANGELOG.md](https://github.com/Bowen999/CAT-Bridge/blob/main/CHANGELOG.md) for all changes.

## Contact

Developed and maintained by Bowen Yang (by8@ualberta.ca, by172@georgetown.edu). Please reach out with problems, comments or suggestions.

## License

MIT (see `LICENSE`).
