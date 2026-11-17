# Changelog

## 1.1.0

### Fixed

- CCF direction was reversed and negative lags were wrong -> `compute_ccf` is now `corr(gene[t], target[t+lag])`: `lag > 0` means the gene leads the metabolite, `lag < 0` the reverse.
- Fold change without replicates gave -52 for zero/negative ratios and turned inf into 0 -> `no_repeat_fc` uses `log2((peak + pseudocount) / (comparison + pseudocount))`; zero vs zero gives 0.
- DESeq2 fold change direction depended on the alphabetical order of the group names -> `fc_comp` uses an explicit contrast, `log2(peak group / comparison group)`.
- `repeat_aggregation_mean` re-sorted the time points alphabetically -> columns keep the order in which the groups first appear in the data columns (the time order is that of the data file, not of the design file's rows; `repeat_aggregation_max` and the no-design path behave the same way).
- Peak in the last column: 1.0 silently switched to the second-highest column -> `find_noontide` now compares with the column on the other side of the same peak and warns (error if neither side exists).
- `compute_score` modified the input DataFrame in place (it overwrote `Granger` with 1-p) and scored DTW as "larger is better" -> input is never modified; DTW distance is "smaller is better"; `Cluster` is optional.
- Annotation score treated words such as disease, release, base, phase as enzymes and missed words followed by punctuation (`ligase,`) -> words are split on punctuation and a built-in exclusion list (`ASE_EXCLUDE`) is applied to the generic `ase` suffix; the input is not modified.
- `top_important_features` ignored how much variance each principal component explains -> loadings are weighted by explained variance ratio; errors are raised instead of printed and returning `None`.
- `pipeline()` crashed with a design file and ranked large Granger p-values first -> it now calls `compute_corr` + `compute_score` (small p ranks higher) and passes `cluster_count` on.
- Metric functions silently dropped rows (constant series in Granger, NaN rows in CCM) and CCM was returned sorted -> every metric returns one row per gene in input order, with NaN where it cannot be computed.
- `scaling()` returned nothing -> it returns the scaled data (labels kept for DataFrames).
- `scale_df` lost the row index -> index is kept.
- `ts_clustering` added a `Cluster` column to the caller's DataFrame -> it returns a Series and leaves the input untouched.
- `compute_corr` ignored `n_components` (CCA) and always used lag 1 for CCF -> both are passed through.
- `plot_result` modified the caller's DataFrame -> it works on a copy.
- `read_upload` printed a message and then failed with an unrelated error for unsupported file types -> raises `ValueError`. `get_target` for a missing target -> raises `KeyError` with the name.

### Changed

- `compute_score` Granger columns are now raw p-values (no longer 1-p); scoring uses `-log10(p)` (clipped to [1e-300, 1]). The same applies to `Granger_reverse`.
- `compute_reverse_granger` returns columns `Name`, `Granger_reverse`, `Granger_reverse_padj` (1.0 used the column name `Granger`) and the raw p-value instead of 1-p.
- New p-value columns (Benjamini-Hochberg over non-NaN values): `Granger_padj`, `Granger_reverse_padj`, `Spearman_p`/`Spearman_padj`, `Pearson_p`/`Pearson_padj`; `CCM_p` (no adjusted value). `compute_score` carries `<metric>_padj` into its output when present; ranking uses the raw value.
- `compute_score(df, col, scaling='minmax')`: new `scaling` argument. `'minmax'` (default) is plain min-max scaling. `'robust'` clips each component to its 1%-99% quantiles first; this resists extreme values but ties the top 1% of every component, so use it only if extreme values remain in your data.
- DESeq2 fold change applies apeGLM LFC shrinkage by default (`shrink=True` in `compute_fc` / `compute_corr`; `shrink=False` for the unshrunk value). Falls back to unshrunk with a warning if shrinkage cannot be verified. `n_cpus` is now a parameter (default: pydeseq2 default; 1.0 hard-coded 8). Needs >= 2 samples per compared group; raw counts are required (a warning is issued for non-integer values).
- With a design file, clustering is run on the replicate-aggregated time series (1.0 used the raw replicate columns). `compute_corr`: `cluster_count=None` or `0` skips clustering.
- Annotation is merged with a left join on `Name`: annotation rows for genes not in the gene table no longer create extra rows (duplicated annotation names keep the first row, with a warning).
- `add_annotation_score`: matching is case-insensitive; a plural such as `kinases` also matches after removing the trailing `s`; words that match only the generic `ase` suffix need at least 5 letters (and must not be in `ASE_EXCLUDE`); an empty or blank description counts as missing and scores 0.1 (1.0: 0, only NaN gave 0.1).
- `no_repeat_fc`: negative input values raise `ValueError`; `NaN` in either column stays `NaN` (1.0 filled it with 0).
- `fc_comp(counts_df, metadata, contrast=None, shrink=True, n_cpus=None)`: new `contrast` argument `(peak_group, comparison_group)`; without it `metadata.attrs['contrast']` (set by `df_for_fc`) is used.
- `pipeline()` returns only the columns of `compute_score` (`Name`, the score column, `log2FoldChange`, optional `Cluster`, `Score`, optional `<col>_padj`); 1.0 returned the whole merged table (including `Description`, `Description_Score` and, with a design file, the Pearson column). Use `compute_corr` for the full table.
- Column alignment check in `compute_corr`: with a design file the aggregated tables are compared by their time points (group names); without a design file the original columns of the gene and metabolite tables are compared (names and order).
- A missing metric or fold change contributes 0 to the score (1.0: NaN score).
- `compute_ccm` returns NaN for all genes and a `UserWarning` when the series is too short (`N - (E-1)*tau < E + 2`) instead of raising.
- `compute_corr` raises `ValueError` if the gene and metabolite tables do not have the same columns in the same order, and for duplicated gene names.
- New parameters: `find_noontide` / `df_for_fc` `window` (`'after'`/`'before'`, default `'after'`) and `offset` (default 1); `compute_corr` `fc_window`, `fc_offset`, `pseudocount` (default 1.0), `shrink`, `n_cpus`, and `compute_fc` `window`, `offset`, `pseudocount`, `shrink`, `n_cpus`; `compute_corr` `metrics` (subset of Granger, CCM, CCA, DTW, CCF, Spearman, Pearson; default all) and `random_state` (default 0); `ts_clustering` `random_state`, `max_iter`; `pipeline` `score_col` (default `'Granger'`) and extra keyword arguments passed to `compute_corr`. `pipeline()` returns the table indexed by `Rank`.
- `plot_result` colours a Granger column by `-log10(p)` and a DTW column by `-DTW` (colour bar label says so), i.e. a darker colour is always stronger evidence (`scoring.METRIC_DIRECTION`).
- `setup.py`: version 1.1.0, `python_requires>=3.9`, `install_requires` now listed. Upper bounds: `pandas<3` (pydeseq2 0.5.2 fails on pandas 3.0, so the design-file fold change does not work there) and `scikit-learn<1.6` (tslearn 0.6.3 fails with a `force_all_finite` error). `openai<1.0` is an optional extra (`catbridge[llm]`), because `catbridge.llm` uses the pre-1.0 OpenAI interface.

#### Behaviour changes after review (second round)

- `compute_dtw`: a constant gene series (including all zeros) now gives `NaN` (before: min-max scaling turned it into zeros and the distance was finite, so a gene without any information could rank first for a spiky target). A constant target gives `NaN` for all genes. `compute_cca` also gives `NaN` for constant series (explicit; no exception is swallowed any more).
- Time order is the column order of the data file: `repeat_aggregation_mean` (and `repeat_aggregation_max`, the no-design path) no longer depend on the row order of the design file, so a design file sorted alphabetically by sample name does not turn the time order into an alphabetical one. `df_for_fc` returns the samples in data-column order, and `fc_comp` without a contrast takes "the first two groups" in data-column order (a contrast is normally given explicitly).
- `compute_cca` / `compute_corr`: `n_components` only accepts 1 (a single gene against a single metabolite has one canonical component); any other value raises `ValueError` (before: sklearn's error was swallowed and the whole CCA column was `NaN` without a message). The parameter stays in the signature.
- `compute_granger` / `compute_reverse_granger` (and so `compute_corr`): if the lag is too large for the number of time points the whole column is `NaN`; this now issues a `UserWarning` that names the number of time points, `maxlag` and the largest usable lag (statsmodels needs more than `3*maxlag + 1` time points, i.e. `maxlag <= (n - 2) // 3`).
- `fc_comp`: the direction self-check now normalises with DESeq2's own size factors (median of ratios) instead of total counts, and never raises: a disagreement only issues a `UserWarning` (the direction is fixed by the explicit contrast). Before, a few very highly expressed genes (composition bias) could make the check fail and abort the analysis with `RuntimeError` although the DESeq2 result was correct. If size factors are not available the check is skipped.
- `fc_comp` returns the gene labels with the type they have in `counts_df` (pydeseq2 turns them into strings; integer gene ids made the later merge on `Name` fail). If two labels have the same string form the labels stay strings and a warning is issued.
- `scale_df` fills missing values along the time axis (previous time point of the same gene); before it copied the value of the previous gene (`ffill` along the rows).

### Structure

The single `catbridge.py` is split into `data_io`, `preprocess`, `metrics`, `foldchange`, `scoring`, `plotting`, `llm` and `pipeline`. `catbridge.py` re-exports them, so `from catbridge import catbridge as cat` and `cat.compute_score(...)` keep working.

### Reproducing 1.0 results

Rankings from 1.0 cannot be reproduced digit for digit with 1.1.0 because of the fixes above. To reproduce old results, use version 1.0 (git commit `deebb82`).
