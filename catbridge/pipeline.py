"""High-level pipelines tying the other modules together."""

import sys
import time
import warnings

import pandas as pd

from .data_io import (read_upload, get_target, _check_numeric_table, _check_design,
                      _check_samples_in_design, _check_target)
from .preprocess import merge_dataframes, repeat_aggregation_mean
from .metrics import (
    compute_granger,
    compute_ccm,
    compute_cca,
    compute_dtw,
    compute_ccf,
    compute_spearman,
    compute_pearson,
    ts_clustering,
)
from .foldchange import find_noontide, df_for_fc, no_repeat_fc, fc_comp
from .scoring import add_annotation_score, compute_score

__all__ = [
    'pipeline',
    'compute_corr',
]

#: Metric names accepted by ``compute_corr(metrics=...)`` (also the order of the output columns).
ALL_METRICS = ['Granger', 'CCM', 'CCA', 'DTW', 'CCF', 'Spearman', 'Pearson']


def _check_metrics(metrics):
    """Return the validated metric names in canonical order (ValueError on unknown / empty)."""
    if metrics is None:
        return list(ALL_METRICS)
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = list(metrics)
    unknown = [m for m in metrics if m not in ALL_METRICS]
    if unknown:
        raise ValueError(f"Unknown metric(s) {unknown}; valid names are {ALL_METRICS}.")
    if not metrics:
        raise ValueError(f"metrics is empty; choose from {ALL_METRICS} (or None for all).")
    return [m for m in ALL_METRICS if m in metrics]


def _check_aligned(gene_cols, metabo_cols, what):
    """Gene and metabolite columns (samples / time points) must match by name AND order."""
    gene_cols, metabo_cols = list(gene_cols), list(metabo_cols)
    if gene_cols == metabo_cols:
        return
    only_g = [c for c in gene_cols if c not in set(metabo_cols)]
    only_m = [c for c in metabo_cols if c not in set(gene_cols)]
    if only_g or only_m or len(gene_cols) != len(metabo_cols):
        raise ValueError(
            f"The {what} of the gene table and the metabolite table do not match. "
            f"Only in gene table: {only_g}; only in metabolite table: {only_m}. "
            f"gene: {gene_cols}; metabolite: {metabo_cols}."
        )
    diff = [(i, g, m) for i, (g, m) in enumerate(zip(gene_cols, metabo_cols)) if g != m]
    raise ValueError(
        f"The {what} of the gene table and the metabolite table have the same names but a "
        f"different order (position, gene, metabolite): {diff}. Reorder them so that they match."
    )


class _Progress:
    """Step messages on stderr when ``verbose`` is true."""

    def __init__(self, verbose):
        self.verbose = bool(verbose)
        self.t0 = time.perf_counter()

    def __call__(self, msg):
        if self.verbose:
            print(f"[catbridge {time.perf_counter() - self.t0:6.1f}s] {msg}", file=sys.stderr,
                  flush=True)


def _merge_annotation(data, annotation):
    """Left join of the annotation on Name (rows of genes not in ``data`` are ignored).

    If one table has numeric names (e.g. integer gene ids) and the other text, the names are
    matched by their string form. A warning is issued if no gene has an annotation."""
    ann = _name_frame(annotation)
    key = data['Name']
    if pd.api.types.is_numeric_dtype(key) != pd.api.types.is_numeric_dtype(ann['Name']):
        str_to_label = dict(zip(key.map(str), key))
        if len(str_to_label) == len(key):
            ann = ann.assign(Name=[str_to_label.get(str(v), None) for v in ann['Name']])
            ann = ann[ann['Name'].notna()]
            ann = ann.assign(Name=ann['Name'].astype(key.dtype))
    out = data.merge(ann, on='Name', how='left')
    if len(annotation) and not key.isin(set(ann['Name'])).any():
        warnings.warn(
            f"None of the {len(annotation)} annotation names matches a gene name (e.g. annotation "
            f"{list(annotation.index[:2])} vs. genes {list(key[:2])}); every gene gets the "
            "'missing description' score 0.1. The first column of the annotation file must hold "
            "the gene names of the gene table.", UserWarning, stacklevel=3)
    return out


def _name_frame(obj, value_name=None):
    """Turn a Series / DataFrame indexed by gene name into a table with a 'Name' column."""
    if isinstance(obj, pd.Series):
        return pd.DataFrame({'Name': obj.index.to_numpy(), value_name or obj.name: obj.to_numpy()})
    return obj.rename_axis('Name').reset_index()


def pipeline(gene_file, metabo_file, design_file, annotation_file, target, cluster_count,
             max_lag=1, aggregation_func=None, *, score_col='Granger', **kwargs):
    """
    One-step analysis: ``compute_corr`` -> ``compute_score`` -> ranked table.

    This is a thin wrapper kept for backward compatibility. Note that the scoring changed:
    the old implementation scored ``MinMax(Granger p-value) + MinMax(log2FoldChange)`` on the
    RAW Granger p-value (so a larger p-value ranked higher -- the direction was wrong) after
    filling missing p-values with 0 ("most significant"). The score is now computed by
    :func:`catbridge.scoring.compute_score` (smaller p ranks higher via -log10(p); missing
    values count as "no evidence"), the same way as ``compute_score(compute_corr(...), col)``.
    The old implementation also ignored ``cluster_count``; it is now passed on, so clustering
    (which needs ``tslearn``) runs whenever ``cluster_count`` is a positive number -- pass
    ``None`` or ``0`` to skip it.

    Parameters:
    - gene_file (str): The filename of the gene count data.
    - metabo_file (str): The filename of the metabolome data.
    - design_file (str): The filename of the experimental design data (can be None).
    - annotation_file (str): The filename of the gene annotation data (can be None).
    - target (str): The target metabolite for the analysis.
    - cluster_count (int): The number of clusters for time series clustering (None / 0: no clustering).
    - max_lag (int): Lag for the Granger causality test and the CCF (default is 1).
    - aggregation_func (function): The function to be used for replicate aggregation
      (default with a design file: ``repeat_aggregation_mean``).
    - score_col (str): The metric used for ranking, one of
      'Granger', 'CCM', 'CCA', 'DTW', 'CCF', 'Spearman', 'Pearson' (default 'Granger').
    - **kwargs: any other keyword-only argument of :func:`compute_corr`
      (E, tau, n_components are positional-or-keyword there, so pass them by name here;
      fc_window, fc_offset, pseudocount, shrink, n_cpus, random_state, dtw_method, verbose).

    Returns:
    - result (pd.DataFrame): the output of ``compute_score(data, score_col)``: columns
      'Name', score_col, 'log2FoldChange', ('Cluster'), 'Score', (f'{score_col}_padj'),
      sorted best first, indexed by the rank (index name 'Rank', starting at 1).
    """
    for reserved in ('lag', 'metrics'):
        if reserved in kwargs:
            raise TypeError(
                f"pipeline() sets '{reserved}' itself ("
                + ("use max_lag" if reserved == 'lag' else "use score_col")
                + "); do not pass it."
            )
    # the score column plus Pearson (which the old design-file pipeline also computed)
    metrics = _check_metrics([score_col, 'Pearson'])
    data = compute_corr(
        gene_file, metabo_file, design_file, annotation_file,
        target, cluster_count, aggregation_func=aggregation_func,
        lag=max_lag, metrics=metrics, **kwargs
    )
    result = compute_score(data, score_col)
    result.index.name = 'Rank'
    return result


def compute_corr(
    gene_file, metabo_file, design_file, annotation_file,
    target, cluster_count, aggregation_func=None,
    lag=1, E=3, tau=1, n_components=1,
    *, metrics=None, fc_window='after', fc_offset=1, pseudocount=1.0,
    shrink=True, n_cpus=None, random_state=0, dtw_method='fastdtw', verbose=False
):
    """
    Computes correlation metrics between gene expression and metabolite data.

    This function reads in gene expression data, metabolite data, optional experimental design,
    and annotation data from specified files. It then processes the data, computes various correlation
    and similarity metrics, fold change and (optionally) clustering, and optionally annotates the
    results, returning one DataFrame with one row per gene.

    Parameters:
    - gene_file (str): Path to the file containing gene count data.
    - metabo_file (str): Path to the file containing metabolome data.
    - design_file (str): Optional; Path to the file containing experimental design data. Default is None.
    - annotation_file (str): Optional; Path to the file containing gene annotation data. Default is None.
    - target (str): The target metabolite for analysis.
    - cluster_count (int): Number of clusters for time series clustering; None or 0 skips the
      clustering (the result then has no 'Cluster' column).
    - aggregation_func (callable, optional): Function for replicate aggregation, called as
      ``aggregation_func(df, design)``. Only used with a design file; default there is
      ``repeat_aggregation_mean``.
    - lag (int, optional): Lag for the Granger causality test (maxlag) AND for the CCF. Default is 1.
      The Granger test needs more than 3*lag+1 time points; if that is not met for any gene, the
      Granger column is all NaN and a UserWarning gives the largest usable lag.
    - E (int, optional): The embedding dimension for Convergent Cross Mapping (CCM). Default is 3. Imply the complexity of the system.
    - tau (int, optional): The delay for Convergent Cross Mapping (CCM). Default is 1.
    - n_components (int, optional): Number of components for Canonical Correlation Analysis (CCA). Only 1 is
      accepted (one gene vs. one metabolite has a single canonical component); anything else raises
      ValueError. The parameter is kept for backward compatibility of the signature.

    Keyword-only parameters:
    - metrics (list of str, optional): which metrics to compute, from
      'Granger', 'CCM', 'CCA', 'DTW', 'CCF', 'Spearman', 'Pearson'. None (default) = all.
      Unknown names raise ValueError.
    - fc_window ('after' or 'before'), fc_offset (int >= 1): which time point the peak time point
      of the target is compared with, see ``find_noontide``. Default: the time point right after.
    - pseudocount (float): pseudocount of the fold change when there is no design file
      (``no_repeat_fc``).
    - shrink (bool), n_cpus (int or None): passed to ``fc_comp`` (DESeq2) when there is a design file.
    - random_state (int): seed of the time series clustering.
    - dtw_method ('fastdtw' or 'exact'): 'fastdtw' (default) is the fastdtw approximation used
      since 1.0; 'exact' is the exact DTW distance, see ``compute_dtw``.
    - verbose (bool): print each step (with the elapsed time) to stderr. Default False.

    Returns:
    - result (pd.DataFrame): one row per gene; columns 'Name', the requested metrics (some with
      extra '_p' / '_padj' columns), the fold change columns ('log2FoldChange', and with a design
      file the other DESeq2 columns), 'Cluster' (if clustering ran) and, with an annotation file,
      'Description' and 'Description_Score'. Genes for which a quantity cannot be computed
      (e.g. a constant series, or too few counts for DESeq2) have NaN there.

    Raises:
    - ValueError: unknown metric names; ``n_components`` other than 1 (when CCA is computed); the gene and metabolite tables do not have the same
      samples / time points in the same order; duplicated gene names; empty tables, rows
      without a name, a missing name column, non-numeric or infinite values; a design table
      without a 'group' column, with duplicated samples or samples without a group, or whose
      samples do not match the data columns; a target that is duplicated, constant or has
      missing time points. Each message names the file / rows / columns concerned.
    - KeyError: the target is not in the metabolite table (similar names are suggested).
    - Various exceptions may be raised due to file reading, data processing, or computation errors.

    Notes:
    - The aggregation_func parameter should be a function that takes two arguments: a DataFrame
      containing data and a DataFrame containing design information, and returns a processed DataFrame
      whose columns are the time points in time order.
    - If the design_file is provided, the data will be aggregated based on the provided aggregation_func.
      All time-series metrics and the clustering are computed on the aggregated tables; the fold change
      is computed by DESeq2 on the raw (unaggregated) gene counts of the two groups that
      ``find_noontide`` picks from the aggregated target, as log2(peak group / comparison group).
      Without a design file the tables are used as they are (columns = time points).
      The time order is always the column order of the data files (with a design file: the order in
      which the groups first appear in the data columns), never the row order of the design file.
    - The function employs a variety of metrics such as Granger causality, Convergent Cross Mapping (CCM),
      Canonical Correlation Analysis (CCA), Dynamic Time Warping (DTW), Cross-Correlation Function (CCF),
      Spearman correlation, and Pearson correlation to analyze the relationships between gene expression
      and metabolite data.
    - If the annotation_file is provided, the function will merge the annotation data with the results,
      and compute an annotation score.
    """
    metrics = _check_metrics(metrics)
    if dtw_method not in ('fastdtw', 'exact'):
        raise ValueError(f"dtw_method must be 'fastdtw' or 'exact', got {dtw_method!r}.")
    say = _Progress(verbose)

    # Read data
    say(f"reading the gene table {gene_file!s}")
    gene = read_upload(gene_file)
    say(f"reading the metabolite table {metabo_file!s}")
    metabo = read_upload(metabo_file)
    _check_numeric_table(gene, 'gene', gene_file, 'gene')
    _check_numeric_table(metabo, 'metabolite', metabo_file, 'metabolite')
    # sample / time point names are compared as text (an Excel file may give 0, a csv '0')
    gene.columns = gene.columns.map(str)
    metabo.columns = metabo.columns.map(str)
    say(f"{gene.shape[0]} genes x {gene.shape[1]} columns, {metabo.shape[0]} metabolites")

    design = None
    if design_file is not None:
        design = _check_design(read_upload(design_file), design_file)
    # a table without a name column: its first sample became the row names
    for tbl, other, what in ((gene, metabo, 'gene'), (metabo, gene, 'metabolite')):
        known = set(design.index) if design is not None else set(other.columns)
        if tbl.index.name is not None and str(tbl.index.name) in known:
            raise ValueError(
                f"The first column of the {what} table is named '{tbl.index.name}', which is a "
                f"{'sample of the design table' if design is not None else 'time point of the other table'}"
                f": the table seems to have no name column. Add a first column with the {what} names.")

    if gene.index.has_duplicates:
        dup = list(gene.index[gene.index.duplicated()].unique()[:5])
        raise ValueError(f"Gene names must be unique; duplicated names in the gene table: {dup}")

    if design is not None:
        for tbl, what in ((gene, 'gene'), (metabo, 'metabolite')):
            _check_samples_in_design(list(tbl.columns), design, what)
        if aggregation_func is None:
            aggregation_func = repeat_aggregation_mean
        # Process data
        processed_gene = aggregation_func(gene, design)
        processed_metabo = aggregation_func(metabo, design)
        _check_aligned(processed_gene.columns, processed_metabo.columns, 'time points (groups)')
        t = get_target(target, processed_metabo)
        _check_target(t, target, processed_metabo.columns)
        # Fold change: DESeq2 on the raw counts of the peak / comparison groups
        counts, design_sub = df_for_fc(processed_metabo, target, gene, design,
                                       window=fc_window, offset=fc_offset)
        say(f"fold change: DESeq2 on {counts.shape[0]} genes x {counts.shape[1]} samples "
            f"(groups {design_sub.attrs.get('contrast')}); this is the slowest step")
        fc = fc_comp(counts, design_sub, contrast=design_sub.attrs.get('contrast'),
                     shrink=shrink, n_cpus=n_cpus)
    else:
        _check_aligned(gene.columns, metabo.columns, 'columns (time points)')
        processed_gene = gene
        processed_metabo = metabo
        t = get_target(target, metabo)
        _check_target(t, target, metabo.columns)
        noontide = find_noontide(metabo, target, window=fc_window, offset=fc_offset)
        fc = no_repeat_fc(gene, noontide, pseudocount=pseudocount)

    n_time = processed_gene.shape[1]
    if n_time <= 3:
        warnings.warn(
            f"Only {n_time} time points: Granger and CCM cannot be computed and the correlation-type "
            "metrics are not informative with so few points (with 2 points every Pearson r is +1 "
            "or -1).", UserWarning, stacklevel=2)

    # Compute corr
    tables = []
    steps = [
        ('Granger', lambda: compute_granger(processed_gene, t, maxlag=lag)),
        ('CCM', lambda: compute_ccm(processed_gene, t, E=E, tau=tau)),
        ('CCA', lambda: compute_cca(processed_gene, t, n_components=n_components)),
        ('DTW', lambda: compute_dtw(processed_gene, t, method=dtw_method)),
        ('CCF', lambda: compute_ccf(processed_gene, t, lag=lag)),
        ('Spearman', lambda: compute_spearman(processed_gene, t)),
        ('Pearson', lambda: compute_pearson(processed_gene, t)),
    ]
    for name, step in steps:
        if name in metrics:
            say(f"{name}")
            tables.append(step())
    tables.append(_name_frame(fc))

    # Clustering on the aggregated time series
    if cluster_count:
        say(f"time series clustering ({cluster_count} clusters)")
        clustering_result = ts_clustering(processed_gene, cluster_count, random_state=random_state)
        tables.append(_name_frame(clustering_result, 'Cluster'))

    data = merge_dataframes(tables)

    if annotation_file is not None:
        say(f"annotation {annotation_file!s}")
        annotation = read_upload(annotation_file)
        if 'Description' not in annotation.columns:
            alt = [c for c in annotation.columns if str(c).strip().lower() == 'description']
            if len(alt) != 1:
                raise ValueError(
                    f"The annotation table ('{annotation_file}') needs a column named 'Description' "
                    f"(the gene description used for the annotation score); its columns are "
                    f"{list(annotation.columns)[:10]}. The first column must hold the gene names.")
            warnings.warn(f"annotation table: using column {alt[0]!r} as 'Description'.",
                          UserWarning, stacklevel=2)
            annotation = annotation.rename(columns={alt[0]: 'Description'})
        if annotation.index.has_duplicates:
            warnings.warn("The annotation table has duplicated names; only the first row of each "
                          "name is used.", UserWarning, stacklevel=2)
            annotation = annotation[~annotation.index.duplicated(keep='first')]
        # left join: annotation rows of genes that are not in the gene table must not
        # become extra (all-NaN) rows of the result
        data = _merge_annotation(data, annotation)
        data = add_annotation_score(data)  # Add a new column for annotation score

    say("done")
    return data
