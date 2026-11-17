"""High-level pipelines tying the other modules together."""

import warnings

import pandas as pd

from .data_io import read_upload, get_target
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
      fc_window, fc_offset, pseudocount, shrink, n_cpus, random_state).

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
    shrink=True, n_cpus=None, random_state=0
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

    Returns:
    - result (pd.DataFrame): one row per gene; columns 'Name', the requested metrics (some with
      extra '_p' / '_padj' columns), the fold change columns ('log2FoldChange', and with a design
      file the other DESeq2 columns), 'Cluster' (if clustering ran) and, with an annotation file,
      'Description' and 'Description_Score'. Genes for which a quantity cannot be computed
      (e.g. a constant series, or too few counts for DESeq2) have NaN there.

    Raises:
    - ValueError: unknown metric names; ``n_components`` other than 1 (when CCA is computed); the gene and metabolite tables do not have the same
      samples / time points in the same order; duplicated gene names.
    - KeyError: the target is not in the metabolite table.
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

    # Read data
    gene = read_upload(gene_file)
    metabo = read_upload(metabo_file)

    if gene.index.has_duplicates:
        dup = list(gene.index[gene.index.duplicated()].unique()[:5])
        raise ValueError(f"Gene names must be unique; duplicated names in the gene table: {dup}")

    if design_file is not None:
        design = read_upload(design_file)
        if aggregation_func is None:
            aggregation_func = repeat_aggregation_mean
        # Process data
        processed_gene = aggregation_func(gene, design)
        processed_metabo = aggregation_func(metabo, design)
        _check_aligned(processed_gene.columns, processed_metabo.columns, 'time points (groups)')
        t = get_target(target, processed_metabo)
        # Fold change: DESeq2 on the raw counts of the peak / comparison groups
        counts, design_sub = df_for_fc(processed_metabo, target, gene, design,
                                       window=fc_window, offset=fc_offset)
        fc = fc_comp(counts, design_sub, contrast=design_sub.attrs.get('contrast'),
                     shrink=shrink, n_cpus=n_cpus)
    else:
        _check_aligned(gene.columns, metabo.columns, 'columns (time points)')
        processed_gene = gene
        processed_metabo = metabo
        t = get_target(target, metabo)
        noontide = find_noontide(metabo, target, window=fc_window, offset=fc_offset)
        fc = no_repeat_fc(gene, noontide, pseudocount=pseudocount)

    # Compute corr
    tables = []
    if 'Granger' in metrics:
        tables.append(compute_granger(processed_gene, t, maxlag=lag))
    if 'CCM' in metrics:
        tables.append(compute_ccm(processed_gene, t, E=E, tau=tau))
    if 'CCA' in metrics:
        tables.append(compute_cca(processed_gene, t, n_components=n_components))
    if 'DTW' in metrics:
        tables.append(compute_dtw(processed_gene, t))
    if 'CCF' in metrics:
        tables.append(compute_ccf(processed_gene, t, lag=lag))
    if 'Spearman' in metrics:
        tables.append(compute_spearman(processed_gene, t))
    if 'Pearson' in metrics:
        tables.append(compute_pearson(processed_gene, t))
    tables.append(_name_frame(fc))

    # Clustering on the aggregated time series
    if cluster_count:
        clustering_result = ts_clustering(processed_gene, cluster_count, random_state=random_state)
        tables.append(_name_frame(clustering_result, 'Cluster'))

    data = merge_dataframes(tables)

    if annotation_file is not None:
        annotation = read_upload(annotation_file)
        if annotation.index.has_duplicates:
            warnings.warn("The annotation table has duplicated names; only the first row of each "
                          "name is used.", UserWarning, stacklevel=2)
            annotation = annotation[~annotation.index.duplicated(keep='first')]
        # left join: annotation rows of genes that are not in the gene table must not
        # become extra (all-NaN) rows of the result
        data = data.merge(_name_frame(annotation), on='Name', how='left')
        data = add_annotation_score(data)  # Add a new column for annotation score

    return data
