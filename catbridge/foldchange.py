"""Fold-change computation: steepest rise of the target, no-replicate FC, DESeq2-based FC."""

import inspect
import warnings

import pandas as pd
import numpy as np

from .data_io import read_upload, _check_numeric_table, _check_design, _check_samples_in_design
from .preprocess import repeat_aggregation_mean

__all__ = [
    'find_steepest_rise',
    'find_noontide',
    'df_for_fc',
    'no_repeat_fc',
    'fc_comp',
    'compute_fc',
]


# ********* Steepest rise ************
def find_steepest_rise(df, row_name):
    """
    Find the interval in which a row (the target metabolite) rises fastest.

    ``df.loc[row_name]`` is the time series of the target (columns in time order; with a
    design file the replicate-aggregated group means). Among all pairs of adjacent columns
    ``i -> i+1`` the one with the largest increase ``x[i+1] - x[i]`` is chosen. Ties go to
    the earliest step. Pairs where either value is NaN are skipped. This is the one and only
    definition of the fold change in CAT Bridge 2.0: the fold change of a gene is
    log2(expression at the end column / expression at the start column) of this step.

    Parameters:
        df (pandas DataFrame): rows = metabolites, columns = time points in time order.
        row_name (str): The name of the target row.

    Returns:
        pandas.Index: ``[end_column, start_column]`` (the column at ``i+1`` first, then the
        column at ``i``), i.e. ``[numerator, reference]`` of the fold change. ``None`` (with a
        ``UserWarning``) if there is no rising step: no adjacent pair without NaN, or the
        series never increases.

    Raises:
        KeyError: ``row_name`` is not in ``df.index``.
        ValueError: the row name is not unique or ``df`` has fewer than two columns.
    """
    if row_name not in df.index:
        raise KeyError(f"Row '{row_name}' is not in the index of the dataframe.")
    row = df.loc[row_name]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"Row name '{row_name}' is not unique in the index.")
    if len(row) < 2:
        raise ValueError(f"The row '{row_name}' needs at least two columns (time points) to "
                         "find a rising step.")

    # Convert to numeric and handle errors by replacing non-numeric values with NaN
    values = pd.to_numeric(row, errors='coerce').to_numpy(dtype=float)
    steps = np.diff(values)                      # NaN where either neighbour is NaN
    valid = np.isfinite(steps)
    if not valid.any():
        reason = "there is no pair of adjacent time points without missing values"
    elif not (steps[valid] > 0).any():
        reason = "its values never increase from one time point to the next"
    else:
        i = int(np.argmax(np.where(valid, steps, -np.inf)))   # first maximum = earliest step
        return df.columns[[i + 1, i]]
    warnings.warn(
        f"No rising step found for '{row_name}': {reason}. The fold change is defined as the "
        "change over the interval in which the target rises fastest, so log2FoldChange is set "
        "to NaN for all genes (it counts as 0 in the score).",
        UserWarning, stacklevel=2)
    return None


def find_noontide(df, row_name):
    """
    Deprecated name of :func:`find_steepest_rise` (kept so that old imports work).

    The meaning changed in CAT Bridge 2.0: 1.x returned ``[peak_column, neighbour_column]``
    (the column with the highest value and the column after / before it, controlled by
    ``window`` and ``offset``, which were removed). It now returns
    ``pd.Index([end_column, start_column])`` of the interval in which ``row_name`` rises
    fastest, or ``None`` with a warning if it never rises. See :func:`find_steepest_rise`.
    """
    return find_steepest_rise(df, row_name)


def _nan_fc(index):
    """A fold-change table with NaN for every gene (used when the target has no rising step)."""
    out = pd.DataFrame({'log2FoldChange': np.full(len(index), np.nan)}, index=index.copy())
    out.index.name = 'Name'
    return out


# ********* df for fc ************
def df_for_fc(df1, target, df2, design):
    """
    Generate the design matrix and matrix for computing the FC score.
    
    Parameters:
        df1 (pandas DataFrame): The DataFrame has been aggregated (processed_metabo).
        target (str): The name of the row to be detected (Capsaicin).
        df2 (pandas DataFrame): The DataFrame for fc computing (gene).
        design (pandas DataFrame): study design, the sample and group information.
    
    Returns:
        volcano_df (pandas DataFrame): Modified matrix for computing FC score (a copy).
        volcano_design (pandas DataFrame): Modified design matrix (a copy; rows in the order of
            the columns of ``df2``). Its ``attrs['contrast']`` is ``(end_group, start_group)``,
            the interval in which the target rises fastest (``find_steepest_rise``); ``fc_comp``
            uses it so that log2FoldChange = log2(end / start).

    Raises:
        ValueError: the target has no rising step (a warning is issued as well), or a group of
            the interval is not in ``design['group']``.
    """
    rise = find_steepest_rise(df1, target)
    if rise is None:
        raise ValueError(f"The target '{target}' has no rising step, so there is no interval to "
                         "compare (see the warning above).")
    end_group, start_group = rise[0], rise[1]

    volcano_design = design[design['group'].isin(rise)].copy()
    # samples in the order of the data columns (time order), not the row order of design
    not_in_data = [c for c in volcano_design.index if c not in df2.columns]
    if not_in_data:
        raise KeyError(f"Samples of design not found in the columns of the gene table: {not_in_data}")
    volcano_design = volcano_design.loc[[c for c in df2.columns if c in volcano_design.index]]
    present = set(volcano_design['group'])
    absent = [g for g in (end_group, start_group) if g not in present]
    if absent:
        raise ValueError(f"Group(s) {absent} of the rising step are not in design['group'].")

    volcano_df = df2[list(volcano_design.index)].copy()
    volcano_design.attrs = {'contrast': (end_group, start_group)}

    return volcano_df, volcano_design


def no_repeat_fc(df, noontide, pseudocount=1.0):
    """
    Compute the FC score for each gene. This function is used for Study Design that has no biological repeats.

    log2FoldChange = log2((end + pseudocount) / (start + pseudocount)).
    A pseudocount keeps zero expression finite (0 vs 0 gives 0). NaN in either column gives NaN.
    
    Parameters:
        df (pandas DataFrame): The DataFrame for fc computing (gene).
        noontide (list): The two column names ``[end, start]`` (from ``find_steepest_rise``).
        pseudocount (float): Added to both columns before the ratio; must be > 0.
    
    Returns:
        new_df (pandas DataFrame): index = genes (named 'Name'), single column 'log2FoldChange'.
        The input is not modified.

    Raises:
        ValueError: pseudocount <= 0, or negative values in the two columns.
    """
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError(f"pseudocount must be a positive finite number, got {pseudocount!r}.")

    peak = pd.to_numeric(df[noontide[0]], errors='coerce').astype(float)
    other = pd.to_numeric(df[noontide[1]], errors='coerce').astype(float)
    if (peak < 0).any() or (other < 0).any():
        neg = list(df.index[((peak < 0) | (other < 0)).to_numpy()][:5])
        raise ValueError("Negative values found in the columns used for fold change "
                         f"({list(noontide)}), e.g. genes {neg}; expression values must be >= 0 "
                         "(the fold change expects expression that is not log-transformed).")

    lfc = np.log2((peak + pseudocount) / (other + pseudocount))

    new_df = pd.DataFrame({'log2FoldChange': lfc.to_numpy()}, index=df.index.copy())
    new_df.index.name = 'Name'
    return new_df


def _resolve_contrast(contrast, metadata, sample_order=None):
    """Return (end, start). Without a contrast, the first two groups in order of
    appearance are used; "order" is the order of ``sample_order`` (the columns of the count
    table = the time order of the data file), not the row order of ``metadata``."""
    if sample_order is not None and all(c in metadata.index for c in sample_order):
        groups_in_order = list(pd.unique(metadata.loc[list(sample_order), 'group']))
    else:
        groups_in_order = list(pd.unique(metadata['group']))
    if contrast is None:
        contrast = metadata.attrs.get('contrast') if hasattr(metadata, 'attrs') else None
    if contrast is None:
        if len(groups_in_order) < 2:
            raise ValueError("metadata['group'] needs at least two groups to compute a fold change.")
        contrast = (groups_in_order[0], groups_in_order[1])
        warnings.warn(
            f"No contrast given; using the first two groups in order of appearance in the "
            f"data columns: "
            f"log2({contrast[0]} / {contrast[1]}).", UserWarning, stacklevel=3)
    contrast = tuple(contrast)
    if len(contrast) != 2 or contrast[0] == contrast[1]:
        raise ValueError(f"contrast must be two different group names (end, start), got {contrast!r}.")
    missing = [g for g in contrast if g not in groups_in_order]
    if missing:
        raise ValueError(f"contrast group(s) {missing} not found in metadata['group'] {groups_in_order}.")
    return contrast


def _accepts(callable_, name):
    try:
        return name in inspect.signature(callable_).parameters
    except (TypeError, ValueError):
        return False


def _get_size_factors(dds, samples):
    """DESeq2 size factors (median-of-ratios) of ``dds`` as an array in the order of ``samples``,
    or None if they cannot be found.  pydeseq2 >= 0.5 keeps them in ``dds.obs['size_factors']``,
    0.4.x in ``dds.obsm['size_factors']``."""
    for attr in ('obs', 'obsm'):
        try:
            sf = getattr(dds, attr)['size_factors']
        except Exception:
            continue
        try:
            if isinstance(sf, (pd.Series, pd.DataFrame)):
                sf = sf.squeeze(axis=1) if isinstance(sf, pd.DataFrame) else sf
                if isinstance(sf, pd.Series) and set(samples) <= set(sf.index):
                    sf = sf.loc[list(samples)]
            sf = np.asarray(sf, dtype=float).reshape(-1)
        except Exception:
            continue
        if len(sf) == len(samples) and np.all(np.isfinite(sf)) and np.all(sf > 0):
            return sf
    return None


def _direction_agrees(counts, is_peak, lfc, size_factors, min_abs=1.0, min_mean=10.0):
    """Fraction of clear-cut genes whose estimated LFC sign matches the raw group-mean ratio.

    counts: samples x genes (ints); is_peak: boolean array over samples; lfc: Series (genes);
    size_factors: the DESeq2 (median-of-ratios) size factors, one per sample. They are used
    instead of total counts, which are misleading when a few genes take a large share of the
    library (composition bias). Returns None if size_factors is None or there are fewer than
    5 clear-cut genes."""
    if size_factors is None:
        return None
    c = counts.to_numpy(dtype=float)
    size = np.asarray(size_factors, dtype=float)
    norm = c / size[:, None]
    m1 = norm[is_peak].mean(axis=0)
    m0 = norm[~is_peak].mean(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = np.log2((m1 + 1.0) / (m0 + 1.0))
    est = lfc.reindex(counts.columns).to_numpy(dtype=float)
    sel = (np.abs(raw) >= min_abs) & ((m1 + m0) / 2 >= min_mean) & np.isfinite(est) & (est != 0)
    if sel.sum() < 5:
        return None
    return float(np.mean(np.sign(raw[sel]) == np.sign(est[sel])))


def _restore_gene_labels(result_index, original_index):
    """pydeseq2 turns gene names into str. Map them back to the labels of ``counts_df`` (e.g. int
    gene ids), via str(label) -> label. If two labels have the same str() the mapping is
    ambiguous: the labels are left as str and a warning is issued."""
    str_to_label = {}
    ambiguous = False
    for label in original_index:
        key = str(label)
        if key in str_to_label and str_to_label[key] != label:
            ambiguous = True
        str_to_label.setdefault(key, label)
    if ambiguous:
        warnings.warn(
            "Gene labels of counts_df that are different but have the same string form (e.g. 1 and "
            "'1'): the labels of the fold-change result are returned as strings.",
            UserWarning, stacklevel=3)
        return result_index
    if all(isinstance(i, str) for i in original_index):
        return result_index
    if not all(str(i) in str_to_label for i in result_index):
        return result_index
    return pd.Index([str_to_label[str(i)] for i in result_index])


def _import_pydeseq2():
    try:
        from pydeseq2.dds import DeseqDataSet  # lazy import (heavy optional dependency)
        from pydeseq2.ds import DeseqStats
    except ImportError as e:
        raise ImportError(
            "The fold change with a design file (replicates) is computed with DESeq2 (pydeseq2), "
            f"which could not be imported ({type(e).__name__}: {e}). Install it with "
            "`pip install pydeseq2`.") from e
    return DeseqDataSet, DeseqStats


def _raise_if_incompatible(error):
    """pydeseq2 0.5.x fails inside DeseqDataSet with pandas >= 3 (a TypeError / UFuncTypeError
    raised by pandas). Turn that into an error that says so; other errors are left alone."""
    try:
        pandas_major = int(pd.__version__.split('.')[0])
    except ValueError:
        return
    if pandas_major < 3:
        return
    try:
        from importlib.metadata import version
        pydeseq2_version = version('pydeseq2')
    except Exception:
        pydeseq2_version = 'unknown'
    if not (pydeseq2_version.startswith(('0.4', '0.5')) or pydeseq2_version == 'unknown'):
        return
    raise RuntimeError(
        f"DESeq2 (pydeseq2 {pydeseq2_version}) failed with pandas {pd.__version__}: "
        f"{type(error).__name__}: {str(error)[:200]}. pydeseq2 0.4/0.5 is known not to work with "
        "pandas 3 "
        "(catbridge requires pandas<3 for this reason). Fix: install pandas<3 in this environment "
        "(`pip install \"pandas<3\"`), or a pydeseq2 release that supports pandas 3. The fold "
        "change was NOT computed; no other method was substituted.") from error


def fc_comp(counts_df, metadata, contrast=None, shrink=True, n_cpus=None):
    """
    Perform a DESeq2 (pydeseq2) analysis given raw counts and sample metadata.

    log2FoldChange is log2(contrast[0] / contrast[1]), i.e. end group / start group of the
    interval in which the target rises fastest, explicitly (the start group is the reference
    level), independent of the alphabetical order of the group names.
    
    Parameters:
    - counts_df (DataFrame): genes x samples RAW COUNTS (DESeq2 needs counts, not TPM/FPKM;
      a warning is issued for clearly non-integer input; values are rounded).
    - metadata (DataFrame): index = sample names, column 'group'. Not modified.
    - contrast (tuple): (end_group, start_group). If None, ``metadata.attrs['contrast']``
      (set by df_for_fc) is used; otherwise the first two groups in order of appearance are
      used, with a warning.
    - shrink (bool): apply pydeseq2 apeGLM LFC shrinkage (``lfc_shrink``) to the log2FoldChange
      coefficient of this contrast (p-values are unchanged). If shrinkage cannot be done with
      a verified direction, falls back to unshrunk LFC with a warning.
    - n_cpus (int or None): passed to pydeseq2 (None: pydeseq2 default).
    
    Returns:
    - DataFrame: pydeseq2 results (baseMean, log2FoldChange, lfcSE, stat, pvalue, padj),
      with ``attrs['lfc_shrunk']`` = True if the apeGLM shrinkage was applied to
      log2FoldChange, False if ``shrink=False`` or the shrinkage failed and the unshrunk value
      is returned (a warning is issued in the latter case),
      index named 'Name' (the gene labels of ``counts_df``, with their original type -- e.g. int
      gene ids stay int; pydeseq2 itself converts them to str). Genes with total count < 10 are
      dropped.

    The result's sign is set by the explicit contrast. As an extra check the estimated sign is
    compared with the raw group-mean ratio normalised with DESeq2's own size factors; a
    disagreement (< 50% of clear-cut genes) only issues a warning, it never raises.
    """
    DeseqDataSet, DeseqStats = _import_pydeseq2()

    peak, ref = _resolve_contrast(contrast, metadata, sample_order=list(counts_df.columns))

    # Align metadata with the samples (columns of counts_df); only the two contrast groups are used
    missing = [c for c in counts_df.columns if c not in metadata.index]
    if missing:
        raise KeyError(f"Samples of counts_df not found in metadata.index: {missing}")
    group = metadata.loc[list(counts_df.columns), 'group']
    keep = group.isin([peak, ref]).to_numpy()
    samples = list(counts_df.columns[keep])
    group = group[keep]
    for g in (peak, ref):
        if int((group == g).sum()) < 2:
            raise ValueError(
                f"Group '{g}' has fewer than 2 samples; DESeq2 needs biological replicates "
                "(use no_repeat_fc for designs without replicates).")

    counts = counts_df[samples]
    values = counts.to_numpy(dtype=float)
    if np.isnan(values).any():
        rows = np.isnan(values).any(axis=1)
        raise ValueError(
            f"counts_df contains NaN: {int(rows.sum())} gene(s) have missing counts in the compared "
            f"samples {samples} (e.g. {list(counts.index[rows][:5])}). DESeq2 needs complete raw "
            "counts: remove these genes or fill in their counts (0 only if they are true zeros).")
    if (values < 0).any():
        raise ValueError("counts_df contains negative values; DESeq2 needs raw counts.")
    if np.any(np.abs(values - np.round(values)) > 1e-3):
        warnings.warn(
            "counts_df contains non-integer values (TPM/FPKM/normalised data?). DESeq2 needs raw "
            "counts; values will be rounded, and the fold changes/p-values are not valid for "
            "normalised expression.", UserWarning, stacklevel=2)

    # Filter genes based on counts (total >= 10), then round to integers (genes columns)
    counts_t = counts.T
    genes_to_keep = counts_t.columns[counts_t.sum(axis=0) >= 10]
    counts_t = counts_t[genes_to_keep].round().astype(int)

    # Private metadata copy (the caller's metadata is never touched)
    cond = group.astype(str).to_numpy()
    peak_s, ref_s = str(peak), str(ref)
    new_api = _accepts(DeseqDataSet.__init__, 'design')
    if new_api:
        # reference level = first category of a Categorical
        condition = pd.Categorical(cond, categories=[ref_s, peak_s])
    else:
        condition = cond
    meta = pd.DataFrame({'condition': condition}, index=pd.Index(samples))

    dds_kwargs = dict(counts=counts_t, metadata=meta, refit_cooks=True)
    if _accepts(DeseqDataSet.__init__, 'n_cpus'):
        dds_kwargs['n_cpus'] = n_cpus
    if _accepts(DeseqDataSet.__init__, 'quiet'):
        dds_kwargs['quiet'] = True
    if new_api:
        dds_kwargs['design'] = '~condition'
    else:  # pydeseq2 0.4.x
        dds_kwargs['design_factors'] = 'condition'
        dds_kwargs['ref_level'] = ['condition', ref_s]
    stats_kwargs = dict(contrast=['condition', peak_s, ref_s])
    if _accepts(DeseqStats.__init__, 'n_cpus'):
        stats_kwargs['n_cpus'] = n_cpus
    if _accepts(DeseqStats.__init__, 'quiet'):
        stats_kwargs['quiet'] = True
    try:
        dds = DeseqDataSet(**dds_kwargs)
        dds.deseq2()
        stat_res = DeseqStats(dds, **stats_kwargs)
        stat_res.summary(lfc_null=0.1, alt_hypothesis="greaterAbs")
    except Exception as e:
        _raise_if_incompatible(e)
        raise

    # Direction self-check (insurance only: the direction is fixed by the explicit contrast).
    # Uses DESeq2's own size factors; never raises, only warns. Skipped if they are unavailable.
    is_peak = (group == peak).to_numpy()
    size_factors = _get_size_factors(dds, samples)
    agree = _direction_agrees(counts_t, is_peak, stat_res.results_df['log2FoldChange'],
                              size_factors)
    if agree is not None and agree < 0.5:
        warnings.warn(
            f"log2FoldChange direction check: only {agree:.0%} of clear-cut genes have the "
            f"sign of log2({peak}/{ref}) (normalised with the DESeq2 size factors). The direction "
            "is set by the explicit contrast, so this is only a warning; please check the result.",
            UserWarning, stacklevel=2)

    shrunk = False
    if shrink:
        unshrunk = stat_res.results_df.copy()
        coeff_candidates = [c for c in stat_res.LFC.columns if c != 'Intercept']
        expected = {f"condition[T.{peak_s}]", f"condition_{peak_s}_vs_{ref_s}"}
        reason = None
        if len(coeff_candidates) != 1 or coeff_candidates[0] not in expected:
            reason = (f"cannot identify the {peak_s}-vs-{ref_s} coefficient "
                      f"(LFC columns: {list(stat_res.LFC.columns)})")
        else:
            try:
                stat_res.lfc_shrink(coeff=coeff_candidates[0])
                agree = _direction_agrees(counts_t, is_peak,
                                          stat_res.results_df['log2FoldChange'], size_factors)
                if agree is not None and agree < 0.5:
                    reason = "shrunk LFC direction does not match the contrast"
            except Exception as e:  # pydeseq2 version differences / convergence problems
                reason = f"lfc_shrink failed ({type(e).__name__}: {e})"
        if reason is not None:
            warnings.warn(f"LFC shrinkage skipped, returning unshrunk log2FoldChange: {reason}.",
                          UserWarning, stacklevel=2)
            stat_res.results_df = unshrunk
        else:
            shrunk = True

    result = stat_res.results_df.copy()
    result.index = _restore_gene_labels(result.index, counts_df.index)
    result.index.name = 'Name'
    result.attrs['lfc_shrunk'] = shrunk
    return result


def compute_fc(
    gene_file, metabo_file, design_file=None, aggregation_func=None, target=None,
    shrink=True, n_cpus=None, pseudocount=1.0,
):
    """
    Read the files and compute the fold change of every gene over the interval in which
    ``target`` rises fastest (see ``find_steepest_rise``): log2(expression at the end time
    point / expression at the start time point).

    With a design file: replicates are aggregated by ``aggregation_func`` (default
    ``repeat_aggregation_mean``) to find the interval, and DESeq2 (``fc_comp``) is run on the
    raw gene table. Without a design file: ``no_repeat_fc`` is used. If the target never rises
    a warning is issued and ``log2FoldChange`` is NaN for all genes.

    Returns:
        DataFrame with a 'log2FoldChange' column (index named 'Name'); ``attrs['fc_contrast']``
        is ``(end, start)`` (None if there is no rising step).
    """
    if target is None:
        raise ValueError("target (the name of the target metabolite) is required.")

    # Read data
    gene = read_upload(gene_file)
    metabo = read_upload(metabo_file)
    _check_numeric_table(gene, 'gene', gene_file, 'gene')
    _check_numeric_table(metabo, 'metabolite', metabo_file, 'metabolite')
    gene.columns = gene.columns.map(str)
    metabo.columns = metabo.columns.map(str)

    if design_file is not None:
        design = _check_design(read_upload(design_file), design_file)
        _check_samples_in_design(list(gene.columns), design, 'gene')
        _check_samples_in_design(list(metabo.columns), design, 'metabolite')
        if aggregation_func is None:
            aggregation_func = repeat_aggregation_mean
        processed_metabo = aggregation_func(metabo, design)
        rise = find_steepest_rise(processed_metabo, target)
        if rise is None:
            fc = _nan_fc(gene.index)
        else:
            counts_df, design_sub = df_for_fc(processed_metabo, target, gene, design)
            fc = fc_comp(counts_df, design_sub, shrink=shrink, n_cpus=n_cpus)
    else:
        rise = find_steepest_rise(metabo, target)
        if rise is None:
            fc = _nan_fc(gene.index)
        else:
            fc = no_repeat_fc(gene, rise, pseudocount=pseudocount)

    fc.attrs['fc_contrast'] = None if rise is None else (rise[0], rise[1])
    return fc
