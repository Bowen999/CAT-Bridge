"""Similarity / causality metrics between each feature (row) and a target series, plus clustering."""

import warnings

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import ccf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from causal_ccm.causal_ccm import ccm
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import MinMaxScaler
from fastdtw import fastdtw
from scipy.stats import spearmanr
from scipy.stats import pearsonr

__all__ = [
    'compute_granger',
    'compute_reverse_granger',
    'granger_list',
    'compute_ccm',
    'compute_cca',
    'compute_dtw',
    'compute_ccf',
    'compute_spearman',
    'compute_pearson',
    'clustering',
    'ts_clustering',
]


# ************* shared helpers **************************
def _prepare(df, target):
    """
    Validate the inputs and return ``(names, X, y)``.

    ``names`` is the list of row labels (input order), ``X`` a float ndarray of
    shape (n_genes, n_timepoints) and ``y`` a float 1-D ndarray.  Neither ``df``
    nor ``target`` is modified (``X`` / ``y`` are read-only views or copies).

    Raises ValueError if ``len(target) != df.shape[1]``.
    """
    y = np.asarray(target, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"target must be one-dimensional, got shape {y.shape}")
    n_cols = df.shape[1]
    if len(y) != n_cols:
        raise ValueError(
            f"Length of target ({len(y)}) does not match the number of "
            f"time points / columns in df ({n_cols})."
        )
    X = df.to_numpy(dtype=float)
    return list(df.index), X, y


def _is_constant(v):
    return np.ptp(v) == 0 or np.std(v) == 0


def _bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values; NaN entries stay NaN."""
    p = np.asarray(p, dtype=float)
    padj = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        padj[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return padj


def _check_positive_int(value, name):
    if int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


# Exceptions that statsmodels / numpy raise when a test is simply not computable
# (too few observations, singular design, constant column).
_GRANGER_EXPECTED_ERRORS = (ValueError, np.linalg.LinAlgError, InfeasibleTestError)


def _granger_p(x_cause, y_effect, maxlag):
    """
    p-value of the ssr F-test for "x_cause Granger-causes y_effect" at lag
    ``maxlag``.  NaN when the test cannot be run.
    """
    if (np.isnan(x_cause).any() or np.isnan(y_effect).any()
            or _is_constant(x_cause) or _is_constant(y_effect)):
        return np.nan
    # statsmodels convention: the test asks whether column 2 helps predict column 1.
    data = np.column_stack([y_effect, x_cause])
    try:
        with warnings.catch_warnings():
            # `verbose` is deprecated in recent statsmodels, but required to keep
            # older versions from printing.
            warnings.filterwarnings('ignore', message='.*verbose.*', category=FutureWarning)
            result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        return float(result[maxlag][0]['ssr_ftest'][1])
    except _GRANGER_EXPECTED_ERRORS:
        return np.nan


def _granger_max_lag(n_obs):
    """Largest ``maxlag`` that ``statsmodels.grangercausalitytests`` accepts for ``n_obs`` time
    points (it requires ``n_obs > 3 * maxlag + 1`` with the constant term); 0 if none."""
    return max((int(n_obs) - 2) // 3, 0)


def _granger_table(df, target, maxlag, direction, col):
    maxlag = _check_positive_int(maxlag, 'maxlag')
    names, X, y = _prepare(df, target)
    p = np.empty(len(names))
    testable = np.zeros(len(names), dtype=bool)   # data would allow a test (not constant / NaN)
    for i, row in enumerate(X):
        if direction == 'forward':      # gene -> target
            p[i] = _granger_p(row, y, maxlag)
            cause, effect = row, y
        else:                           # target -> gene
            p[i] = _granger_p(y, row, maxlag)
            cause, effect = y, row
        testable[i] = not (np.isnan(cause).any() or np.isnan(effect).any()
                           or _is_constant(cause) or _is_constant(effect))
    if testable.any() and np.isnan(p[testable]).all():
        max_lag = _granger_max_lag(len(y))
        warnings.warn(
            f"{col}: the test could not be computed for any gene: {len(y)} time points with "
            f"maxlag={maxlag}. statsmodels needs more than 3*maxlag+1 observations, so the largest "
            f"usable lag is {max_lag}"
            + (" (no lag is possible with this few time points)" if max_lag == 0 else "")
            + ". Returning NaN for all genes.",
            UserWarning, stacklevel=3,
        )
    return pd.DataFrame({'Name': names, col: p, col + '_padj': _bh_adjust(p)})


# ************* 2.1 Granger **************************
def compute_granger(df, target, maxlag=1):
    """
    Granger causality of each gene (row of ``df``) on the target: gene -> target.

    Parameters:
        df (pandas DataFrame): rows = genes, columns = time points (in order).
        target (list / ndarray / Series): target series, same length as df.columns.
        maxlag (int): lag used for the test (the p-value at exactly this lag is reported).

    Returns:
        DataFrame with columns ``Name``, ``Granger``, ``Granger_padj`` -- one row
        per input gene, in input order.  ``Granger`` is the raw p-value (0..1,
        small = the gene helps predict the target) of the ssr F-test at lag
        ``maxlag``; ``Granger_padj`` is the Benjamini-Hochberg adjusted value over
        all non-NaN p-values.  NaN where the test cannot be run (constant series,
        NaN in the data, too few time points).

    Raises:
        ValueError: if ``len(target) != df.shape[1]`` or ``maxlag`` is not a positive integer.
    """
    return _granger_table(df, target, maxlag, 'forward', 'Granger')


def compute_reverse_granger(df, target, maxlag=1):
    """
    Reverse-direction Granger causality: target -> gene.

    Same inputs / output conventions as :func:`compute_granger`, but the columns
    are ``Name``, ``Granger_reverse`` (raw p-value, NOT 1 - p) and
    ``Granger_reverse_padj``.
    """
    return _granger_table(df, target, maxlag, 'reverse', 'Granger_reverse')


def granger_list(A, B, maxlag):
    """
    Compute the Granger test p-value (ssr F-test) for every lag 1..maxlag.

    Following the statsmodels convention, the test asks whether ``B`` Granger-causes
    ``A`` (``A`` is the predicted series, ``B`` the candidate cause).

    Returns:
        dict {lag: p-value}.
    """
    
    # Combine the two lists into a 2D array
    data = np.column_stack((A, B))
    
    # Perform the Granger causality test and return the result
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*verbose.*', category=FutureWarning)
        result = grangercausalitytests(data, maxlag, verbose=False)

    # In this function, we'll return just the p-value for each lag.
    p_values = {lag: result[lag][0]['ssr_ftest'][1] for lag in result.keys()}
    
    return p_values


# ****************** 2.2 CCM (Convergent cross mapping)*******************************
def compute_ccm(df, target, E=3, tau=1):
    """
    Convergent cross mapping between each gene (X) and the target (Y), using
    ``causal_ccm.ccm(gene, target, tau, E).causality()``.

    Returns:
        DataFrame ``Name``, ``CCM`` (correlation r), ``CCM_p`` (its p-value); one
        row per input gene in input order (not sorted).  Rows containing NaN give NaN.

    If the embedded series is too short -- ``N - (E-1)*tau < E + 2`` points -- CCM
    is undefined (the nearest-neighbour search needs E+2 points); all rows are
    NaN and a UserWarning is emitted instead of raising.
    """
    E = _check_positive_int(E, 'E')
    tau = _check_positive_int(tau, 'tau')
    names, X, y = _prepare(df, target)
    n = len(y)
    r = np.full(len(names), np.nan)
    p = np.full(len(names), np.nan)

    n_embedded = n - (E - 1) * tau
    if n_embedded < E + 2:
        warnings.warn(
            f"CCM skipped: {n} time points with E={E}, tau={tau} give only "
            f"{n_embedded} embedded points, but at least E+2={E + 2} are needed. "
            "Returning NaN for all genes.",
            UserWarning, stacklevel=2,
        )
    elif not np.isnan(y).any():
        for i, row in enumerate(X):
            if np.isnan(row).any():
                continue
            with warnings.catch_warnings():
                # constant predictions make pearsonr warn and return NaN
                warnings.simplefilter('ignore')
                r[i], p[i] = ccm(row, y, tau, E).causality()

    return pd.DataFrame({'Name': names, 'CCM': r, 'CCM_p': p})


# ******************** 2.3 CCA (Canonical correlation) ********************
def compute_cca(df, target, n_components=1):
    """
    Canonical correlation between each gene and the target.

    Note: with one gene (1 feature) against one metabolite (1 feature), CCA
    degenerates to the absolute Pearson correlation |r|; the sign is lost
    (values are in [0, 1]).  Use ``compute_pearson`` for a signed value.

    Parameters:
        n_components (int): must be 1. One gene against one metabolite has only one canonical
            component; anything else raises ValueError (it used to give an all-NaN column
            without any message).

    Returns:
        DataFrame ``Name``, ``CCA``; one row per input gene in input order.
        NaN for constant series (of the gene or the target) and for NaN in the data.

    Raises:
        ValueError: ``n_components != 1`` or ``len(target) != df.shape[1]``.
    """
    if isinstance(n_components, bool) or n_components != 1:
        raise ValueError(
            f"n_components must be 1, got {n_components!r}: CCA between a single gene and a "
            "single metabolite has only one canonical component.")
    names, X, y = _prepare(df, target)
    metabolite_concentration = y.reshape(-1, 1)
    out = np.full(len(names), np.nan)

    # It's essential to have more than one sample to perform CCA; a constant or NaN-containing
    # target / gene carries no information (same convention as the other metrics: NaN)
    if len(y) > 1 and not (np.isnan(y).any() or _is_constant(y)):
        for i, row in enumerate(X):
            if np.isnan(row).any() or _is_constant(row):
                continue
            gene_expression = row.reshape(-1, 1)
            cca = CCA(n_components=1)
            cca.fit(gene_expression, metabolite_concentration)
            gene_c, metabolite_c = cca.transform(gene_expression, metabolite_concentration)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[i] = np.corrcoef(gene_c.T, metabolite_c.T)[0, 1]

    return pd.DataFrame({'Name': names, 'CCA': out})


# ******************* 2.4 DTW (Dynamic Time Warping) *************************
def compute_dtw(df, target):
    """
    Dynamic time warping *distance* between each (min-max scaled) gene series and
    the (min-max scaled) target.  Direction: smaller = more similar (0 = identical).

    Returns:
        DataFrame ``Name``, ``DTW``; one row per input gene in input order.
        Rows / target containing NaN give NaN. A constant gene series gives NaN (min-max
        scaling would turn it into all zeros and give a finite, "good-looking" distance although
        it carries no information); a constant target gives NaN for all genes.
    """
    names, X, y = _prepare(df, target)
    out = np.full(len(names), np.nan)
    scaler = MinMaxScaler()

    if not (np.isnan(y).any() or _is_constant(y)):
        array2_norm = scaler.fit_transform(y.reshape(-1, 1))
        for i, row in enumerate(X):
            if np.isnan(row).any() or _is_constant(row):
                continue
            array1_norm = scaler.fit_transform(row.reshape(-1, 1))
            distance, _path = fastdtw(array1_norm, array2_norm)
            out[i] = distance

    return pd.DataFrame({'Name': names, 'DTW': out})


# ************************* 2.5 CCF *****************************
def compute_ccf(df, target, lag=1):
    """
    Cross-correlation  CCF = corr(gene[t], target[t + lag]).

    * lag > 0: the gene LEADS the metabolite by ``lag`` steps (the direction CAT
      Bridge looks for);
    * lag < 0: the metabolite leads the gene by ``|lag|`` steps;
    * lag = 0: contemporaneous correlation.

    Uses ``statsmodels.tsa.stattools.ccf`` (``ccf(a, b)[k] = corr(a[t+k], b[t])``,
    global mean/std, ``adjusted=True``), so ``lag >= 0`` is ``ccf(target, gene)[lag]``
    and ``lag < 0`` is ``ccf(gene, target)[-lag]``.

    Returns:
        DataFrame ``Name``, ``CCF``; one row per input gene in input order.
        NaN for constant series, NaN in the data, or ``|lag|`` beyond the series length.
    """
    if int(lag) != lag:
        raise ValueError(f"lag must be an integer, got {lag!r}")
    lag = int(lag)
    names, X, y = _prepare(df, target)
    out = np.full(len(names), np.nan)

    if not (np.isnan(y).any() or _is_constant(y)):
        for i, row in enumerate(X):
            if np.isnan(row).any() or _is_constant(row):
                continue
            if lag >= 0:
                cc = ccf(y, row)
                k = lag
            else:
                cc = ccf(row, y)
                k = -lag
            if k < len(cc):
                out[i] = cc[k]

    return pd.DataFrame({'Name': names, 'CCF': out})


# ***************** 2.6 Spearman / 2.7 Pearson *********************
def _corr_table(df, target, func, col):
    names, X, y = _prepare(df, target)
    r = np.full(len(names), np.nan)
    p = np.full(len(names), np.nan)
    if len(y) >= 2 and not (np.isnan(y).any() or _is_constant(y)):
        for i, row in enumerate(X):
            if np.isnan(row).any() or _is_constant(row):
                continue
            r[i], p[i] = func(row, y)
    return pd.DataFrame({'Name': names, col: r, col + '_p': p, col + '_padj': _bh_adjust(p)})


def compute_spearman(df, target):
    """
    Spearman rank correlation of each gene with the target (sign kept).

    Returns:
        DataFrame ``Name``, ``Spearman``, ``Spearman_p``, ``Spearman_padj``
        (Benjamini-Hochberg over non-NaN p); one row per gene in input order.
        Constant series / NaN in the data give NaN.
    """
    return _corr_table(df, target, spearmanr, 'Spearman')


def compute_pearson(df, target):
    """
    Pearson correlation of each gene with the target (sign kept).

    Returns:
        DataFrame ``Name``, ``Pearson``, ``Pearson_p``, ``Pearson_padj``
        (Benjamini-Hochberg over non-NaN p); one row per gene in input order.
        Constant series / NaN in the data give NaN.
    """
    return _corr_table(df, target, pearsonr, 'Pearson')


# ************** fuzz c mean clustering *********************
def clustering(df, n_clusters):
    import skfuzzy as fuzz  # lazy import (heavy optional dependency)

    # Prepare the data
    data = df.values.T  # The cmeans function expects data to be of shape (n_features, n_samples)
    
    # Perform fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, n_clusters, 2, error=0.005, maxiter=1000)
    
    # Get the cluster assignment for each sample
    # The argmax function is used to assign each sample to the cluster for which it has the highest membership value
    cluster_assignment = u.argmax(axis=0)
    
    # Create a DataFrame to hold the results
    results_df = pd.DataFrame({'Name': df.index, 'Cluster': cluster_assignment})
    
    return results_df


# ts Clustering
def ts_clustering(df, n_clusters, random_state=0, max_iter=5):
    """
    Cluster the time series data in the input dataframe using the TS-KMeans algorithm.
    
    Parameters:
        df (pandas.DataFrame): The input dataframe.
        n_clusters (int): The number of clusters to create.
        random_state (int): Seed for TimeSeriesKMeans (reproducible results). Default 0.
        max_iter (int): Maximum k-means iterations. Default 5.

    Returns:
        pandas.Series named 'Cluster', indexed like ``df.index`` (the input is not modified).
    """
    from tslearn.clustering import TimeSeriesKMeans  # lazy import (heavy optional dependency)
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    # Convert DataFrame to NumPy array for compatibility with tslearn
    data = df.values

    # Rescale the time series data so that their mean is 0 and their standard deviation is 1
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    data_scaled = scaler.fit_transform(data)

    # Create the KMeans model
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", max_iter=max_iter, random_state=random_state)

    # Fit the model to the data
    km.fit(data_scaled)

    # Get the cluster labels for each time series
    labels = km.labels_

    # Return the labels as a Series named 'Cluster' (the input df is left untouched)
    return pd.Series(labels, index=df.index, name='Cluster')
