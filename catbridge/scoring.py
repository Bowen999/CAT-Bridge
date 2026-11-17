"""Scoring and ranking of candidate features."""

import re

import pandas as pd
import numpy as np

from scipy.stats import linregress  # linear regression
from sklearn.decomposition import PCA

__all__ = [
    'score',
    'keywords_scores',
    'add_annotation_score',
    'trend_strength',
    'top_important_features',
    'compute_modulus',
    'compute_score',
    'METRIC_DIRECTION',
    'ASE_EXCLUDE',
]


# ***************  compute score ******************
def score(df):
    """
    Legacy helper: sum all columns except 'Name' into 'Score' and rank rows by it.

    This is the old, simple additive score and is NOT the same thing as
    :func:`compute_score` (which normalises, uses the modulus and handles metric
    direction). It is kept for backward compatibility. The input is not modified;
    a sorted copy (with 'Score' and 'Rank' columns) is returned. Any 'Score' /
    'Rank' columns already present in `df` are not included in the sum.
    """
    df = df.copy()
    # Exclude the 'Name' column (and any previous Score/Rank) and compute row sums
    df['Score'] = df.drop(columns=['Name', 'Score', 'Rank'], errors='ignore').sum(axis=1)

    # Rank rows from high to low based on the sum, '1' being the highest rank
    df['Rank'] = df['Score'].rank(method='min', ascending=False)

    # Sort dataframe by rank
    return df.sort_values('Rank')


# Annotation -> Score
keywords_scores = {'ase': 0.2, 'enzyme': 0.2, 'synthase': 0.2}

# Words that end in "ase" but are not enzymes. Only applied to the generic 'ase'
# suffix keyword (words that *really* match e.g. 'synthase' are never excluded).
ASE_EXCLUDE = frozenset({
    'disease', 'release', 'base', 'phase', 'case', 'increase', 'decrease',
    'database', 'purchase', 'phrase', 'erase', 'chase', 'cease', 'lease',
    'please', 'vase',
    # further non-enzyme words found in the pepper annotation / general English
    'prophase', 'prometaphase', 'metaphase', 'anaphase', 'telophase', 'interphase',
    'nucleobase', 'grease', 'crease', 'decease', 'appease',
})
_ASE_MIN_LEN = 5

_MISSING_DESCRIPTION_SCORE = 0.1


def _is_missing_text(value):
    """True for NaN/None/non-strings/blank strings."""
    return (not isinstance(value, str)) or value.strip() == ''


def add_annotation_score(df, keywords=keywords_scores):
    """
    Return a copy of `df` with a new column 'Description_Score', computed from
    the 'Description' column and the `keywords` dictionary ({suffix: score}).

    Matching rules:
      * the description is lower-cased and split into words on any
        non-alphanumeric character (so 'ligase,' and 'phosphatase;' match, and
        hyphenated words such as 'ATP-dependent-kinase' are split so that any
        part can match);
      * a word matches a keyword if it ends with it (a plural form such as
        'kinases' is also tried without its trailing 's'). For the generic suffix
        'ase' the word must also be at least 5 letters long and must not be in
        ``ASE_EXCLUDE`` (disease, release, base, phase, ...);
      * the highest score among all matches is used;
      * description present but no match -> 0;
      * description missing (NaN / None / empty / non-string) -> 0.1. This is
        deliberately kept: 0.1 means "unknown", sitting between "known enzyme"
        (0.2) and "known non-enzyme" (0), i.e. a neutral prior, so genes without
        annotation are neither rewarded nor penalised as if they were
        non-enzymes.

    Raises KeyError if there is no 'Description' column. The input is not modified.
    """
    if 'Description' not in df.columns:
        raise KeyError("add_annotation_score: the DataFrame has no 'Description' column")

    kw = [(str(k).lower(), float(v)) for k, v in keywords.items()]

    def compute_description_score(description):
        if _is_missing_text(description):
            return _MISSING_DESCRIPTION_SCORE
        best = None
        for token in re.split(r'[\W_]+', description.lower()):
            if not token:
                continue
            # also try the singular of simple plurals (kinases -> kinase)
            forms = (token, token[:-1]) if token.endswith('s') else (token,)
            for word in forms:
                for keyword, kscore in kw:
                    if not word.endswith(keyword):
                        continue
                    if keyword == 'ase' and (len(word) < _ASE_MIN_LEN or word in ASE_EXCLUDE):
                        continue
                    if best is None or kscore > best:
                        best = kscore
        return 0.0 if best is None else best

    out = df.copy()
    out['Description_Score'] = out['Description'].map(compute_description_score).astype(float)
    return out


# ************* Upward trend strength *************
def trend_strength(df):
    """
    Function to calculate and sort the strength of the upward trend in each row of a DataFrame.

    This function calculates the trend strength (r-value) for each row in the DataFrame 
    using linear regression. It then adds a new column to the DataFrame with these r-values. 
    Finally, it sorts the DataFrame by these r-values in descending order and returns the 
    sorted DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be processed. 

    Returns:
    df_sorted_by_trend_strength (pandas.DataFrame): The input DataFrame, but with an added 
    column for trend strength and sorted by this column in descending order.
    """

    # Function to calculate the trend strength (r-value)
    def upward_trend_strength(row):
        _, _, rvalue, _, _ = linregress(x=range(len(row)), y=row)
        return rvalue

    # Apply function to each row to get trend strengths
    trend_strength = df.apply(upward_trend_strength, axis=1)

    # Add a new column to df with trend strengths
    df_with_trend_strength = df.assign(trend_strength=trend_strength)

    # Sort df by trend strength (r-value) in descending order
    df_sorted_by_trend_strength = df_with_trend_strength.sort_values(by='trend_strength', ascending=False)

    # Return sorted dataframe
    return df_sorted_by_trend_strength


# ****************** Importance *******************
def top_important_features(df, n_top):
    """
    Rank features (genes / metabolites) by their importance in a PCA model.

    importance_j = sum_k explained_variance_ratio_k * loading_kj ** 2

    i.e. the squared loading of feature j on principal component k, weighted by
    the share of variance that component explains. (The unweighted sum of squared
    loadings over all components, used previously, treats a component that
    explains 0.1% of the variance like the first one and so only reflects
    geometry: nearly constant features could rank high.)

    Parameters:
    df (pd.DataFrame): rows are features (e.g. genes), columns are samples / time
                       points (the frame is transposed internally so that PCA
                       runs over samples). The index holds the feature names.
                       Must not contain NaN.
    n_top (int): The number of top features to return.

    Returns:
    pd.DataFrame with columns 'Gene' and 'Importance', sorted by importance
    (descending), at most n_top rows. Errors are raised, not swallowed.
    """
    # Transpose so that rows are samples/time points and columns are genes
    df_transposed = df.transpose()

    pca = PCA()
    pca.fit(df_transposed)

    # (n_components, n_features) loadings and the variance share of each component
    loadings = pca.components_
    evr = np.nan_to_num(pca.explained_variance_ratio_)  # all-constant data -> 0/0

    importance = np.sum(evr[:, None] * loadings ** 2, axis=0)

    importance_df = pd.DataFrame({'Gene': df_transposed.columns, 'Importance': importance})
    importance_df = importance_df.sort_values('Importance', ascending=False, kind='mergesort')
    return importance_df.head(n_top)


def compute_modulus(a, b, c=None):
    if c is None:
        return (a ** 2 + b ** 2) ** 0.5
    else:
        return (a ** 2 + b ** 2 + c ** 2) ** 0.5


# Direction of each metric: how to turn the column into "evidence strength".
#   'pvalue' : raw p-value, smaller is stronger  -> -log10(p)
#   'lower'  : a distance, smaller is stronger   -> -x
#   'higher' : larger is stronger (sign kept: a negative correlation scores low)
# Column names not listed here are treated as 'higher'.
METRIC_DIRECTION = {
    'Granger': 'pvalue',
    'Granger_reverse': 'pvalue',
    'DTW': 'lower',
}

_P_FLOOR = 1e-300


def _evidence_strength(values, direction):
    """Map a raw metric column to a 'larger = stronger evidence' float Series."""
    x = pd.to_numeric(values, errors='coerce').astype(float)
    x = x.where(np.isfinite(x))  # +-inf -> NaN
    if direction == 'pvalue':
        return -np.log10(x.clip(lower=_P_FLOOR, upper=1.0))
    if direction == 'lower':
        return -x
    return x


def _normalize(x, scaling):
    """Scale to 0-1 ignoring NaN; NaN -> 0 and a constant column -> all 0."""
    x = pd.to_numeric(x, errors='coerce').astype(float)
    x = x.where(np.isfinite(x))
    valid = x.dropna()
    if valid.empty:
        return pd.Series(0.0, index=x.index)
    if scaling == 'robust':
        lo, hi = np.quantile(valid.to_numpy(), [0.01, 0.99])
        x = x.clip(lower=lo, upper=hi)
    else:
        lo, hi = valid.min(), valid.max()
    span = hi - lo
    if not span > 0:
        return pd.Series(0.0, index=x.index)
    return ((x - lo) / span).fillna(0.0)


def compute_score(df, col, scaling='minmax'):
    """
    Combine one association metric, the fold change and the annotation into a
    0-1 score per gene and rank the genes (best first).

        Score = sqrt(norm_col**2 + norm_fc**2 + Description_Score**2)

    rescaled to 0-1 (min-max). The Description_Score term is only used when that
    column exists. The input DataFrame is never modified.

    Parameters:
    df: DataFrame with 'Name', `col`, 'log2FoldChange'; optional 'Cluster',
        'Description_Score' and f'{col}_padj'.
    col: the metric column used for ranking, e.g. 'Granger', 'CCM', 'CCA', 'DTW',
        'CCF', 'Spearman', 'Pearson'.
    scaling: 'minmax' (default) or 'robust'. Quantiles and min/max ignore NaN;
        a constant column normalises to 0 (no division by zero).
        * 'minmax' is plain min-max scaling (the formula of the paper). It keeps
          the differences in magnitude between the top-ranked genes, but is
          sensitive to extreme values that are still in the data.
        * 'robust' clips both inputs to their [1%, 99%] quantiles before min-max
          scaling, so a few extreme values cannot squash all other genes. The
          price: every gene in the top 1% of a component is clipped to exactly
          1.0 and they tie, which removes resolution among the top-ranked genes
          (with ~35,000 genes, ~350 genes per component). Use it only if you have
          checked that residual extreme values are left in your data. The usual
          sources are already handled upstream (pseudo-count for fold changes
          without replicates, LFC shrinkage for DESeq2).

    Metric direction (module constant METRIC_DIRECTION; column names not in it are
    treated as 'higher'):
      * 'Granger', 'Granger_reverse' (raw p-values): evidence = -log10(clip(p,
        1e-300, 1)). Smaller p ranks higher. The old 1-p made p=0.001 and p=0.04
        differ by only 0.04, i.e. almost no resolution.
      * 'DTW' (distance): evidence = -x, smaller distance ranks higher.
      * everything else (CCM, CCA, CCF, Spearman, Pearson, ...): evidence = x,
        sign kept, so a negative correlation gets a low score (by design).

    Why raw p and not padj for ranking: with few time points the BH-adjusted q
    values contain many ties, which would destroy ranking resolution. f'{col}_padj'
    is only carried through to the output as a significance reference.

    Missing values: if the metric or log2FoldChange is NaN, the corresponding
    normalised component is set to 0 ("no evidence") instead of letting the whole
    row's Score become NaN. Non-finite values (+-inf) are treated like NaN.

    Returns a DataFrame sorted by Score (descending) with columns ['Name', col,
    'log2FoldChange', ('Cluster' if present), 'Score'] plus f'{col}_padj' if
    present. The index starts at 1 and is the rank. `col` holds the original
    values (for Granger: the raw p-value, no longer 1-p).

    Raises KeyError if `col`, 'log2FoldChange' or 'Name' is missing, ValueError
    for an unknown `scaling`.
    """
    col = str(col)
    if scaling not in ('robust', 'minmax'):
        raise ValueError(f"scaling must be 'robust' or 'minmax', got {scaling!r}")
    for required in (col, 'log2FoldChange', 'Name'):
        if required not in df.columns:
            raise KeyError(f"compute_score: column {required!r} not found in the DataFrame "
                           f"(available: {list(df.columns)})")

    direction = METRIC_DIRECTION.get(col, 'higher')

    work = df.copy()
    work['norm_col'] = _normalize(_evidence_strength(work[col], direction), scaling)
    work['norm_fc'] = _normalize(work['log2FoldChange'], scaling)

    squares = work['norm_col'] ** 2 + work['norm_fc'] ** 2
    if 'Description_Score' in work.columns:
        desc = pd.to_numeric(work['Description_Score'], errors='coerce').fillna(0.0).astype(float)
        squares = squares + desc ** 2
    work['Score'] = np.sqrt(squares)

    # min-max to 0-1 (constant -> 0)
    smin, smax = work['Score'].min(), work['Score'].max()
    work['Score'] = (work['Score'] - smin) / (smax - smin) if smax > smin else 0.0

    # stable sort: ties keep their original order, so results are deterministic
    work = work.sort_values(by='Score', ascending=False, kind='mergesort')

    cols = ['Name', col, 'log2FoldChange']
    if 'Cluster' in work.columns:
        cols.append('Cluster')
    cols.append('Score')
    padj = f'{col}_padj'
    if padj in work.columns and padj not in cols:
        cols.append(padj)

    result = work[cols].reset_index(drop=True)
    result.index = pd.RangeIndex(1, len(result) + 1)
    return result
