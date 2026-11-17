"""Similarity / causality metrics between each feature (row) and a target series, plus clustering.

All metrics are computed for the whole gene x time matrix at once (numpy), not gene by gene.
Each batched implementation reproduces the statistic of the reference implementation it
replaces (statsmodels ``grangercausalitytests`` / ``ccf``, ``causal_ccm.ccm``, ``fastdtw``,
``scipy.stats.spearmanr`` / ``pearsonr``, sklearn ``CCA``, tslearn ``TimeSeriesKMeans``) to
floating-point precision; ``tests/test_metrics_equivalence.py`` checks this against the
reference libraries when they are installed.
"""

import inspect
import warnings

import pandas as pd
import numpy as np

from ._lazy import LazyModule, lazy_callable

# imported on first use (keeps `import catbridge` fast)
special = LazyModule('scipy.special')
stats = LazyModule('scipy.stats')
cdist = lazy_callable('scipy.spatial.distance', 'cdist')

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

# Genes per batch. Bounds the memory of the batched computations (e.g. the DTW predecessor
# table is genes x n x n bytes); the results do not depend on it.
_CHUNK = 8192


# ************* shared helpers **************************
def _prepare(df, target):
    """
    Validate the inputs and return ``(names, X, y)``.

    ``names`` is the list of row labels (input order), ``X`` a float ndarray of
    shape (n_genes, n_timepoints) and ``y`` a float 1-D ndarray.  Neither ``df``
    nor ``target`` is modified (``X`` / ``y`` are read-only views or copies).

    Raises ValueError if ``len(target) != df.shape[1]`` or if ``df`` has non-numeric cells.
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
    try:
        X = df.to_numpy(dtype=float)
    except (TypeError, ValueError) as e:
        bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        raise ValueError(
            f"The table contains non-numeric values ({e}); non-numeric columns: {bad[:10]}. "
            "Convert or remove them first (catbridge.read_upload reports the offending cells)."
        ) from None
    return list(df.index), X, y


def _is_constant(v):
    return np.ptp(v) == 0 or np.std(v) == 0


def _row_ok(X):
    """Rows that carry information: all values finite and not constant.

    "constant" uses the same test as before (``ptp == 0 or std == 0``); rows with NaN/inf are
    not usable either (the metrics give NaN for them)."""
    finite = np.isfinite(X).all(axis=1)
    with np.errstate(invalid='ignore'):
        const = (np.ptp(X, axis=1) == 0) | (np.std(X, axis=1) == 0)
    return finite & ~const


def _target_ok(y):
    return bool(np.isfinite(y).all()) and not _is_constant(y)


def _bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values; NaN entries stay NaN.

    Same arithmetic as ``statsmodels.stats.multitest.multipletests(p, method='fdr_bh')[1]``
    over the non-NaN values (statsmodels is not imported for this)."""
    p = np.asarray(p, dtype=float)
    padj = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        pv = p[ok]
        order = np.argsort(pv)
        ps = np.take(pv, order)
        nobs = len(ps)
        ecdf = np.arange(1, nobs + 1) / float(nobs)
        corrected = np.minimum.accumulate((ps / ecdf)[::-1])[::-1]
        corrected[corrected > 1] = 1
        out = np.empty_like(corrected)
        out[order] = corrected
        padj[ok] = out
    return padj


def _check_positive_int(value, name):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _chunks(n, size=_CHUNK):
    for start in range(0, n, size):
        yield slice(start, min(n, start + size))


def _supports_axis(func):
    try:
        return 'axis' in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _pearson_rows(A, B):
    """``scipy.stats.pearsonr`` of each row of A with the matching row of B (r, p).

    Uses scipy's own vectorised implementation (``axis=1``, scipy >= 1.9) so the values are
    those of the 1-D call; older scipy falls back to a loop. Constant rows give NaN silently."""
    from scipy.stats import pearsonr
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    r = np.full(A.shape[0], np.nan)
    p = np.full(A.shape[0], np.nan)
    if A.shape[0] == 0:
        return r, p
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # ConstantInputWarning / NearConstantInputWarning
        if _supports_axis(pearsonr):
            for sl in _chunks(A.shape[0]):
                res = pearsonr(A[sl], B[sl], axis=1)
                r[sl], p[sl] = res[0], res[1]
        else:  # pragma: no cover - old scipy
            for i in range(A.shape[0]):
                r[i], p[i] = pearsonr(A[i], B[i])
    return r, p


# ************* 2.1 Granger **************************
def _ols_ssr_rank(A, b):
    """``statsmodels OLS(b, A).fit()`` (default method 'pinv') for a stack of designs
    A (g, m, k) and responses b (g, m): residual sum of squares and matrix rank of the design.

    Same steps as statsmodels' ``pinv_extended`` (SVD, rcond=1e-15), ``np.dot`` products and
    ``matrix_rank(diag(singular values))``. The SVDs are computed for the whole stack at once
    (LAPACK per matrix, identical to single calls); the few small products are done per
    matrix with ``np.dot`` exactly as statsmodels does, because the p-value of a
    near-zero F statistic is very sensitive to the last bits of the two ssr values (a batched
    matmul changes the summation order and moved such p-values by up to ~2e-8)."""
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    s_orig = s
    cutoff = 1e-15 * s.max(axis=1, keepdims=True)
    big = s > cutoff
    sinv = np.zeros_like(s)
    sinv[big] = 1.0 / s[big]
    g = A.shape[0]
    ssr = np.empty(g)
    dot, transpose, multiply = np.dot, np.transpose, np.multiply
    for i in range(g):
        pinv = dot(transpose(vt[i]), multiply(sinv[i][:, np.newaxis], transpose(u[i])))
        beta = dot(pinv, b[i])
        wresid = b[i] - dot(A[i], beta)
        ssr[i] = dot(wresid, wresid)
    k = A.shape[2]
    rank = (s_orig > s_orig.max(axis=1, keepdims=True) * k * np.finfo(float).eps).sum(axis=1)
    return ssr, rank


def _granger_p_batch(cause, effect, maxlag):
    """p-value of the ssr F-test of ``statsmodels.tsa.stattools.grangercausalitytests(
    np.column_stack([effect_i, cause_i]), maxlag)[maxlag][0]['ssr_ftest'][1]`` for every row i.

    ``cause`` / ``effect`` are (g, n) arrays of finite, non-constant rows. NaN where statsmodels
    raises: too few observations (``n <= 3*maxlag + 1``), a constant lag column or a
    perfect fit at ANY lag 1..maxlag (statsmodels tests all lags up to maxlag and raises
    InfeasibleTestError at the first infeasible one)."""
    g, n = cause.shape
    p = np.full(g, np.nan)
    if g == 0 or n <= 3 * maxlag + 1:
        return p
    eps = np.finfo(float).eps
    ok = np.ones(g, dtype=bool)
    for lag in range(1, maxlag + 1):
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            break
        E, C = effect[idx], cause[idx]
        m = n - lag
        # dta = lagmat2ds(x, lag, trim='both', dropex=1): [effect_t, effect lags 1..lag,
        # cause lags 1..lag]. statsmodels regresses on the column VIEW dta[:, 0]; it is kept
        # as a strided view here too, because BLAS rounds differently for strided vectors.
        dta = np.stack([E[:, lag:]] + [E[:, lag - j:n - j] for j in range(1, lag + 1)]
                       + [C[:, lag - j:n - j] for j in range(1, lag + 1)], axis=2)
        yt = dta[:, :, 0]
        lagcols = dta[:, :, 1:]
        # add_constant + "column with constant values" check -> InfeasibleTestError
        const_col = (lagcols.max(axis=1) == lagcols.min(axis=1)).any(axis=1)
        ones = np.ones((len(idx), m, 1))
        own = np.concatenate([lagcols[:, :, :lag], ones], axis=2)
        joint = np.concatenate([lagcols, ones], axis=2)
        ssr_own = np.empty(len(idx))
        ssr_joint = np.empty(len(idx))
        rank_joint = np.empty(len(idx))
        for sl in _chunks(len(idx)):
            ssr_own[sl], _ = _ols_ssr_rank(own[sl], yt[sl])
            ssr_joint[sl], rank_joint[sl] = _ols_ssr_rank(joint[sl], yt[sl])
        tss = np.empty(len(idx))
        for i in range(len(idx)):       # statsmodels centered_tss (per row, same np.dot)
            yc = yt[i] - yt[i].mean()
            tss[i] = np.dot(yc, yc)
        with np.errstate(divide='ignore', invalid='ignore'):
            rsq = 1 - ssr_joint / tss
            bad = (const_col | (tss == 0) | (ssr_joint == 0) | np.isnan(rsq)
                   | (ssr_joint / tss < eps))
        ok[idx[bad]] = False
        if lag == maxlag:
            good = ~bad
            df_resid = m - rank_joint[good]
            with np.errstate(divide='ignore', invalid='ignore'):
                fgc1 = (ssr_own[good] - ssr_joint[good]) / ssr_joint[good] / lag * df_resid
            p[idx[good]] = stats.f.sf(fgc1, lag, df_resid)
    return p


def _granger_max_lag(n_obs):
    """Largest ``maxlag`` that ``statsmodels.grangercausalitytests`` accepts for ``n_obs`` time
    points (it requires ``n_obs > 3 * maxlag + 1`` with the constant term); 0 if none."""
    return max((int(n_obs) - 2) // 3, 0)


def _granger_table(df, target, maxlag, direction, col):
    maxlag = _check_positive_int(maxlag, 'maxlag')
    names, X, y = _prepare(df, target)
    p = np.full(len(names), np.nan)
    # data would allow a test (no NaN, not constant) -- used for the warning below
    testable = _row_ok(X) & _target_ok(y)
    rows = np.flatnonzero(testable)
    if rows.size:
        Y = np.broadcast_to(y, (rows.size, len(y)))
        if direction == 'forward':      # gene -> target
            p[rows] = _granger_p_batch(X[rows], Y, maxlag)
        else:                           # target -> gene
            p[rows] = _granger_p_batch(Y, X[rows], maxlag)
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
        ``maxlag``, i.e. ``statsmodels.tsa.stattools.grangercausalitytests(
        column_stack([target, gene]), maxlag)[maxlag][0]['ssr_ftest'][1]`` (computed for all
        genes at once with the same OLS steps); ``Granger_padj`` is the Benjamini-Hochberg
        adjusted value over all non-NaN p-values.  NaN where the test cannot be run
        (constant series, NaN in the data, too few time points, or where statsmodels would
        raise InfeasibleTestError at any lag 1..maxlag).

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
    from statsmodels.tsa.stattools import grangercausalitytests  # lazy: heavy import

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
def _ccm_neighbours(y, E, tau):
    """Shadow manifold of the target and its nearest neighbours / weights, as in
    ``causal_ccm.ccm(X, Y=target, tau, E)`` (only Y enters the manifold, so these are the
    same for every gene). Returns (t_steps, neighbour time steps (m, E+1), weights (m, E+1))."""
    L = len(y)
    t_steps = np.arange((E - 1) * tau, L)
    vecs = np.array([[y[t - k * tau] for k in range(E)] for t in t_steps])
    dists = cdist(vecs, vecs)
    eps = 0.000001
    nn = np.empty((len(t_steps), E + 1), dtype=int)
    w = np.empty((len(t_steps), E + 1))
    for row in range(len(t_steps)):
        dist_t = dists[row]
        nearest_inds = np.argsort(dist_t)[1:E + 1 + 1]
        nearest_distances = dist_t[nearest_inds]
        u = np.exp(-nearest_distances / np.max([eps, nearest_distances[0]]))
        w[row] = u / np.sum(u)
        nn[row] = t_steps[nearest_inds]
    return t_steps, nn, w


def compute_ccm(df, target, E=3, tau=1):
    """
    Convergent cross mapping between each gene (X) and the target (Y): the value of
    ``causal_ccm.ccm(gene, target, tau, E).causality()``, computed for all genes at once
    (causal_ccm itself is not needed).

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
    elif np.isfinite(y).all():
        rows = np.flatnonzero(np.isfinite(X).all(axis=1))
        if rows.size:
            t_steps, nn, w = _ccm_neighbours(y, E, tau)
            for sl in _chunks(rows.size):
                Xs = X[rows[sl]]
                # C-contiguous rows: numpy sums a strided axis in a different order, and
                # causal_ccm passes fresh (contiguous) arrays to pearsonr
                x_true = np.ascontiguousarray(Xs[:, t_steps])
                x_hat = np.ascontiguousarray((w * Xs[:, nn]).sum(axis=2))
                r[rows[sl]], p[rows[sl]] = _pearson_rows(x_true, x_hat)

    return pd.DataFrame({'Name': names, 'CCM': r, 'CCM_p': p})


# ******************** 2.3 CCA (Canonical correlation) ********************
def compute_cca(df, target, n_components=1):
    """
    Canonical correlation between each gene and the target.

    With one gene (1 feature) against one metabolite (1 feature), CCA degenerates to the
    absolute Pearson correlation |r|; the sign is lost (values are in [0, 1]). It is computed
    as ``|pearsonr(gene, target)[0]|`` for all genes at once (the value sklearn's
    ``CCA(n_components=1)`` gives, up to floating-point rounding). Use ``compute_pearson``
    for a signed value.

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
    out = np.full(len(names), np.nan)

    # It's essential to have more than one sample to perform CCA; a constant or NaN-containing
    # target / gene carries no information (same convention as the other metrics: NaN)
    if len(y) > 1 and _target_ok(y):
        rows = np.flatnonzero(_row_ok(X))
        r, _ = _pearson_rows(X[rows], np.broadcast_to(y, (rows.size, len(y))))
        out[rows] = np.abs(r)

    return pd.DataFrame({'Name': names, 'CCA': out})


# ******************* 2.4 DTW (Dynamic Time Warping) *************************
def _minmax_rows(X):
    """``sklearn.preprocessing.MinMaxScaler().fit_transform(row.reshape(-1, 1))`` for every
    (non-constant) row: x * scale + min with scale = 1 / (max - min), min = 0 - xmin * scale."""
    xmin = X.min(axis=1, keepdims=True)
    scale = 1 / (X.max(axis=1, keepdims=True) - xmin)
    return X * scale + (0 - xmin * scale)


def _dtw_dp(A, b, window=None, want_path=False):
    """Dynamic programming of fastdtw's ``__dtw`` for every row of A against b.

    D[i, j] = |a_i - b_j| + min(D[i-1, j], D[i, j-1], D[i-1, j-1]) over the cells of
    ``window`` (g, n, nb) bool (None = all cells), cells outside the window are inf. Ties are
    broken like ``min(..., key=...)`` in fastdtw: (i-1, j) first, then (i, j-1), then (i-1, j-1).
    Returns the distances (g,) and, with ``want_path``, the warping path as a bool mask."""
    g, n = A.shape
    nb = len(b)
    inf = np.inf
    prev = np.full((g, nb + 1), inf)
    prev[:, 0] = 0.0
    pred = np.zeros((g, n + 1, nb + 1), dtype=np.int8) if want_path else None
    for i in range(1, n + 1):
        cur = np.full((g, nb + 1), inf)
        ai = A[:, i - 1]
        for j in range(1, nb + 1):
            dt = np.abs(ai - b[j - 1])
            best = prev[:, j] + dt                  # (i-1, j)
            code = np.zeros(g, dtype=np.int8)
            cand = cur[:, j - 1] + dt               # (i, j-1)
            sel = cand < best
            best = np.where(sel, cand, best)
            code[sel] = 1
            cand = prev[:, j - 1] + dt              # (i-1, j-1)
            sel = cand < best
            best = np.where(sel, cand, best)
            code[sel] = 2
            if window is not None:
                inside = window[:, i - 1, j - 1]
                best = np.where(inside, best, inf)
                code[~inside] = 0
            cur[:, j] = best
            if want_path:
                pred[:, i, j] = code
        prev = cur
    dist = prev[:, nb].copy()
    if not want_path:
        return dist, None
    path = np.zeros((g, n, nb), dtype=bool)
    ci = np.full(g, n)
    cj = np.full(g, nb)
    rows = np.arange(g)
    for _ in range(n + nb + 1):
        active = ~((ci == 0) & (cj == 0))
        if not active.any():
            break
        if ((ci[active] == 0) | (cj[active] == 0)).any():
            raise RuntimeError("DTW backtracking left the grid (bug: unreachable cell).")
        path[rows[active], ci[active] - 1, cj[active] - 1] = True
        code = pred[rows, ci, cj]
        step_i = active & (code != 1)          # up (0) and diagonal (2) move i
        step_j = active & (code != 0)          # left (1) and diagonal (2) move j
        ci = ci - step_i
        cj = cj - step_j
    return dist, path


def _expand_window(path, len_x, len_y, radius):
    """fastdtw's ``__expand_window`` for a stack of coarse paths (g, nc, nbc) -> fine window
    (g, len_x, len_y) bool."""
    g, nc, nbc = path.shape
    rx = (len_x - 1) // 2 + 1        # coarse indices that map into the fine grid
    ry = (len_y - 1) // 2 + 1
    # dilate the path by `radius` in both directions (cells outside the coarse grid included)
    dil = np.zeros((g, rx, ry), dtype=bool)
    for a in range(-radius, radius + 1):
        for c in range(-radius, radius + 1):
            # dil[i, j] |= path[i - a, j - c]
            i0, i1 = max(0, a), min(rx, nc + a)
            j0, j1 = max(0, c), min(ry, nbc + c)
            if i0 < i1 and j0 < j1:
                dil[:, i0:i1, j0:j1] |= path[:, i0 - a:i1 - a, j0 - c:j1 - c]
    # each coarse cell -> its 2 x 2 fine cells
    win_ = np.repeat(np.repeat(dil, 2, axis=1), 2, axis=2)[:, :len_x, :len_y]
    # rows: contiguous run starting at the first cell at or after the previous row's start
    window = np.zeros((g, len_x, len_y), dtype=bool)
    jj = np.arange(len_y)
    start = np.zeros(g, dtype=int)
    for i in range(len_x):
        row = win_[:, i, :] & (jj[None, :] >= start[:, None])
        has = row.any(axis=1)
        if not has.all():
            raise RuntimeError("fastdtw window construction failed (bug: empty window row).")
        first = row.argmax(axis=1)
        after = (jj[None, :] > first[:, None]) & ~win_[:, i, :]
        end = np.where(after.any(axis=1), after.argmax(axis=1), len_y)
        window[:, i, :] = (jj[None, :] >= first[:, None]) & (jj[None, :] < end[:, None])
        start = first
    return window


def _fastdtw_batch(A, b, radius=1):
    """``fastdtw.fastdtw(a, b, radius)[0]`` (with |a_i - b_j| as the cost) for every row a of A.

    Same recursion as fastdtw 0.3.4: halve both series (dropping an odd last point), solve
    the coarse problem, expand its path by ``radius`` into a window, and run the windowed DTW;
    series shorter than radius + 2 are solved exactly."""
    def rec(A, b, want_path):
        n, nb = A.shape[1], len(b)
        if n < radius + 2 or nb < radius + 2:
            return _dtw_dp(A, b, None, want_path)
        A2 = (A[:, 0:n - n % 2:2] + A[:, 1:n - n % 2:2]) / 2
        b2 = (b[0:nb - nb % 2:2] + b[1:nb - nb % 2:2]) / 2
        _, path = rec(A2, b2, True)
        window = _expand_window(path, n, nb, radius)
        return _dtw_dp(A, b, window, want_path)

    out = np.empty(A.shape[0])
    n = A.shape[1]
    size = max(1, min(_CHUNK, int(5e6 // max(1, (n + 1) ** 2))))
    for sl in _chunks(A.shape[0], size):
        out[sl] = rec(A[sl], b, False)[0]
    return out


def _dtw_exact_batch(A, b):
    """Exact DTW distance (|a_i - b_j| cost, full grid) for every row of A against b."""
    out = np.empty(A.shape[0])
    for sl in _chunks(A.shape[0]):
        out[sl] = _dtw_dp(A[sl], b, None, False)[0]
    return out


def compute_dtw(df, target, *, method='fastdtw'):
    """
    Dynamic time warping *distance* between each (min-max scaled) gene series and
    the (min-max scaled) target.  Direction: smaller = more similar (0 = identical).

    Parameters:
        method: ``'fastdtw'`` (default) -- the value of ``fastdtw.fastdtw(gene, target)``
            (radius 1), the approximation used by CAT Bridge since 1.0, reproduced for all
            genes at once (the fastdtw package is not needed). ``'exact'`` -- the exact DTW
            distance that fastdtw approximates (always <= the fastdtw value). The two differ;
            see the CHANGELOG (1.1.0, "Speed and input checks") for how much on the demo data.

    Returns:
        DataFrame ``Name``, ``DTW``; one row per input gene in input order.
        Rows / target containing NaN give NaN. A constant gene series gives NaN (min-max
        scaling would turn it into all zeros and give a finite, "good-looking" distance although
        it carries no information); a constant target gives NaN for all genes.
    """
    if method not in ('fastdtw', 'exact'):
        raise ValueError(f"method must be 'fastdtw' or 'exact', got {method!r}.")
    names, X, y = _prepare(df, target)
    out = np.full(len(names), np.nan)

    if _target_ok(y):
        b = _minmax_rows(y[None, :])[0]
        rows = np.flatnonzero(_row_ok(X))
        if rows.size:
            A = _minmax_rows(X[rows])
            out[rows] = _fastdtw_batch(A, b) if method == 'fastdtw' else _dtw_exact_batch(A, b)

    return pd.DataFrame({'Name': names, 'DTW': out})


# ************************* 2.5 CCF *****************************
def compute_ccf(df, target, lag=1):
    """
    Cross-correlation  CCF = corr(gene[t], target[t + lag]).

    * lag > 0: the gene LEADS the metabolite by ``lag`` steps (the direction CAT
      Bridge looks for);
    * lag < 0: the metabolite leads the gene by ``|lag|`` steps;
    * lag = 0: contemporaneous correlation.

    Same value as ``statsmodels.tsa.stattools.ccf`` (``ccf(a, b)[k] = corr(a[t+k], b[t])``,
    global mean/std, ``adjusted=True``), so ``lag >= 0`` is ``ccf(target, gene)[lag]``
    and ``lag < 0`` is ``ccf(gene, target)[-lag]``:
    ``sum_t (a[t+k] - mean a)(b[t] - mean b) / (n - k) / (std a * std b)``, summed directly
    for all genes at once. (statsmodels sums with an FFT; the two differ by ~1e-16, and the
    FFT result itself varies at that level with the memory layout, the direct sum does not.)

    Returns:
        DataFrame ``Name``, ``CCF``; one row per input gene in input order.
        NaN for constant series, NaN in the data, or ``|lag|`` beyond the series length.
    """
    if isinstance(lag, bool) or int(lag) != lag:
        raise ValueError(f"lag must be an integer, got {lag!r}")
    lag = int(lag)
    names, X, y = _prepare(df, target)
    out = np.full(len(names), np.nan)
    n = len(y)
    k = abs(lag)

    if _target_ok(y) and k < n:
        rows = np.flatnonzero(_row_ok(X))
        Xs = X[rows]
        xo = Xs - Xs.mean(axis=1, keepdims=True)
        yo = y - y.mean()
        if lag >= 0:      # ccf(target, gene)[k]: target[t + k] * gene[t]
            num = (yo[k:][None, :] * xo[:, :n - k]).sum(axis=1)
        else:             # ccf(gene, target)[k]: gene[t + k] * target[t]
            num = (xo[:, k:] * yo[:n - k][None, :]).sum(axis=1)
        out[rows] = (num / (n - k)) / (np.std(y) * np.std(Xs, axis=1))

    return pd.DataFrame({'Name': names, 'CCF': out})


# ***************** 2.6 Spearman / 2.7 Pearson *********************
def _spearman_rows(X, y):
    """``scipy.stats.spearmanr(row, y)`` for every row of X (r, two-sided p).

    Same steps as scipy: average ranks, ``np.corrcoef`` of the two rank vectors, then
    t = r * sqrt(dof / ((r + 1)(1 - r))) and p = 2 * t_dof.sf(|t|). Ranks are multiples of
    1/2, so the covariances are exact and r is the same number as scipy's."""
    n = X.shape[1]
    rx = stats.rankdata(X, axis=1)
    ry = stats.rankdata(y)
    xm = rx - rx.mean(axis=1, keepdims=True)
    ym = ry - ry.mean()
    inv = np.true_divide(1, n - 1)
    c_xx = np.einsum('ij,ij->i', xm, xm) * inv
    c_yy = np.dot(ym, ym) * inv
    c_xy = (xm @ ym) * inv
    r = (c_xy / np.sqrt(c_yy)) / np.sqrt(c_xx)
    r = np.clip(r, -1, 1)
    dof = n - 2
    with np.errstate(divide='ignore'):
        t = r * np.sqrt((dof / ((r + 1.0) * (1.0 - r))).clip(0))
    p = 2 * special.stdtr(dof, -np.abs(t))
    return r, p


def _corr_table(df, target, kind, col):
    names, X, y = _prepare(df, target)
    r = np.full(len(names), np.nan)
    p = np.full(len(names), np.nan)
    if len(y) >= 2 and _target_ok(y):
        rows = np.flatnonzero(_row_ok(X))
        if rows.size:
            if kind == 'spearman':
                r[rows], p[rows] = _spearman_rows(X[rows], y)
            else:
                r[rows], p[rows] = _pearson_rows(X[rows], np.broadcast_to(y, (rows.size, len(y))))
    return pd.DataFrame({'Name': names, col: r, col + '_p': p, col + '_padj': _bh_adjust(p)})


def compute_spearman(df, target):
    """
    Spearman rank correlation of each gene with the target (sign kept).

    Returns:
        DataFrame ``Name``, ``Spearman``, ``Spearman_p``, ``Spearman_padj``
        (Benjamini-Hochberg over non-NaN p); one row per gene in input order.
        Constant series / NaN in the data give NaN. Values are those of
        ``scipy.stats.spearmanr(gene, target)``.
    """
    return _corr_table(df, target, 'spearman', 'Spearman')


def compute_pearson(df, target):
    """
    Pearson correlation of each gene with the target (sign kept).

    Returns:
        DataFrame ``Name``, ``Pearson``, ``Pearson_p``, ``Pearson_padj``
        (Benjamini-Hochberg over non-NaN p); one row per gene in input order.
        Constant series / NaN in the data give NaN. Values are those of
        ``scipy.stats.pearsonr(gene, target)``.
    """
    return _corr_table(df, target, 'pearson', 'Pearson')


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
class _EmptyCluster(Exception):
    pass


def _ts_kmeans_euclidean(X, n_clusters, max_iter, random_state, tol=1e-6, max_attempts=10):
    """Labels of ``tslearn.clustering.TimeSeriesKMeans(n_clusters, metric='euclidean',
    max_iter, random_state=random_state).fit(X)`` (tslearn 0.6.3, n_init=1, k-means++ init),
    with the same random number stream and the same numpy/scipy/sklearn calls.

    X: (n_ts, sz, 1) float array. Returns (labels, n_empty) where n_empty > 0 means that
    every attempt (tslearn makes up to 10) ended with an empty cluster."""
    from sklearn.cluster import kmeans_plusplus
    from sklearn.utils import check_random_state

    n_ts, sz, d = X.shape
    X2 = X.reshape((n_ts, -1))
    x_squared_norms = cdist(X2, np.zeros((1, sz * d)), metric='sqeuclidean').reshape((1, -1))
    rs = check_random_state(random_state)

    def assign(centers):
        dists = cdist(X2, centers.reshape((n_clusters, -1)), metric='euclidean')
        labels = dists.argmin(axis=1)
        empty = n_clusters - len(np.unique(labels))
        inertia = np.sum(dists[np.arange(n_ts), labels] ** 2) / n_ts
        return labels, empty, inertia

    best, min_inertia, labels = None, np.inf, None
    for _attempt in range(max_attempts):
        centers = kmeans_plusplus(X2, n_clusters, x_squared_norms=x_squared_norms.ravel(),
                                  random_state=rs)[0].reshape((-1, sz, d))
        # tslearn's _check_full_length: TimeSeriesResampler(sz) (np.interp onto the same grid)
        grid = np.linspace(0, 1, sz)
        centers = np.stack([np.stack([np.interp(grid, grid, c[:, di]) for di in range(d)], axis=1)
                            for c in centers])
        old_inertia = np.inf
        failed = False
        for _it in range(max_iter):
            labels, empty, inertia = assign(centers)
            if empty:
                failed = True
                break
            for k in range(n_clusters):
                centers[k] = np.average(X[labels == k], axis=0,
                                        weights=np.ones(int((labels == k).sum())))
            if np.abs(old_inertia - inertia) < tol:
                break
            old_inertia = inertia
        if not failed:
            if inertia < min_inertia:
                best, min_inertia = centers.copy(), inertia
            break
    if best is None:
        return labels, n_clusters - len(np.unique(labels))
    labels, empty, _ = assign(best)
    return labels, empty


def ts_clustering(df, n_clusters, random_state=0, max_iter=5):
    """
    Cluster the time series data in the input dataframe using the TS-KMeans algorithm.

    Each series is z-scored (mean 0, std 1; a constant series becomes all zeros) and the
    series are clustered by k-means with the Euclidean distance (k-means++ initialisation,
    at most ``max_iter`` iterations). This is what ``tslearn.clustering.TimeSeriesKMeans(
    metric='euclidean')`` does; it is computed here directly with numpy / scipy / sklearn
    (same labels as tslearn 0.6.3), so tslearn is not needed.

    Parameters:
        df (pandas.DataFrame): The input dataframe.
        n_clusters (int): The number of clusters to create.
        random_state (int): Seed of the k-means++ initialisation (reproducible results). Default 0.
        max_iter (int): Maximum k-means iterations. Default 5.

    Returns:
        pandas.Series named 'Cluster', indexed like ``df.index`` (the input is not modified).
        Rows with missing values are not clustered: they get NaN (and the Series is float),
        with a warning.

    Raises:
        ValueError: ``n_clusters`` is not a positive integer, or there are fewer usable
            series than clusters.
    """
    n_clusters = _check_positive_int(n_clusters, 'n_clusters')
    max_iter = _check_positive_int(max_iter, 'max_iter')
    _, data, _ = _prepare(df, np.zeros(df.shape[1]))
    finite = np.isfinite(data).all(axis=1)
    if not finite.all():
        warnings.warn(
            f"ts_clustering: {int((~finite).sum())} series contain missing / infinite values "
            "and are not clustered (Cluster = NaN).", UserWarning, stacklevel=2)
    n_ok = int(finite.sum())
    if n_ok < n_clusters:
        raise ValueError(
            f"ts_clustering: {n_ok} usable series for n_clusters={n_clusters}; "
            "use fewer clusters (or cluster_count=None to skip the clustering).")

    # TimeSeriesScalerMeanVariance(mu=0, std=1)
    X = data[finite][:, :, None]
    mean_t = np.nanmean(X, axis=1, keepdims=True)
    std_t = np.nanstd(X, axis=1, keepdims=True)
    std_t[std_t == 0.] = 1.
    X = (X - mean_t) * 1.0 / std_t + 0.0

    labels, n_empty = _ts_kmeans_euclidean(X, n_clusters, max_iter, random_state)
    if n_empty:
        warnings.warn(
            f"ts_clustering: {n_empty} of the {n_clusters} clusters are empty (too few distinct "
            "series for this many clusters).", UserWarning, stacklevel=2)

    if finite.all():
        return pd.Series(labels, index=df.index, name='Cluster')
    out = np.full(len(df.index), np.nan)
    out[finite] = labels
    return pd.Series(out, index=df.index, name='Cluster')
