"""The batched (whole-matrix) metrics give the values of the reference implementations they
replaced: statsmodels grangercausalitytests / ccf, causal_ccm, fastdtw, sklearn CCA, scipy
spearmanr / pearsonr, tslearn TimeSeriesKMeans, statsmodels multipletests.

Each test is skipped when its reference library is not installed (or does not run here).
The data include the awkward cases: constant series, an affine copy of the target (perfect
fit), ties, runs of zeros, NaN, very different magnitudes.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from catbridge import metrics as M


def _frame(n_genes, n_time, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_genes, n_time)) * rng.uniform(0.1, 1e4, size=(n_genes, 1))
    target = rng.normal(size=n_time).cumsum() + 5
    X[0] = target * 3 + 1                         # affine copy of the target
    X[1] = 2.0                                    # constant
    X[2] = np.round(X[2] / X[2].std())            # ties
    X[3, : n_time // 2] = 0.0                     # run of zeros
    X[4] = np.arange(n_time, dtype=float)         # linear trend
    X[5] = rng.integers(0, 3, n_time)             # many ties
    X[6] = X[7]                                   # duplicated gene
    X[8, n_time // 2] = np.nan                    # missing value
    X[9] = 1e-8 * X[9]                            # tiny values
    X[10] = np.roll(target, 1)                    # target shifted by one step
    X[11] = rng.poisson(3, n_time)                # counts
    df = pd.DataFrame(X, index=[f"g{i}" for i in range(n_genes)])
    return df, target


LENGTHS = [3, 5, 7, 8, 12, 20, 47]
FRAMES = [_frame(30, n, seed) for seed, n in enumerate(LENGTHS)]
IDS = [f"n{n}" for n in LENGTHS]


def _assert_same(new, ref, rtol=1e-9, atol=1e-12):
    new, ref = np.asarray(new, float), np.asarray(ref, float)
    assert np.array_equal(np.isnan(new), np.isnan(ref)), (np.flatnonzero(np.isnan(new) != np.isnan(ref)))
    np.testing.assert_allclose(new, ref, rtol=rtol, atol=atol, equal_nan=True)


def _rows(df):
    return df.to_numpy(dtype=float)


# ------------------------------------------------------------------ Granger
@pytest.mark.parametrize("df,target", FRAMES, ids=IDS)
@pytest.mark.parametrize("maxlag", [1, 2, 3])
@pytest.mark.parametrize("direction", ['forward', 'reverse'])
def test_granger_matches_statsmodels(df, target, maxlag, direction):
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tools.sm_exceptions import InfeasibleTestError
    y = np.asarray(target, float)
    ref = []
    for row in _rows(df):
        cause, effect = (row, y) if direction == 'forward' else (y, row)
        if (np.isnan(row).any() or M._is_constant(row)):
            ref.append(np.nan)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = grangercausalitytests(np.column_stack([effect, cause]), maxlag=maxlag,
                                            verbose=False)
            ref.append(res[maxlag][0]['ssr_ftest'][1])
        except (ValueError, np.linalg.LinAlgError, InfeasibleTestError):
            ref.append(np.nan)
    func = M.compute_granger if direction == 'forward' else M.compute_reverse_granger
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = func(df, target, maxlag=maxlag).iloc[:, 1]
    _assert_same(got, ref)


def test_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(0)
    p = rng.uniform(size=500) ** 3
    p[::7] = p[3]                  # ties
    p[::11] = np.nan
    ok = ~np.isnan(p)
    ref = np.full(p.shape, np.nan)
    ref[ok] = multipletests(p[ok], method='fdr_bh')[1]
    np.testing.assert_array_equal(M._bh_adjust(p), ref)


# ------------------------------------------------------------------ CCM
@pytest.mark.parametrize("df,target", FRAMES[2:], ids=IDS[2:])
@pytest.mark.parametrize("E,tau", [(2, 1), (3, 1), (3, 2), (4, 1)])
def test_ccm_matches_causal_ccm(df, target, E, tau):
    ccm_mod = pytest.importorskip("causal_ccm.causal_ccm")
    y = np.asarray(target, float)
    if len(y) - (E - 1) * tau < E + 2:
        pytest.skip("series too short for this E / tau")
    ref_r, ref_p = [], []
    for row in _rows(df):
        if np.isnan(row).any():
            ref_r.append(np.nan); ref_p.append(np.nan)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r, p = ccm_mod.ccm(row, y, tau, E).causality()
        ref_r.append(r); ref_p.append(p)
    got = M.compute_ccm(df, target, E=E, tau=tau)
    _assert_same(got['CCM'], ref_r, rtol=1e-12, atol=1e-14)
    _assert_same(got['CCM_p'], ref_p, rtol=1e-12, atol=1e-14)


# ------------------------------------------------------------------ DTW
def _scaled(v):
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler().fit_transform(np.asarray(v, float).reshape(-1, 1))


@pytest.mark.parametrize("df,target", FRAMES, ids=IDS)
def test_dtw_default_matches_fastdtw(df, target):
    fastdtw = pytest.importorskip("fastdtw")
    b = _scaled(target)
    ref = []
    for row in _rows(df):
        if np.isnan(row).any() or M._is_constant(row):
            ref.append(np.nan)
            continue
        ref.append(fastdtw.fastdtw(_scaled(row), b)[0])
    _assert_same(M.compute_dtw(df, target)['DTW'], ref, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("n", list(range(1, 41)) + [64, 65, 100])
def test_fastdtw_replica_any_length_and_radius(n):
    fastdtw = pytest.importorskip("fastdtw")
    rng = np.random.default_rng(n)
    A = rng.normal(size=(12, n))
    A[1] = A[0]                       # identical rows
    A[2] = np.round(A[2])            # ties
    b = rng.normal(size=n)
    for radius in (1, 2):
        ref = [fastdtw.fastdtw(a, b, radius=radius)[0] for a in A]
        np.testing.assert_allclose(M._fastdtw_batch(A, b, radius=radius), ref, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("df,target", FRAMES, ids=IDS)
def test_dtw_exact_matches_full_dtw_and_bounds_fastdtw(df, target):
    fastdtw = pytest.importorskip("fastdtw")
    b = _scaled(target)
    ref = []
    for row in _rows(df):
        if np.isnan(row).any() or M._is_constant(row):
            ref.append(np.nan)
            continue
        ref.append(fastdtw.dtw(_scaled(row), b)[0])       # exact DTW of the fastdtw package
    exact = M.compute_dtw(df, target, method='exact')['DTW']
    _assert_same(exact, ref, rtol=1e-13, atol=1e-13)
    approx = M.compute_dtw(df, target)['DTW']
    assert (exact.to_numpy()[~np.isnan(exact)] <= approx.to_numpy()[~np.isnan(approx)] + 1e-12).all()


def test_dtw_method_is_checked():
    df, target = FRAMES[2]
    with pytest.raises(ValueError, match="method"):
        M.compute_dtw(df, target, method='fast')


# ------------------------------------------------------------------ CCA / CCF / correlations
@pytest.mark.parametrize("df,target", FRAMES, ids=IDS)
def test_cca_matches_sklearn(df, target):
    from sklearn.cross_decomposition import CCA
    y = np.asarray(target, float).reshape(-1, 1)
    ref = []
    for row in _rows(df):
        if np.isnan(row).any() or M._is_constant(row):
            ref.append(np.nan)
            continue
        g = row.reshape(-1, 1)
        cca = CCA(n_components=1).fit(g, y)
        a, b = cca.transform(g, y)
        ref.append(np.corrcoef(a.T, b.T)[0, 1])
    _assert_same(M.compute_cca(df, target)['CCA'], ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("df,target", FRAMES, ids=IDS)
@pytest.mark.parametrize("lag", [-2, -1, 0, 1, 2])
def test_ccf_matches_statsmodels(df, target, lag):
    from statsmodels.tsa.stattools import ccf
    y = np.asarray(target, float)
    ref = []
    for row in _rows(df):
        if np.isnan(row).any() or M._is_constant(row):
            ref.append(np.nan)
            continue
        cc = ccf(y, row) if lag >= 0 else ccf(row, y)
        ref.append(cc[abs(lag)] if abs(lag) < len(cc) else np.nan)
    _assert_same(M.compute_ccf(df, target, lag=lag)['CCF'], ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("df,target", FRAMES, ids=IDS)
@pytest.mark.parametrize("kind", ['spearman', 'pearson'])
def test_correlations_match_scipy(df, target, kind):
    from scipy.stats import pearsonr, spearmanr
    func = spearmanr if kind == 'spearman' else pearsonr
    y = np.asarray(target, float)
    ref_r, ref_p = [], []
    for row in _rows(df):
        if np.isnan(row).any() or M._is_constant(row):
            ref_r.append(np.nan); ref_p.append(np.nan)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r, p = func(row, y)
        ref_r.append(r); ref_p.append(p)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)     # no stray numpy warnings
        got = (M.compute_spearman if kind == 'spearman' else M.compute_pearson)(df, target)
    col = 'Spearman' if kind == 'spearman' else 'Pearson'
    # same numbers (scipy's own vectorised code / the same float operations)
    np.testing.assert_array_equal(got[col].to_numpy(), np.asarray(ref_r, float))
    np.testing.assert_array_equal(got[col + '_p'].to_numpy(), np.asarray(ref_p, float))


# ------------------------------------------------------------------ clustering
def _tslearn_labels(df, k, random_state, max_iter):
    pytest.importorskip("tslearn")
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    try:
        data = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(df.values)
        km = TimeSeriesKMeans(n_clusters=k, metric="euclidean", max_iter=max_iter,
                              random_state=random_state).fit(data)
    except TypeError as e:  # tslearn 0.6.3 with scikit-learn >= 1.6 (force_all_finite)
        pytest.skip(f"tslearn does not run in this environment: {e}")
    return km.labels_


@pytest.mark.parametrize("k,random_state,max_iter", [(2, 0, 5), (3, 0, 5), (5, 1, 5), (8, 0, 50)])
@pytest.mark.parametrize("n_time", [6, 7, 20])
def test_ts_clustering_matches_tslearn(k, random_state, max_iter, n_time):
    rng = np.random.default_rng(n_time + k)
    X = np.vstack([rng.normal(size=(40, n_time)).cumsum(axis=1),
                   np.zeros((3, n_time)), np.full((2, n_time), 5.0)])
    X[10] = X[11]
    df = pd.DataFrame(X, index=[f"g{i}" for i in range(len(X))])
    ref = _tslearn_labels(df, k, random_state, max_iter)
    got = M.ts_clustering(df, k, random_state=random_state, max_iter=max_iter)
    np.testing.assert_array_equal(got.to_numpy(), ref)
