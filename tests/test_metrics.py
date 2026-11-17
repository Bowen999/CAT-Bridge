"""Regression tests for catbridge.metrics (one test per confirmed bug / contract item)."""
import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import grangercausalitytests

from catbridge import metrics
from catbridge.metrics import (
    compute_granger, compute_reverse_granger, compute_ccm, compute_cca,
    compute_dtw, compute_ccf, compute_spearman, compute_pearson, ts_clustering,
)

ALL_COMPUTE = [
    compute_granger, compute_reverse_granger, compute_ccm, compute_cca,
    compute_dtw, compute_ccf, compute_spearman, compute_pearson,
]
IDS = [f.__name__ for f in ALL_COMPUTE]


# ---------------------------------------------------------------- fixtures
def _leader_follower(n=250, seed=0):
    """gene (iid) leads target by one step: target[t] = 0.9*gene[t-1] + noise."""
    rng = np.random.default_rng(seed)
    gene = rng.normal(size=n)
    target = np.empty(n)
    target[0] = rng.normal(scale=0.1)
    target[1:] = 0.9 * gene[:-1] + rng.normal(scale=0.1, size=n - 1)
    return gene, target


def _small_frame(n_genes=12, n_time=10, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(n_genes, n_time)).cumsum(axis=1),
        index=[f"g{i}" for i in range(n_genes)],
        columns=[f"t{j}" for j in range(n_time)],
    )
    target = rng.normal(size=n_time).cumsum()
    return df, target


def _messy_frame():
    """12 genes x 10 time points incl. a constant gene and a gene with NaN."""
    df, target = _small_frame()
    df.iloc[2] = 3.0
    df.iloc[5, 4] = np.nan
    return df, target



# ---------------------------------------------------------------- CCF direction
def test_ccf_gene_leads_target():
    gene, target = _leader_follower()
    df = pd.DataFrame([gene], index=["lead"])
    assert compute_ccf(df, target, lag=1)["CCF"].iloc[0] > 0.8
    assert abs(compute_ccf(df, target, lag=-1)["CCF"].iloc[0]) < 0.2


def test_ccf_target_leads_gene():
    target, gene = _leader_follower()  # roles swapped: target leads gene
    df = pd.DataFrame([gene], index=["follow"])
    assert abs(compute_ccf(df, target, lag=1)["CCF"].iloc[0]) < 0.2
    assert compute_ccf(df, target, lag=-1)["CCF"].iloc[0] > 0.8


def test_ccf_matches_definition_and_lag_range():
    gene, target = _leader_follower(n=60, seed=3)
    df = pd.DataFrame([gene], index=["g"])
    # lag=0 is the contemporaneous correlation (adjusted=True => close to Pearson)
    assert compute_ccf(df, target, lag=0)["CCF"].iloc[0] == pytest.approx(
        np.corrcoef(gene, target)[0, 1], abs=1e-6)
    # lag=+-k agrees (loosely; statsmodels uses global mean/std) with corr(gene[t], target[t+k])
    for k in (1, 2, -1, -2):
        a, b = (gene[:-k], target[k:]) if k > 0 else (gene[-k:], target[:k])
        assert compute_ccf(df, target, lag=k)["CCF"].iloc[0] == pytest.approx(
            np.corrcoef(a, b)[0, 1], abs=0.1)
    n = len(target)
    for bad in (n, n + 5, -n, -n - 5):
        assert np.isnan(compute_ccf(df, target, lag=bad)["CCF"].iloc[0])


# ---------------------------------------------------------------- Granger
def test_granger_direction_and_pvalue_semantics():
    gene, target = _leader_follower()
    df = pd.DataFrame([gene], index=["g"])
    fwd = compute_granger(df, target, maxlag=1)
    rev = compute_reverse_granger(df, target, maxlag=1)
    assert list(fwd.columns) == ["Name", "Granger", "Granger_padj"]
    assert list(rev.columns) == ["Name", "Granger_reverse", "Granger_reverse_padj"]
    p_fwd, p_rev = fwd["Granger"].iloc[0], rev["Granger_reverse"].iloc[0]
    assert 0 <= p_fwd < 1e-6          # gene -> target is real
    assert 0.05 < p_rev <= 1          # target -> gene is not (and is a raw p, not 1-p)

    # raw p-values straight from statsmodels (column order [effect, cause])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_f = grangercausalitytests(np.column_stack([target, gene]), 1, verbose=False)[1][0]["ssr_ftest"][1]
        ref_r = grangercausalitytests(np.column_stack([gene, target]), 1, verbose=False)[1][0]["ssr_ftest"][1]
    assert p_fwd == pytest.approx(ref_f)
    assert p_rev == pytest.approx(ref_r)


def test_granger_reverse_direction_when_target_leads():
    target, gene = _leader_follower()
    df = pd.DataFrame([gene], index=["g"])
    assert compute_granger(df, target)["Granger"].iloc[0] > 0.05
    assert compute_reverse_granger(df, target)["Granger_reverse"].iloc[0] < 1e-6


@pytest.mark.parametrize("func,col", [(compute_granger, "Granger"),
                                      (compute_reverse_granger, "Granger_reverse"),
                                      (compute_spearman, "Spearman"),
                                      (compute_pearson, "Pearson")])
def test_padj_is_bh_adjusted(func, col):
    rng = np.random.default_rng(7)
    n_time = 30
    df = pd.DataFrame(rng.normal(size=(25, n_time)), index=[f"g{i}" for i in range(25)])
    target = rng.normal(size=n_time)
    df.iloc[0] = target * 2 + rng.normal(scale=0.2, size=n_time)  # one strongly related gene
    df.iloc[4] = 1.0                                              # constant -> NaN
    df.iloc[9, 3] = np.nan                                        # NaN -> NaN
    res = func(df, target)
    p = res[col + ("_p" if col in ("Spearman", "Pearson") else "")].to_numpy()
    padj = res[col + "_padj"].to_numpy()

    # NaN rows have NaN in the adjusted column too, and nowhere else
    assert np.array_equal(np.isnan(p), np.isnan(padj))
    assert np.isnan(p[4]) and np.isnan(p[9])
    ok = ~np.isnan(p)
    assert ok.sum() >= 20
    assert np.all(padj[ok] >= p[ok] - 1e-12)
    assert np.all(padj[ok] <= 1 + 1e-12)
    # monotone: ordering by p never contradicts ordering by padj
    order = np.argsort(p[ok], kind="stable")
    assert np.all(np.diff(padj[ok][order]) >= -1e-12)
    # matches an independent BH computation over the non-NaN p-values
    m = ok.sum()
    ps = np.sort(p[ok])
    bh = np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1)
    assert np.allclose(np.sort(padj[ok]), np.sort(bh))


# ---------------------------------------------------------------- rows kept, order kept
@pytest.mark.parametrize("func", ALL_COMPUTE, ids=IDS)
def test_no_rows_dropped_or_reordered(func):
    df, target = _messy_frame()
    # reversed alphabetical order makes accidental sorting visible
    df = df.sort_index(ascending=False)
    res = func(df, target)
    assert len(res) == len(df)
    assert list(res["Name"]) == list(df.index)
    assert res.columns[0] == "Name"
    # the constant (g2) and NaN (g5) genes are present as NaN in the value column -- for EVERY
    # metric (incl. DTW and CCA: a constant series carries no information)
    value_col = res.columns[1]
    assert np.isnan(res.set_index("Name").loc["g2", value_col])
    assert np.isnan(res.set_index("Name").loc["g5", value_col])
    # and a normal gene is computed (Granger p can legitimately be NaN only for infeasible tests)
    if func not in (compute_granger, compute_reverse_granger):
        assert not np.isnan(res.set_index("Name").loc["g0", value_col])


def test_duplicate_gene_names_keep_all_rows():
    df, target = _small_frame(6, 10)
    df.index = ["a", "a", "b", "b", "c", "c"]
    for func in ALL_COMPUTE:
        assert list(func(df, target)["Name"]) == list(df.index)


def test_constant_target_gives_nan_not_error():
    df, _ = _small_frame()
    target = np.ones(df.shape[1])
    for func in (compute_granger, compute_reverse_granger, compute_ccf,
                 compute_spearman, compute_pearson, compute_dtw, compute_cca):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = func(df, target)
        assert len(res) == len(df)
        assert res.iloc[:, 1].isna().all()


# ---------------------------------------------------------------- CCM
def test_ccm_columns_and_values():
    df, target = _small_frame(8, 12)
    res = compute_ccm(df, target, E=3, tau=1)
    assert list(res.columns) == ["Name", "CCM", "CCM_p"]
    assert list(res["Name"]) == list(df.index)       # not sorted by CCM
    assert res["CCM"].between(-1, 1).all()
    assert res["CCM_p"].between(0, 1).all()


@pytest.mark.parametrize("n_time", [4, 3, 2, 1])
def test_ccm_too_short_returns_nan_with_warning(n_time):
    df, target = _small_frame(5, n_time)
    with pytest.warns(UserWarning, match="CCM"):
        res = compute_ccm(df, target, E=3, tau=1)
    assert len(res) == 5 and list(res["Name"]) == list(df.index)
    assert res[["CCM", "CCM_p"]].isna().all().all()


def test_ccm_length_boundary():
    # N - (E-1)*tau must be >= E+2 : E=3, tau=1 -> N >= 7 ; N=6 is too short
    df, target = _small_frame(4, 6)
    with pytest.warns(UserWarning):
        assert compute_ccm(df, target, E=3, tau=1)["CCM"].isna().all()
    df, target = _small_frame(4, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert compute_ccm(df, target, E=3, tau=1)["CCM"].notna().all()
    # with tau=2, N=7 gives 7-4=3 < 5 points
    with pytest.warns(UserWarning):
        assert compute_ccm(df, target, E=3, tau=2)["CCM"].isna().all()


def test_ccm_nan_row_is_nan_and_silent(capsys):
    df, target = _messy_frame()
    res = compute_ccm(df, target).set_index("Name")
    assert np.isnan(res.loc["g5", "CCM"]) and np.isnan(res.loc["g5", "CCM_p"])
    assert capsys.readouterr().out == ""            # old version print()ed the data


# ---------------------------------------------------------------- inputs not modified
@pytest.mark.parametrize("func", ALL_COMPUTE, ids=IDS)
def test_compute_does_not_modify_inputs(func):
    df, target = _messy_frame()
    df_copy = df.copy(deep=True)
    for tgt in (list(target), np.array(target), pd.Series(target)):
        tgt_copy = tgt.copy()
        func(df, tgt)
        assert df.equals(df_copy)
        assert list(df.columns) == list(df_copy.columns) and list(df.index) == list(df_copy.index)
        assert np.array_equal(np.asarray(tgt), np.asarray(tgt_copy), equal_nan=True)


def test_ts_clustering_does_not_modify_input():
    df, _ = _small_frame(10, 8)
    df_copy = df.copy(deep=True)
    out = ts_clustering(df, 3)
    assert df.equals(df_copy)
    assert "Cluster" not in df.columns
    # same type / shape as before: Series named 'Cluster', indexed like df
    assert isinstance(out, pd.Series) and out.name == "Cluster"
    assert out.index.equals(df.index) and len(out) == len(df)


def test_ts_clustering_reproducible_and_kwargs():
    df, _ = _small_frame(20, 8, seed=5)
    a = ts_clustering(df, 3)
    b = ts_clustering(df, 3, random_state=0, max_iter=5)
    assert a.equals(b)
    c = ts_clustering(df, 3, random_state=1, max_iter=50)
    assert len(c) == len(df)


def test_ts_clustering_output_merges_on_name():
    """downstream merge_dataframes does ``left.merge(series, on='Name', how='outer')``"""
    df, target = _small_frame(10, 8)
    df.index.name = "Name"
    merged = compute_pearson(df, target).merge(ts_clustering(df, 2), on="Name", how="outer")
    assert len(merged) == len(df) and "Cluster" in merged.columns


# ---------------------------------------------------------------- length check
@pytest.mark.parametrize("func", ALL_COMPUTE, ids=IDS)
def test_length_mismatch_raises_valueerror(func):
    df, target = _small_frame(5, 10)
    with pytest.raises(ValueError, match=r"\b9\b.*\b10\b|\b10\b.*\b9\b"):
        func(df, target[:-1])
    with pytest.raises(ValueError):
        func(df, list(target) + [0.0])


@pytest.mark.parametrize("func", ALL_COMPUTE, ids=IDS)
def test_target_container_types_equivalent(func):
    df, target = _small_frame(6, 10)
    ref = func(df, np.array(target))
    for tgt in (list(target), pd.Series(target), pd.Series(target, index=list("abcdefghij"))):
        pd.testing.assert_frame_equal(func(df, tgt), ref)


# ---------------------------------------------------------------- DTW / CCA / correlations
def test_dtw_identical_series_is_zero():
    df, target = _small_frame(5, 10)
    df.iloc[1] = target
    res = compute_dtw(df, target)
    assert res["DTW"].iloc[1] == 0
    assert (res["DTW"] >= 0).all()
    # distance semantics: a dissimilar series is farther than an identical one
    df.iloc[2] = np.linspace(1, 0, len(target)) if target[0] < target[-1] else np.linspace(0, 1, len(target))
    res = compute_dtw(df, target)
    assert res["DTW"].iloc[2] > res["DTW"].iloc[1] == 0


def test_cca_is_absolute_pearson_for_single_gene():
    df, target = _small_frame(10, 12, seed=9)
    df.iloc[0] = -np.asarray(target) + np.random.default_rng(0).normal(scale=0.1, size=12)
    cca = compute_cca(df, target)["CCA"].to_numpy()
    pear = compute_pearson(df, target)["Pearson"].to_numpy()
    assert pear[0] < 0                         # sign is kept by Pearson ...
    assert np.allclose(cca, np.abs(pear), atol=1e-6)   # ... and lost in CCA


@pytest.mark.parametrize("func,col", [(compute_spearman, "Spearman"), (compute_pearson, "Pearson")])
def test_correlations_keep_sign_and_columns(func, col):
    df, target = _small_frame(6, 10)
    df.iloc[0] = np.asarray(target) * 2 + 1
    df.iloc[1] = -np.asarray(target)
    df.iloc[2] = 4.0
    res = func(df, target)
    assert list(res.columns) == ["Name", col, col + "_p", col + "_padj"]
    assert res[col].iloc[0] == pytest.approx(1.0)
    assert res[col].iloc[1] == pytest.approx(-1.0)
    assert res[[col, col + "_p", col + "_padj"]].iloc[2].isna().all()


# ---------------------------------------------------------------- misc API
def test_granger_expected_errors_only_and_short_series():
    # too few time points -> NaN (ValueError inside statsmodels), not an exception (but a warning)
    df, target = _small_frame(3, 3)
    with pytest.warns(UserWarning, match="Granger"):
        res = compute_granger(df, target)
    assert res[["Granger", "Granger_padj"]].isna().all().all()
    assert len(res) == 3


def test_granger_invalid_maxlag():
    df, target = _small_frame(3, 10)
    with pytest.raises(ValueError):
        compute_granger(df, target, maxlag=0)


def test_granger_list_unchanged_semantics():
    gene, target = _leader_follower(n=100)
    p = metrics.granger_list(target, gene, 2)   # does gene (B) Granger-cause target (A)?
    assert set(p) == {1, 2} and p[1] < 1e-6


# ---------------------------------------------------------------- DTW / CCA: constant series
def test_dtw_constant_gene_is_nan_including_zeros():
    rng = np.random.default_rng(0)
    target = np.array([1, 1, 1, 1, 9, 1, 1, 1, 1, 1.0])          # spiky target
    df = pd.DataFrame(rng.normal(5, 1, (20, 10)), index=[f"g{i}" for i in range(20)])
    df.loc["const"] = 5.0
    df.loc["zeros"] = 0.0
    res = compute_dtw(df, target).set_index("Name")["DTW"]
    assert np.isnan(res["const"]) and np.isnan(res["zeros"])
    assert res.drop(["const", "zeros"]).notna().all()
    # the non-constant genes are unchanged by the constant ones being present
    only = compute_dtw(df.drop(["const", "zeros"]), target).set_index("Name")["DTW"]
    pd.testing.assert_series_equal(res.drop(["const", "zeros"]), only)


def test_dtw_constant_target_all_nan():
    df, _ = _small_frame()
    for const in (np.zeros(df.shape[1]), np.full(df.shape[1], 3.0)):
        res = compute_dtw(df, const)
        assert len(res) == len(df) and res["DTW"].isna().all()


def test_cca_constant_gene_and_constant_target_are_nan():
    df, target = _small_frame()
    df.iloc[1] = 0.0
    df.iloc[2] = 7.0
    res = compute_cca(df, target).set_index("Name")["CCA"]
    assert np.isnan(res["g1"]) and np.isnan(res["g2"])
    assert res.drop(["g1", "g2"]).notna().all()
    assert compute_cca(df, np.ones(df.shape[1]))["CCA"].isna().all()


# ---------------------------------------------------------------- CCA n_components
@pytest.mark.parametrize("bad", [2, 3, 0])
def test_cca_n_components_other_than_one_raises(bad):
    df, target = _small_frame()
    with pytest.raises(ValueError, match="n_components"):
        compute_cca(df, target, n_components=bad)


def test_cca_n_components_one_still_works():
    df, target = _small_frame()
    assert compute_cca(df, target, n_components=1)["CCA"].notna().all()
    assert compute_cca(df, target)["CCA"].notna().all()


# ---------------------------------------------------------------- Granger: lag too large
def test_granger_lag_too_large_warns_with_max_lag_and_still_works_at_max():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(15, 9)), index=[f"g{i}" for i in range(15)])
    target = rng.normal(size=9)
    # 9 time points: statsmodels needs nobs > 3*maxlag + 1 -> largest usable lag is 2
    for func in (compute_granger, compute_reverse_granger):
        with pytest.warns(UserWarning, match=r"9 time points.*maxlag=3.*largest usable lag is 2"):
            res = func(df, target, maxlag=3)
        assert res.iloc[:, 1].isna().all() and len(res) == 15
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ok = func(df, target, maxlag=2)
        assert ok.iloc[:, 1].notna().all()


def test_granger_max_lag_formula_matches_statsmodels():
    rng = np.random.default_rng(2)
    for n in range(4, 16):
        best = 0
        for lag in range(1, 8):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    grangercausalitytests(rng.normal(size=(n, 2)), lag, verbose=False)
                best = lag
            except ValueError:
                pass
        assert metrics._granger_max_lag(n) == best, n


def test_granger_no_warning_when_only_constant_genes_are_nan():
    df, target = _small_frame(4, 10)
    df.iloc[1] = 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = compute_granger(df, target, maxlag=1)
    assert np.isnan(res["Granger"].iloc[1]) and res["Granger"].drop(1).notna().all()
    # all genes constant -> nothing testable -> no warning (not a lag problem)
    flat = pd.DataFrame(np.ones((3, 10)), index=list("abc"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compute_granger(flat, target, maxlag=1)["Granger"].isna().all()
