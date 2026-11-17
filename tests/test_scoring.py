"""Regression tests for catbridge.scoring (compute_score, annotation, PCA importance)."""
import numpy as np
import pandas as pd
import pytest

from catbridge.scoring import (
    ASE_EXCLUDE,
    METRIC_DIRECTION,
    add_annotation_score,
    compute_score,
    score,
    top_important_features,
    trend_strength,
)


def make_df(n=40, seed=0, **extra):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'Name': [f'g{i}' for i in range(n)],
        'Granger': rng.uniform(0.001, 0.9, n),
        'Granger_padj': rng.uniform(0.01, 1.0, n),
        'CCM': rng.uniform(-1, 1, n),
        'DTW': rng.uniform(0.1, 5, n),
        'Pearson': rng.uniform(-1, 1, n),
        'log2FoldChange': rng.normal(0, 1, n),
        'Cluster': rng.integers(0, 3, n),
    })
    for k, v in extra.items():
        df[k] = v
    return df


def rank_of(res, name):
    return int(res.index[res['Name'] == name][0])


# ---------------------------------------------------------------- compute_score
@pytest.mark.parametrize('col', ['Granger', 'DTW', 'Pearson', 'CCM'])
@pytest.mark.parametrize('scaling', ['robust', 'minmax'])
def test_idempotent_and_input_untouched(col, scaling):
    df = make_df()
    before = df.copy()
    r1 = compute_score(df, col, scaling=scaling)
    assert df.equals(before)
    assert list(df.columns) == list(before.columns)
    r2 = compute_score(df, col, scaling=scaling)
    assert df.equals(before)
    pd.testing.assert_frame_equal(r1, r2)


def test_output_shape_and_columns():
    df = make_df()
    res = compute_score(df, 'Granger')
    assert list(res.columns) == ['Name', 'Granger', 'log2FoldChange', 'Cluster', 'Score', 'Granger_padj']
    assert res.index[0] == 1 and res.index[-1] == len(df)
    assert res['Score'].is_monotonic_decreasing
    assert res['Score'].between(0, 1).all()
    assert res['Score'].max() == pytest.approx(1.0)
    assert res['Score'].min() == pytest.approx(0.0)
    # the metric column keeps the ORIGINAL values (raw p for Granger)
    orig = df.set_index('Name')['Granger']
    assert np.allclose(res.set_index('Name')['Granger'], orig.loc[res['Name']].to_numpy())


def test_padj_only_if_present():
    df = make_df().drop(columns=['Granger_padj'])
    res = compute_score(df, 'Granger')
    assert list(res.columns) == ['Name', 'Granger', 'log2FoldChange', 'Cluster', 'Score']


def test_granger_smaller_p_ranks_higher():
    df = pd.DataFrame({'Name': list('abcd'), 'Granger': [0.04, 0.001, 0.5, 1e-8],
                       'log2FoldChange': [1.0] * 4})
    res = compute_score(df, 'Granger')
    assert list(res['Name']) == ['d', 'b', 'a', 'c']


def test_granger_resolves_small_p_values():
    # with the old 1-p, p=0.001 and p=1e-6 were indistinguishable; -log10 separates them
    df = pd.DataFrame({'Name': list('abc'), 'Granger': [1e-6, 1e-3, 0.9],
                       'log2FoldChange': [0.0, 0.0, 0.0]})
    res = compute_score(df, 'Granger', scaling='minmax')
    s = res.set_index('Name')['Score']
    assert s['a'] > s['b'] > s['c']
    assert s['b'] > 0.3  # 1e-3 sits half-way (in log space) between 1e-6 and 0.9


def test_dtw_smaller_distance_ranks_higher():
    df = pd.DataFrame({'Name': list('abcd'), 'DTW': [3.0, 0.5, 9.0, 1.0],
                       'log2FoldChange': [1.0] * 4})
    res = compute_score(df, 'DTW')
    assert list(res['Name']) == ['b', 'd', 'a', 'c']
    assert res.set_index('Name').loc['b', 'DTW'] == 0.5  # raw values kept


def test_pearson_larger_r_ranks_higher_and_negative_scores_low():
    df = pd.DataFrame({'Name': list('abcd'), 'Pearson': [0.2, 0.9, -0.8, 0.5],
                       'log2FoldChange': [1.0] * 4})
    res = compute_score(df, 'Pearson')
    assert list(res['Name']) == ['b', 'd', 'a', 'c']


def test_direction_table():
    assert METRIC_DIRECTION['Granger'] == 'pvalue'
    assert METRIC_DIRECTION['Granger_reverse'] == 'pvalue'
    assert METRIC_DIRECTION['DTW'] == 'lower'
    assert 'Pearson' not in METRIC_DIRECTION  # unknown -> 'higher'


def test_granger_reverse_uses_pvalue_direction():
    df = pd.DataFrame({'Name': list('abc'), 'Granger_reverse': [0.5, 0.001, 0.04],
                       'log2FoldChange': [1.0] * 3})
    assert list(compute_score(df, 'Granger_reverse')['Name']) == ['b', 'c', 'a']


@pytest.mark.parametrize('outlier', [-52.0, 500.0])
def test_robust_resists_fc_outlier_minmax_does_not(outlier):
    rng = np.random.default_rng(1)
    n = 1000  # realistic gene count; with n~100 a single outlier still shifts the interpolated 99% quantile
    fc = rng.normal(0, 1, n)
    fc[0] = outlier
    df = pd.DataFrame({'Name': [f'g{i}' for i in range(n)], 'CCM': rng.uniform(0.2, 0.8, n),
                       'log2FoldChange': fc})

    def norm_fc_std(scaling):
        # constant metric -> Score is a function of norm_fc only
        const = df.assign(CCM=0.5)
        r = compute_score(const, 'CCM', scaling=scaling).set_index('Name')['Score']
        return r.drop('g0').std()

    robust, minmax = norm_fc_std('robust'), norm_fc_std('minmax')
    assert robust > 0.1
    assert minmax < 0.1
    assert robust > 3 * minmax


@pytest.mark.parametrize('col', ['Granger', 'DTW', 'Pearson'])
def test_default_scaling_is_minmax(col):
    import inspect
    assert inspect.signature(compute_score).parameters['scaling'].default == 'minmax'
    df = make_df(n=200, seed=3)
    pd.testing.assert_frame_equal(compute_score(df, col), compute_score(df, col, scaling='minmax'))
    # and it really differs from 'robust' on data with a tail
    assert not compute_score(df, col).equals(compute_score(df, col, scaling='robust'))


def test_robust_ties_top_percent_minmax_keeps_resolution():
    """Records the trade-off: 'robust' clips each component at its 99% quantile, so all
    genes in the top 1% tie at 1.0; 'minmax' (default) keeps them apart."""
    n = 2000
    df = pd.DataFrame({'Name': [f'g{i}' for i in range(n)],
                       'Pearson': np.linspace(0.0, 1.0, n),   # all different, increasing
                       'log2FoldChange': 1.0})                # same FC for every gene -> no tie-break from FC
    top = [f'g{i}' for i in range(n - 10, n)]                 # 10 highest = top 0.5% (< 1%)

    robust = compute_score(df, 'Pearson', scaling='robust').set_index('Name')['Score']
    assert robust.loc[top].nunique() == 1                      # tied
    assert robust.loc[top].iloc[0] == robust.max()

    minmax = compute_score(df, 'Pearson', scaling='minmax').set_index('Name')['Score']
    assert minmax.loc[top].nunique() == len(top)               # all distinct
    assert minmax.loc[top].is_monotonic_increasing


def test_constant_columns_no_division_by_zero():
    df = pd.DataFrame({'Name': list('abc'), 'CCM': [0.5] * 3, 'log2FoldChange': [1.0] * 3})
    for scaling in ('robust', 'minmax'):
        res = compute_score(df, 'CCM', scaling=scaling)
        assert (res['Score'] == 0).all()


def test_nan_metric_not_nan_score_and_not_above_valid_row():
    df = pd.DataFrame({
        'Name': list('abcde'),
        'Pearson': [0.9, 0.1, np.nan, 0.5, 0.3],
        'log2FoldChange': [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    res = compute_score(df, 'Pearson').set_index('Name')
    assert not res['Score'].isna().any()
    valid = res.drop('c')['Score']
    assert res.loc['c', 'Score'] <= valid.min() + 1e-12


def test_nan_fc_gives_finite_score():
    df = pd.DataFrame({'Name': list('abc'), 'Pearson': [0.9, 0.5, 0.1],
                       'log2FoldChange': [np.nan, 1.0, 2.0]})
    res = compute_score(df, 'Pearson')
    assert not res['Score'].isna().any()
    res = compute_score(df.assign(Pearson=np.nan), 'Pearson')
    assert not res['Score'].isna().any()


def test_granger_nan_p():
    df = make_df(n=10)
    df.loc[3, 'Granger'] = np.nan
    res = compute_score(df, 'Granger')
    assert not res['Score'].isna().any()


def test_works_without_cluster_and_in_unit_interval():
    df = make_df().drop(columns=['Cluster'])
    res = compute_score(df, 'Granger')
    assert 'Cluster' not in res.columns
    assert res['Score'].between(0, 1).all()


def test_description_score_is_used_when_present():
    df = pd.DataFrame({'Name': list('ab'), 'Pearson': [0.5, 0.5], 'log2FoldChange': [0.0, 0.0],
                       'Description_Score': [0.2, 0.0]})
    res = compute_score(df, 'Pearson')
    assert list(res['Name']) == ['a', 'b']


def test_missing_columns_raise_keyerror():
    df = make_df()
    with pytest.raises(KeyError):
        compute_score(df, 'NoSuchColumn')
    with pytest.raises(KeyError):
        compute_score(df.drop(columns=['log2FoldChange']), 'Granger')
    with pytest.raises(ValueError):
        compute_score(df, 'Granger', scaling='zscore')


def test_empty_frame_and_string_column_name():
    df = make_df(n=0)
    res = compute_score(df, 'Granger')
    assert len(res) == 0


# ------------------------------------------------------------------ annotation
@pytest.mark.parametrize('desc,expected', [
    ('disease resistance protein', 0.0),
    ('DNA ligase, ATP-dependent', 0.2),
    ('Protein Kinase', 0.2),
    ('capsaicin synthase', 0.2),
    ('release factor', 0.0),
    ('hypothetical protein', 0.0),
    ('Phosphatase; putative', 0.2),
    ('Probable enzyme', 0.2),
    ('cell cycle phase protein', 0.0),
    ('Database entry', 0.0),
    ('ATP-dependent-kinase', 0.2),
    ('dnase I', 0.2),
    ('Serine/threonine kinases', 0.2),
    ('Diseases and releases', 0.0),
    ('nucleobase transporter', 0.0),
    ('anaphase-promoting complex', 0.0),
    (np.nan, 0.1),
    (None, 0.1),
    ('', 0.1),
    ('   ', 0.1),
    (12.0, 0.1),
])
def test_annotation_score(desc, expected):
    df = pd.DataFrame({'Name': ['x'], 'Description': [desc]}, dtype=object)
    out = add_annotation_score(df)
    assert out['Description_Score'].iloc[0] == pytest.approx(expected)


def test_annotation_does_not_mutate_and_requires_description():
    df = pd.DataFrame({'Name': ['x', 'y'], 'Description': ['kinase', np.nan]})
    before = df.copy()
    out = add_annotation_score(df)
    assert df.equals(before) and 'Description_Score' not in df.columns
    assert list(out['Description_Score']) == [0.2, 0.1]
    with pytest.raises(KeyError):
        add_annotation_score(df.drop(columns=['Description']))


def test_annotation_custom_keywords_and_max():
    df = pd.DataFrame({'Name': ['x'], 'Description': ['transferase and transporter']})
    out = add_annotation_score(df, keywords={'ase': 0.2, 'transporter': 0.5})
    assert out['Description_Score'].iloc[0] == 0.5


def test_ase_exclusion_list_contents():
    for w in ['disease', 'release', 'base', 'phase', 'case', 'increase', 'decrease', 'database',
              'purchase', 'phrase', 'erase', 'chase', 'cease', 'lease', 'please', 'vase']:
        assert w in ASE_EXCLUDE


# ----------------------------------------------------------------------- PCA
def test_top_important_features_variance_weighted():
    rng = np.random.default_rng(0)
    cols = [f't{i}' for i in range(8)]
    data = {'hi': 10 * rng.normal(size=8)}
    for i in range(12):
        data[f'const{i}'] = 5 + 1e-4 * rng.normal(size=8)
    df = pd.DataFrame(data, index=cols).T  # rows = genes, columns = time points
    out = top_important_features(df, 5)
    assert list(out.columns) == ['Gene', 'Importance']
    assert out['Gene'].iloc[0] == 'hi'
    full = top_important_features(df, 100)
    assert full.loc[full['Gene'] != 'hi', 'Importance'].max() < 1e-3 * full['Importance'].iloc[0]
    assert len(out) == 5
    assert out['Importance'].is_monotonic_decreasing


def test_top_important_features_does_not_swallow_errors():
    df = pd.DataFrame(np.ones((3, 4)), index=list('abc'))
    df.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        top_important_features(df, 2)


# ------------------------------------------------------------- legacy helpers
def test_legacy_score_and_trend_strength_do_not_mutate():
    df = pd.DataFrame({'Name': list('abc'), 'x': [1.0, 2.0, 3.0], 'y': [3.0, 1.0, 0.0]})
    before = df.copy()
    out = score(df)
    assert df.equals(before)
    assert list(out['Name'])[0] in 'abc' and 'Rank' in out.columns
    ts_in = pd.DataFrame([[1, 2, 3, 4], [4, 3, 2, 1]], index=['up', 'down'])
    ts_before = ts_in.copy()
    ts = trend_strength(ts_in)
    assert ts_in.equals(ts_before)
    assert list(ts.index) == ['up', 'down']
