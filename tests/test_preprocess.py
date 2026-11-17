"""Regression tests for catbridge.preprocess."""
import warnings

import numpy as np
import pandas as pd
import pytest

from catbridge.preprocess import (
    merge_dataframes, normalize, repeat_aggregation_max, repeat_aggregation_mean,
    scale_column, scale_df, scaling,
)


def _design(groups, reps=3):
    idx, grp = [], []
    for g in groups:
        for r in range(1, reps + 1):
            idx.append(f'{g}_{r}')
            grp.append(g)
    return pd.DataFrame({'group': grp}, index=idx)


def _data(design, n_rows=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(1, 100, (n_rows, len(design))), columns=design.index,
                        index=[f'r{i}' for i in range(n_rows)])


# ------------------------------------------------------------ repeat_aggregation_mean
@pytest.mark.parametrize("groups", [
    ['day0', 'day7', 'day16', 'day30'],     # alphabetical order would be day0, day16, day30, day7
    ['day00', 'day07', 'day16', 'day30'],   # zero padded: alphabetical == time order
    ['t9', 't10', 't2'],
])
def test_repeat_aggregation_mean_keeps_time_order_and_means(groups):
    design = _design(groups)
    df = _data(design)
    df_before, design_before = df.copy(), design.copy()
    res = repeat_aggregation_mean(df, design)
    assert list(res.columns) == groups
    assert list(res.index) == list(df.index)
    for g in groups:
        expected = df[design.index[design['group'] == g]].mean(axis=1)
        np.testing.assert_allclose(res[g].to_numpy(), expected.to_numpy())
    pd.testing.assert_frame_equal(df, df_before)
    pd.testing.assert_frame_equal(design, design_before)


def test_repeat_aggregation_mean_order_follows_df_columns_not_design():
    """Time order = column order of the data file; the design row order is irrelevant."""
    groups = ['day0', 'day7', 'day16', 'day30']
    design = _design(groups)
    df = _data(design)
    for shuffled in (design.sort_index(),                      # alphabetical by sample name
                     design.iloc[::-1],                        # reversed
                     design.sample(frac=1, random_state=0)):   # random
        res = repeat_aggregation_mean(df, shuffled)
        assert list(res.columns) == groups
        expected = df[design.index[design['group'] == 'day16']].mean(axis=1)
        np.testing.assert_allclose(res['day16'].to_numpy(), expected.to_numpy())
    # df columns reversed -> the output follows the DATA (reversed time order)
    res = repeat_aggregation_mean(df[design.index[::-1]], design)
    assert list(res.columns) == groups[::-1]


def test_repeat_aggregation_mean_and_max_give_the_same_column_order():
    groups = ['d0', 'd7', 'd16', 'd30']
    # base names equal the group names so that design['<base>_1'] -> group is valid for max too
    design = _design(groups)
    df = _data(design, n_rows=5, seed=2)
    variants = {
        'as in data': design,
        'sorted by sample name': design.sort_index(),
        'reversed rows': design.iloc[::-1],
        'shuffled rows': design.sample(frac=1, random_state=3),
    }
    for name, d in variants.items():
        mean_cols = list(repeat_aggregation_mean(df, d).columns)
        max_cols = list(repeat_aggregation_max(df, d).columns)
        assert mean_cols == max_cols == groups, name
    # with a reversed data file both follow the data
    rev = df[design.index[::-1]]
    assert (list(repeat_aggregation_mean(rev, design).columns)
            == list(repeat_aggregation_max(rev, design).columns) == groups[::-1])


def test_repeat_aggregation_mean_subset_of_columns_and_nan():
    groups = ['day0', 'day7', 'day16']
    design = _design(groups)
    df = _data(design)
    df.iloc[0, 0] = np.nan
    sub = df[[c for c in df.columns if not c.startswith('day7')]]
    res = repeat_aggregation_mean(sub, design)
    assert list(res.columns) == ['day0', 'day16']   # absent group dropped, order kept
    assert res.loc['r0', 'day0'] == pytest.approx(df.loc['r0', ['day0_2', 'day0_3']].mean())


def test_repeat_aggregation_mean_unknown_column_keyerror():
    design = _design(['a', 'b'])
    df = _data(design)
    df['mystery_1'] = 1.0
    with pytest.raises(KeyError, match="mystery_1"):
        repeat_aggregation_mean(df, design)


# ------------------------------------------------------------ repeat_aggregation_max
def _max_reference(df, design):
    """Straight re-implementation of the author's algorithm (per-row loop)."""
    groups = list(dict.fromkeys(design['group']))
    reps = len(design) // len(groups)
    base = list(dict.fromkeys(c.rsplit('_', 1)[0] for c in df.columns))
    out = pd.DataFrame(index=df.index, columns=base, dtype=float)
    for idx in df.index:
        means = {s: df.filter(regex=f'_{s}$').loc[idx].mean() for s in range(1, reps + 1)}
        best = max(means, key=means.get)
        for b in base:
            out.loc[idx, b] = df.loc[idx, f'{b}_{best}']
    out.columns = [design['group'].to_dict().get(f'{b}_1', f'{b}_1') for b in base]
    return out


def test_repeat_aggregation_max_matches_reference_and_keeps_time_order():
    groups = ['d0', 'd7', 'd16', 'd30']
    # sample names like "00D_1" -> base "00D"; design maps "<base>_1" -> group is only valid for
    # the first replicate, so use base == group-ish names
    design = pd.DataFrame({'group': [f'g{b}' for b in ['0', '7', '16', '30'] for _ in range(3)]},
                          index=[f'{b}D_{r}' for b in ['0', '7', '16', '30'] for r in (1, 2, 3)])
    df = _data(design, n_rows=8, seed=4)
    df_before = df.copy()
    res = repeat_aggregation_max(df, design)
    assert list(res.columns) == ['g0', 'g7', 'g16', 'g30']
    ref = _max_reference(df, design)
    pd.testing.assert_frame_equal(res, ref, check_dtype=False, check_names=False)
    pd.testing.assert_frame_equal(df, df_before)


def test_repeat_aggregation_max_all_nan_row_does_not_crash():
    design = pd.DataFrame({'group': ['a'] * 2 + ['b'] * 2},
                          index=['a_1', 'a_2', 'b_1', 'b_2'])
    df = pd.DataFrame([[np.nan] * 4, [1, 2, 3, 4]], index=['x', 'y'], columns=design.index)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        res = repeat_aggregation_max(df, design)
    assert res.loc['x'].isna().all()
    assert res.loc['y'].tolist() == [2, 4]   # replicate _2 has the larger mean


# ------------------------------------------------------------ scaling / normalize / scale_*
def test_scaling_returns_value_and_keeps_labels():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [10.0, 20.0, 40.0]}, index=['x', 'y', 'z'])
    before = df.copy()
    mm = scaling(df, 'min-max')
    assert mm is not None
    assert isinstance(mm, pd.DataFrame)
    assert list(mm.index) == ['x', 'y', 'z'] and list(mm.columns) == ['a', 'b']
    assert mm['a'].tolist() == [0.0, 0.5, 1.0]
    par = scaling(df, 'pareto')
    assert par is not None
    np.testing.assert_allclose(par['a'].to_numpy(), (df['a'] - 2.0) / np.sqrt(1.0))
    pd.testing.assert_frame_equal(df, before)


def test_normalize_methods_and_no_mutation():
    df = pd.DataFrame({'a': [0.0, 1.0, 3.0], 'b': [1.0, 1.0, 5.0]}, index=['x', 'y', 'z'])
    before = df.copy()
    np.testing.assert_allclose(normalize(df, 'log2')['a'], np.log2(df['a'] + 1))
    np.testing.assert_allclose(normalize(df, 'log10')['b'], np.log10(df['b'] + 1))
    z = normalize(df, 'z-score')
    assert z['a'].mean() == pytest.approx(0.0, abs=1e-12)
    pd.testing.assert_frame_equal(df, before)


def test_scale_df_keeps_index_and_handles_nan_without_deprecated_api():
    df = pd.DataFrame({'a': [1.0, np.nan, 100.0], 'b': [2.0, 4.0, 8.0]}, index=['x', 'y', 'z'])
    before = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('error')    # no FutureWarning/DeprecationWarning (fillna(method=...))
        res = scale_df(df)
    assert list(res.index) == ['x', 'y', 'z']
    assert res.min().min() == pytest.approx(0.0) and res.max().max() == pytest.approx(1.0)
    pd.testing.assert_frame_equal(df, before)


def test_scale_column_keeps_index_and_copy():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [5, 5, 5]}, index=['x', 'y', 'z'])
    before = df.copy()
    res = scale_column(df, 'a')
    assert res['a'].tolist() == [0.0, 0.5, 1.0]
    assert list(res.index) == ['x', 'y', 'z']
    pd.testing.assert_frame_equal(df, before)


# ------------------------------------------------------------ merge_dataframes
def test_merge_dataframes_on_name_column_or_index_name():
    a = pd.DataFrame({'Name': ['g1', 'g2', 'g3'], 'G': [1, 2, 3]})
    fc = pd.DataFrame({'log2FoldChange': [1.0, 2.0]}, index=pd.Index(['g1', 'g4'], name='Name'))
    res = merge_dataframes([a, fc])
    assert set(res['Name']) == {'g1', 'g2', 'g3', 'g4'}   # outer merge, 'Name' stays a column
    assert res.loc[res['Name'] == 'g1', 'log2FoldChange'].iloc[0] == 1.0
    with pytest.raises(ValueError):
        merge_dataframes([])


def test_scale_df_fills_missing_along_time_not_from_previous_gene():
    x = pd.DataFrame([[10.0, 20.0, 30.0], [40.0, np.nan, 60.0], [70.0, 80.0, 90.0]],
                     index=['geneA', 'geneB', 'geneC'], columns=['t0', 't1', 't2'])
    filled_expected = np.log10(x + 1e-10).ffill(axis=1)
    # geneB t1 takes geneB t0 (40), NOT geneA t1 (20)
    assert filled_expected.loc['geneB', 't1'] == pytest.approx(np.log10(40.0 + 1e-10))
    res = scale_df(x)
    lo, hi = filled_expected.min().to_numpy(), filled_expected.max().to_numpy()
    want = (filled_expected - lo) / (hi - lo)
    pd.testing.assert_frame_equal(res, want)
    assert not res.isna().any().any()
