"""Regression tests for catbridge.foldchange (and the data_io helpers it relies on)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from catbridge import foldchange as fcm
from catbridge.foldchange import (
    compute_fc, df_for_fc, fc_comp, find_noontide, no_repeat_fc,
)


def _row_df(values, cols=None, name='T'):
    cols = cols or [f'c{i}' for i in range(len(values))]
    return pd.DataFrame([values], index=[name], columns=cols)


# ---------------------------------------------------------------- find_noontide
@pytest.mark.parametrize("values,window,offset,expected", [
    # peak in the middle (index 2 of 6)
    ([1, 2, 9, 3, 1, 0], 'after', 1, ['c2', 'c3']),
    ([1, 2, 9, 3, 1, 0], 'after', 2, ['c2', 'c4']),
    ([1, 2, 9, 3, 1, 0], 'before', 1, ['c2', 'c1']),
    ([1, 2, 9, 3, 1, 0], 'before', 2, ['c2', 'c0']),
    # peak in the first column, window='after' works
    ([9, 2, 1, 3, 1, 0], 'after', 1, ['c0', 'c1']),
    ([9, 2, 1, 3, 1, 0], 'after', 2, ['c0', 'c2']),
    # peak in the last column, window='before' works
    ([1, 2, 1, 3, 1, 9], 'before', 1, ['c5', 'c4']),
    ([1, 2, 1, 3, 1, 9], 'before', 2, ['c5', 'c3']),
])
def test_find_noontide_in_range(values, window, offset, expected):
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # no fallback warning expected
        res = find_noontide(_row_df(values), 'T', window=window, offset=offset)
    assert isinstance(res, pd.Index)
    assert list(res) == expected


def test_find_noontide_default_is_after_offset1():
    df = _row_df([1, 5, 2, 1])
    assert list(find_noontide(df, 'T')) == ['c1', 'c2']


def test_find_noontide_peak_last_column_after_falls_back_with_warning():
    df = _row_df([1, 2, 3, 9])
    with pytest.warns(UserWarning, match="no column 1"):
        res = find_noontide(df, 'T', window='after', offset=1)
    # old behaviour used the 2nd highest column (c2/c3 pair unrelated to the peak) -> now c3 vs c2
    assert list(res) == ['c3', 'c2']


def test_find_noontide_peak_first_column_before_falls_back_with_warning():
    df = _row_df([9, 2, 3, 1])
    with pytest.warns(UserWarning):
        res = find_noontide(df, 'T', window='before', offset=1)
    assert list(res) == ['c0', 'c1']


def test_find_noontide_offset2_fallback_other_side_and_both_sides_missing():
    df = _row_df([1, 2, 3, 9])
    with pytest.warns(UserWarning):
        assert list(find_noontide(df, 'T', 'after', 2)) == ['c3', 'c1']
    short = _row_df([1, 9, 1])  # peak in the middle, offset 2 impossible on both sides
    with pytest.raises(ValueError):
        find_noontide(short, 'T', 'after', 2)
    with pytest.raises(ValueError):
        find_noontide(_row_df([5.0]), 'T')  # single column


def test_find_noontide_does_not_use_second_highest_when_peak_is_last():
    # old behaviour: peak last -> picked the second highest column (c0) and its successor (c1)
    df = _row_df([8, 1, 1, 9])
    with pytest.warns(UserWarning):
        res = find_noontide(df, 'T')
    assert res[0] == 'c3'


def test_find_noontide_errors():
    df = _row_df([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="only non-numeric or missing"):
        find_noontide(df, 'T')
    with pytest.raises(KeyError, match="nope"):
        find_noontide(_row_df([1, 2]), 'nope')
    with pytest.raises(ValueError):
        find_noontide(_row_df([1, 2]), 'T', window='sideways')
    with pytest.raises(ValueError):
        find_noontide(_row_df([1, 2]), 'T', offset=0)


def test_find_noontide_ignores_nan_and_does_not_modify_input():
    df = _row_df([1, np.nan, 7, 2])
    before = df.copy()
    assert list(find_noontide(df, 'T')) == ['c2', 'c3']
    pd.testing.assert_frame_equal(df, before)


def test_find_noontide_reproduces_paper_version_when_peak_far_enough():
    # case-study version: [peak, column two positions before the peak]
    df = _row_df([1, 2, 3, 9, 4])
    assert list(find_noontide(df, 'T', window='before', offset=2)) == ['c3', 'c1']


# ---------------------------------------------------------------- no_repeat_fc
def _two_col(pairs):
    return pd.DataFrame(pairs, columns=['a', 'b'], index=[f'g{i}' for i in range(len(pairs))])


def test_no_repeat_fc_basic_value():
    res = no_repeat_fc(_two_col([(8, 2)]), ['a', 'b'])
    assert res.columns.tolist() == ['log2FoldChange']
    assert res['log2FoldChange'].iloc[0] == pytest.approx(np.log2(9 / 3))


def test_no_repeat_fc_zeros_are_finite_and_no_minus_52():
    df = _two_col([(0, 5), (5, 0), (0, 0), (8, 2)])
    res = no_repeat_fc(df, ['a', 'b'])
    v = res['log2FoldChange']
    assert np.isfinite(v).all()
    assert v.iloc[0] == pytest.approx(np.log2(1 / 6))
    assert v.iloc[1] == pytest.approx(np.log2(6 / 1))
    assert v.iloc[2] == 0.0
    assert (v > -50).all()
    assert not np.isinf(v).any()


def test_no_repeat_fc_pseudocount_and_index():
    df = _two_col([(3, 1)])
    res = no_repeat_fc(df, ['a', 'b'], pseudocount=0.5)
    assert res['log2FoldChange'].iloc[0] == pytest.approx(np.log2(3.5 / 1.5))
    assert list(res.index) == ['g0']
    assert res.index.name == 'Name'  # so that merge_dataframes(on='Name') works


def test_no_repeat_fc_nan_stays_nan():
    df = _two_col([(np.nan, 2), (3, np.nan), (3, 1)])
    v = no_repeat_fc(df, ['a', 'b'])['log2FoldChange']
    assert np.isnan(v.iloc[0]) and np.isnan(v.iloc[1])
    assert np.isfinite(v.iloc[2])


def test_no_repeat_fc_validation():
    with pytest.raises(ValueError, match="[Nn]egative"):
        no_repeat_fc(_two_col([(-1, 2)]), ['a', 'b'])
    for pc in (0, -1, np.nan):
        with pytest.raises(ValueError, match="pseudocount"):
            no_repeat_fc(_two_col([(1, 2)]), ['a', 'b'], pseudocount=pc)


def test_no_repeat_fc_does_not_modify_input():
    df = _two_col([(0, 5), (np.nan, 1), (8, 2)])
    df['extra'] = 1.0
    before = df.copy()
    no_repeat_fc(df, ['a', 'b'])
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------- df_for_fc
def _design(groups, reps=3):
    idx, grp = [], []
    for g in groups:
        for r in range(1, reps + 1):
            idx.append(f'{g}_{r}')
            grp.append(g)
    return pd.DataFrame({'group': grp}, index=idx)


def test_df_for_fc_contrast_attrs_and_copies():
    groups = ['day0', 'day7', 'day16', 'day30']
    design = _design(groups)
    metabo_agg = pd.DataFrame([[1, 2, 10, 3]], index=['Cap'], columns=groups)
    rng = np.random.default_rng(1)
    gene = pd.DataFrame(rng.integers(0, 100, (5, len(design))), columns=design.index,
                        index=[f'g{i}' for i in range(5)])
    gene_before, design_before = gene.copy(), design.copy()

    counts, dsub = df_for_fc(metabo_agg, 'Cap', gene, design)
    assert dsub.attrs['contrast'] == ('day16', 'day30')
    assert set(dsub['group']) == {'day16', 'day30'}
    assert list(counts.columns) == list(dsub.index)
    pd.testing.assert_frame_equal(gene, gene_before)
    pd.testing.assert_frame_equal(design, design_before)
    assert 'contrast' not in design.attrs

    counts2, dsub2 = df_for_fc(metabo_agg, 'Cap', gene, design, window='before', offset=2)
    assert dsub2.attrs['contrast'] == ('day16', 'day0')
    dsub2.loc[dsub2.index[0], 'group'] = 'zzz'  # copy: does not leak to the design
    pd.testing.assert_frame_equal(design, design_before)


def test_df_for_fc_samples_follow_data_columns_not_design_row_order():
    groups = ['day0', 'day7', 'day16', 'day30']
    design = _design(groups)
    metabo_agg = pd.DataFrame([[1, 2, 10, 3]], index=['Cap'], columns=groups)
    rng = np.random.default_rng(1)
    gene = pd.DataFrame(rng.integers(0, 100, (5, len(design))), columns=design.index,
                        index=[f'g{i}' for i in range(5)])
    shuffled = design.sort_index(ascending=False)       # design rows in a different order
    counts, dsub = df_for_fc(metabo_agg, 'Cap', gene, shuffled)
    assert list(counts.columns) == [c for c in gene.columns if c in dsub.index]
    assert list(dsub.index) == list(counts.columns)
    assert dsub.attrs['contrast'] == ('day16', 'day30')   # contrast comes from find_noontide


def test_df_for_fc_sample_missing_from_gene_table_raises_keyerror():
    groups = ['a', 'b', 'c']
    design = _design(groups)
    metabo_agg = pd.DataFrame([[1, 9, 2]], index=['T'], columns=groups)
    gene = pd.DataFrame(np.ones((2, len(design))), columns=design.index, index=['g0', 'g1'])
    with pytest.raises(KeyError, match="b_2"):
        df_for_fc(metabo_agg, 'T', gene.drop(columns=['b_2']), design)


def test_resolve_contrast_default_uses_data_column_order_not_metadata_row_order():
    cols = [f'{g}_{r}' for g in ('d9', 'd12', 'd30') for r in (1, 2)]
    meta = pd.DataFrame({'group': [c.rsplit('_', 1)[0] for c in cols]}, index=cols)
    meta_sorted = meta.sort_index()          # alphabetical: d12 first, then d30, d9
    with pytest.warns(UserWarning, match="first two groups"):
        got = fcm._resolve_contrast(None, meta_sorted, sample_order=cols)
    assert got == ('d9', 'd12')              # data column order
    with pytest.warns(UserWarning, match="first two groups"):
        assert fcm._resolve_contrast(None, meta_sorted) == ('d12', 'd30')   # old fallback


# ---------------------------------------------------------------- direction self-check (no pydeseq2 needed)
def _toy_counts():
    rng = np.random.default_rng(0)
    base = rng.uniform(50, 500, 30)
    peak = np.array([base * 4] * 3 + [base] * 3)          # first 3 samples = peak group, 4x
    counts = pd.DataFrame(peak.astype(int), columns=[f'g{i}' for i in range(30)])
    return counts, np.array([True] * 3 + [False] * 3)


def test_direction_agrees_uses_given_size_factors_and_skips_without():
    counts, is_peak = _toy_counts()
    lfc = pd.Series(2.0, index=counts.columns)
    assert fcm._direction_agrees(counts, is_peak, lfc, np.ones(6)) == 1.0
    assert fcm._direction_agrees(counts, is_peak, -lfc, np.ones(6)) == 0.0
    # no size factors -> the check is skipped (None), never a guess from total counts
    assert fcm._direction_agrees(counts, is_peak, lfc, None) is None
    # the size factors matter: dividing the peak group by 16 turns the raw ratio around
    sf = np.array([16.0] * 3 + [1.0] * 3)
    assert fcm._direction_agrees(counts, is_peak, lfc, sf) == 0.0


def test_get_size_factors_from_obs_obsm_and_missing():
    samples = ['s1', 's2', 's3']

    class Dds:
        pass

    d = Dds()
    d.obs = pd.DataFrame({'size_factors': [0.5, 1.0, 2.0]}, index=['s3', 's1', 's2'])
    np.testing.assert_allclose(fcm._get_size_factors(d, samples), [1.0, 2.0, 0.5])   # by label
    d = Dds()
    d.obs = pd.DataFrame(index=samples)                       # no column
    d.obsm = {'size_factors': np.array([1.0, 2.0, 4.0])}      # pydeseq2 0.4.x layout
    np.testing.assert_allclose(fcm._get_size_factors(d, samples), [1.0, 2.0, 4.0])
    d = Dds()
    d.obs = pd.DataFrame(index=samples)
    d.obsm = {}
    assert fcm._get_size_factors(d, samples) is None
    d.obsm = {'size_factors': np.array([1.0, 0.0, 4.0])}      # invalid -> not used
    assert fcm._get_size_factors(d, samples) is None


def test_restore_gene_labels():
    res = pd.Index(['1000', '1001'])
    out = fcm._restore_gene_labels(res, pd.Index([1000, 1001, 1002]))
    assert list(out) == [1000, 1001] and out.dtype == np.int64
    # str labels: unchanged
    assert fcm._restore_gene_labels(res, pd.Index(['1000', '1001'])) is res
    # ambiguous (1 and '1'): not restored, with a warning
    with pytest.warns(UserWarning, match="same string form"):
        out = fcm._restore_gene_labels(pd.Index(['1']), pd.Index([1, '1']))
    assert list(out) == ['1']


# ---------------------------------------------------------------- fc_comp (needs a working pydeseq2)
def _make_counts(pk, ot, seed=0, n_genes=150, n_up=20, reps=3):
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 100.0, n_genes) + 20
    fold = np.ones(n_genes)
    fold[:n_up] = 8.0                      # first genes: clearly higher in the peak group
    mu = np.hstack([base[:, None] * fold[:, None] * np.ones((1, reps)),
                    base[:, None] * np.ones((1, reps))])
    counts = rng.negative_binomial(10, 10 / (10 + mu))
    cols = [f'{pk}_{i}' for i in range(1, reps + 1)] + [f'{ot}_{i}' for i in range(1, reps + 1)]
    df = pd.DataFrame(counts, index=[f'g{i}' for i in range(n_genes)], columns=cols)
    df.index.name = 'Name'
    meta = pd.DataFrame({'group': [pk] * reps + [ot] * reps}, index=cols)
    return df, meta, n_up


@pytest.fixture(scope='module')
def deseq_ok():
    """Skip DESeq2 tests where pydeseq2 itself is broken (e.g. pydeseq2 0.5.x + pandas 3)."""
    pytest.importorskip('pydeseq2')
    from pydeseq2.dds import DeseqDataSet
    counts, meta, _ = _make_counts('a1', 'b1', n_genes=30)
    m = pd.DataFrame({'condition': meta['group'].to_numpy()}, index=meta.index)
    try:
        import inspect
        kw = dict(counts=counts.T, metadata=m)
        if 'design' in inspect.signature(DeseqDataSet.__init__).parameters:
            kw['design'] = '~condition'
        else:
            kw['design_factors'] = 'condition'
        DeseqDataSet(**kw).deseq2()
    except Exception as e:  # third-party incompatibility, not our bug
        pytest.skip(f"pydeseq2 does not run in this environment: {type(e).__name__}: {e}")


@pytest.mark.parametrize("pk,ot", [('day30', 'day50'),    # alphabetical == time order
                                   ('day9', 'day12')])    # alphabetical != time order
@pytest.mark.parametrize("shrink", [False, True])
def test_fc_comp_direction_is_peak_over_other(deseq_ok, pk, ot, shrink):
    counts, meta, n_up = _make_counts(pk, ot)
    meta_before = meta.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        warnings.simplefilter('ignore', FutureWarning)
        res = fc_comp(counts, meta, contrast=(pk, ot), shrink=shrink)
    assert {'log2FoldChange', 'pvalue', 'padj'} <= set(res.columns)
    assert res.index.name == 'Name'
    up = res.loc[[f'g{i}' for i in range(n_up)], 'log2FoldChange']
    assert (up > 0).all()
    assert up.median() > 1.5
    pd.testing.assert_frame_equal(meta, meta_before)   # metadata untouched (no 'condition' column)
    assert 'condition' not in meta.columns


def test_fc_comp_sign_independent_of_group_naming(deseq_ok):
    out = {}
    for pk, ot in [('day30', 'day50'), ('day9', 'day12')]:
        counts, meta, n_up = _make_counts(pk, ot, seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[(pk, ot)] = fc_comp(counts, meta, contrast=(pk, ot), shrink=False)['log2FoldChange']
    a, b = out[('day30', 'day50')], out[('day9', 'day12')]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-3, atol=1e-4)


def test_fc_comp_swapped_contrast_flips_sign(deseq_ok):
    counts, meta, n_up = _make_counts('day9', 'day12', seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fwd = fc_comp(counts, meta, contrast=('day9', 'day12'), shrink=False)['log2FoldChange']
        rev = fc_comp(counts, meta, contrast=('day12', 'day9'), shrink=False)['log2FoldChange']
    np.testing.assert_allclose(fwd.to_numpy(), -rev.to_numpy(), rtol=1e-3, atol=1e-4)


def test_fc_comp_contrast_from_attrs_and_default_warning(deseq_ok):
    counts, meta, n_up = _make_counts('day9', 'day12', seed=7)
    # file order is the "wrong" way round on purpose; attrs say (day12, day9) -> log2(day12/day9)
    meta2 = meta.copy()
    meta2.attrs['contrast'] = ('day12', 'day9')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fc_comp(counts, meta2, shrink=False)
    assert (res.loc[[f'g{i}' for i in range(n_up)], 'log2FoldChange'] < 0).all()
    # no contrast anywhere -> first two groups in order of appearance, with a warning
    with pytest.warns(UserWarning, match="first two groups"):
        res2 = fc_comp(counts, meta, shrink=False)
    assert (res2.loc[[f'g{i}' for i in range(n_up)], 'log2FoldChange'] > 0).all()


def test_fc_comp_shrinkage_shrinks_toward_zero(deseq_ok):
    counts, meta, n_up = _make_counts('day9', 'day12', seed=11, n_genes=200)
    # add low-count noisy genes
    rng = np.random.default_rng(0)
    low = pd.DataFrame(rng.poisson(1.5, (40, counts.shape[1])), columns=counts.columns,
                       index=[f'low{i}' for i in range(40)])
    low.index.name = 'Name'
    counts = pd.concat([counts, low])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = fc_comp(counts, meta, contrast=('day9', 'day12'), shrink=False)
        shr = fc_comp(counts, meta, contrast=('day9', 'day12'), shrink=True)
    common = raw.index.intersection(shr.index)
    lowg = [g for g in common if g.startswith('low')]
    assert len(lowg) > 5
    assert shr.loc[lowg, 'log2FoldChange'].abs().mean() < raw.loc[lowg, 'log2FoldChange'].abs().mean()
    # p-values are not changed by shrinkage
    pd.testing.assert_series_equal(raw.loc[common, 'pvalue'], shr.loc[common, 'pvalue'])


def test_fc_comp_warns_on_non_count_input_and_validates(deseq_ok):
    counts, meta, _ = _make_counts('day9', 'day12', seed=2, n_genes=60)
    tpm = counts + 0.37
    with pytest.warns(UserWarning, match="raw counts"):
        fc_comp(tpm, meta, contrast=('day9', 'day12'), shrink=False)
    with pytest.raises(ValueError, match="not found in metadata"):
        fc_comp(counts, meta, contrast=('day9', 'nope'))
    with pytest.raises(ValueError):
        fc_comp(counts, meta, contrast=('day9', 'day9'))
    with pytest.raises(ValueError, match="negative"):
        fc_comp(counts - 1000, meta, contrast=('day9', 'day12'))


# ---------------------------------------------------------------- compute_fc (no-replicate path, files)
def test_compute_fc_without_design_passes_window_and_offset(tmp_path):
    cols = ['s1', 's2', 's3', 's4']
    gene = pd.DataFrame([[1, 10, 3, 4], [0, 0, 0, 0]], index=['g1', 'g2'], columns=cols)
    gene.index.name = 'Name'
    metabo = pd.DataFrame([[1, 9, 2, 1]], index=['T'], columns=cols)
    metabo.index.name = 'Name'
    gp, mp = tmp_path / 'gene.csv', tmp_path / 'metabo.csv'
    gene.to_csv(gp)
    metabo.to_csv(mp)
    fc = compute_fc(str(gp), str(mp), target='T')
    assert fc.loc['g1', 'log2FoldChange'] == pytest.approx(np.log2(11 / 4))
    assert fc.loc['g2', 'log2FoldChange'] == 0.0
    fc_b = compute_fc(str(gp), str(mp), target='T', window='before', offset=1)
    assert fc_b.loc['g1', 'log2FoldChange'] == pytest.approx(np.log2(11 / 2))


def test_compute_fc_requires_target(tmp_path):
    p = tmp_path / 'x.csv'
    pd.DataFrame({'a': [1]}, index=pd.Index(['g'], name='Name')).to_csv(p)
    with pytest.raises(ValueError, match="target"):
        compute_fc(str(p), str(p))


# ---------------------------------------------------------------- data_io
def test_get_target_missing_raises_keyerror_with_name():
    from catbridge.data_io import get_target
    df = pd.DataFrame([[1, 2]], index=['Cap'], columns=['a', 'b'])
    assert get_target('Cap', df) == [1, 2]
    with pytest.raises(KeyError, match="Nope"):
        get_target('Nope', df)


def test_read_upload_unsupported_format_raises_valueerror(tmp_path):
    from catbridge.data_io import read_upload
    p = tmp_path / 'table.json'
    p.write_text('{}')
    with pytest.raises(ValueError, match="Unsupported"):
        read_upload(str(p))


def test_read_upload_csv_and_tsv(tmp_path):
    from catbridge.data_io import read_upload
    df = pd.DataFrame({'a': [1, 2]}, index=pd.Index(['x', 'y'], name='Name'))
    df.to_csv(tmp_path / 'a.csv')
    df.to_csv(tmp_path / 'a.tsv', sep='\t')
    pd.testing.assert_frame_equal(read_upload(str(tmp_path / 'a.csv')), df)
    pd.testing.assert_frame_equal(read_upload(str(tmp_path / 'a.tsv')), df)


# ---------------------------------------------------------------- fc_comp: direction self-check never raises
def _composition_bias_counts(seed, n=400):
    """3 giant genes take ~60% of the library in the peak group only; all others are unchanged."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(80, 800, n)
    samples = [f'p_{i}' for i in range(3)] + [f'q_{i}' for i in range(3)]
    cnt = []
    for s_ in samples:
        m = mu.copy()
        m[:3] = (mu.sum() * (1.5 if s_.startswith('p') else 0.05)) / 3
        cnt.append(rng.poisson(m * rng.lognormal(0, .05, n)))
    counts = pd.DataFrame(np.array(cnt).T, index=[f'g{i}' for i in range(n)], columns=samples)
    meta = pd.DataFrame({'group': ['pk'] * 3 + ['ot'] * 3}, index=samples)
    return counts, meta


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_fc_comp_composition_bias_does_not_raise_and_unchanged_genes_near_zero(deseq_ok, seed):
    counts, meta = _composition_bias_counts(seed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = fc_comp(counts, meta, contrast=('pk', 'ot'), shrink=False)
    assert not [x for x in w if 'direction check' in str(x.message)]
    unchanged = res.loc[[f'g{i}' for i in range(3, 400)], 'log2FoldChange']
    assert len(unchanged) == 397
    assert abs(unchanged.median()) < 0.15


def test_fc_comp_composition_bias_with_shrinkage(deseq_ok):
    counts, meta = _composition_bias_counts(4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fc_comp(counts, meta, contrast=('pk', 'ot'), shrink=True)
    assert abs(res.loc[[f'g{i}' for i in range(3, 400)], 'log2FoldChange'].median()) < 0.15


def test_fc_comp_direction_mismatch_only_warns(deseq_ok, monkeypatch):
    counts, meta, _ = _make_counts('day9', 'day12', seed=3, n_genes=60)
    monkeypatch.setattr(fcm, '_direction_agrees', lambda *a, **k: 0.0)
    with pytest.warns(UserWarning, match="direction check"):
        res = fc_comp(counts, meta, contrast=('day9', 'day12'), shrink=False)
    assert len(res) > 0


def test_fc_comp_without_size_factors_skips_check(deseq_ok, monkeypatch):
    counts, meta, _ = _make_counts('day9', 'day12', seed=3, n_genes=60)
    monkeypatch.setattr(fcm, '_get_size_factors', lambda dds, samples: None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = fc_comp(counts, meta, contrast=('day9', 'day12'), shrink=True)
    assert len(res) > 0
    assert not [x for x in w if 'direction check' in str(x.message)
                or 'shrinkage skipped' in str(x.message)]


def test_fc_comp_check_receives_deseq2_size_factors(deseq_ok, monkeypatch):
    counts, meta, _ = _make_counts('day9', 'day12', seed=3, n_genes=60)
    seen = []
    real = fcm._direction_agrees

    def spy(c, is_peak, lfc, size_factors, **kw):
        seen.append(size_factors)
        return real(c, is_peak, lfc, size_factors, **kw)

    monkeypatch.setattr(fcm, '_direction_agrees', spy)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fc_comp(counts, meta, contrast=('day9', 'day12'), shrink=True)
    assert len(seen) == 2                     # once unshrunk, once after shrinkage
    # independent median-of-ratios size factors (genes with a positive count in every sample)
    c = counts.to_numpy(dtype=float).T
    c = c[:, c.sum(axis=0) >= 10]
    c = c[:, (c > 0).all(axis=0)]
    logc = np.log(c)
    ref = np.exp(np.median(logc - logc.mean(axis=0), axis=1))
    for sf in seen:
        assert sf is not None and len(sf) == 6 and np.all(sf > 0)
        np.testing.assert_allclose(sf, ref, rtol=1e-6)


# ---------------------------------------------------------------- fc_comp: gene labels
def test_fc_comp_restores_integer_gene_labels(deseq_ok):
    counts, meta, n_up = _make_counts('day9', 'day12', seed=3, n_genes=60)
    counts.index = np.arange(1000, 1060)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fc_comp(counts, meta, contrast=('day9', 'day12'), shrink=False)
    assert res.index.dtype == np.int64 and res.index.name == 'Name'
    assert set(res.index) <= set(counts.index)
    assert (res.loc[list(range(1000, 1000 + n_up)), 'log2FoldChange'] > 0).all()


def test_fc_comp_string_labels_unchanged(deseq_ok):
    counts, meta, _ = _make_counts('day9', 'day12', seed=3, n_genes=60)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fc_comp(counts, meta, contrast=('day9', 'day12'), shrink=False)
    assert all(isinstance(i, str) for i in res.index)
