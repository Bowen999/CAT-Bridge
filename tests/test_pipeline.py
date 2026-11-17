"""Tests for catbridge.pipeline (compute_corr / pipeline) on a small synthetic data set.

Tests that need a working tslearn (clustering) or pydeseq2 (DESeq2 fold change) skip
themselves when the third-party package is missing or does not run in this environment.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from catbridge import pipeline as pl
from catbridge.pipeline import compute_corr, pipeline
from catbridge.scoring import compute_score

# 6 time points with the default E=3 are too short for CCM (all NaN + a UserWarning)
pytestmark = pytest.mark.filterwarnings('ignore:CCM skipped')

GROUPS = ['day0', 'day7', 'day16', 'day30', 'day50', 'day60']   # NOT zero padded on purpose
ALPHA_ORDER = sorted(GROUPS)                                    # day0, day16, day30, day50, day60, day7
assert ALPHA_ORDER != GROUPS
REPS = 3
N_GENES = 40
TRUE_GENE = 'G_true'
TARGET = 'Target'
# mean profiles (arbitrary units), time order day0..day60
TRUE_PROFILE = np.array([30, 80, 300, 700, 150, 50], dtype=float)
TARGET_PROFILE = np.array([1, 2, 4, 10, 5, 2], dtype=float) * 1e5
PEAK = int(np.argmax(TARGET_PROFILE))                           # day30
assert GROUPS[PEAK] == 'day30'


# ------------------------------------------------------------------ data
def _make_dataset(tmp_path, seed=3):
    """Write gene / metabolite / design / annotation tables; return their paths.

    gene.tsv, metabolite.tsv: 40 genes (resp. 3 metabolites) x 6 groups x 3 replicates.
    gene_mean.tsv, metabolite_mean.tsv: the same without replicates (6 columns = group names).
    """
    rng = np.random.default_rng(seed)
    cols = [f"{g}_{r}" for g in GROUPS for r in range(1, REPS + 1)]
    design = pd.DataFrame({'group': [c.rsplit('_', 1)[0] for c in cols]},
                          index=pd.Index(cols, name='sample'))

    profiles = {TRUE_GENE: TRUE_PROFILE}
    for i in range(N_GENES - 1):
        base = rng.uniform(100, 600)
        profiles[f'G{i:02d}'] = base * rng.lognormal(0.0, 0.3, size=len(GROUPS))
    # a few decoys: strongly anti-correlated / late-rising genes
    for i in range(3):
        profiles[f'G{i:02d}'] = TRUE_PROFILE[::-1] * rng.uniform(0.8, 1.2)
    for i in range(3, 6):
        profiles[f'G{i:02d}'] = np.linspace(50, 800, len(GROUPS)) * rng.uniform(0.8, 1.2)
    genes = pd.DataFrame(
        {c: [rng.poisson(profiles[g][GROUPS.index(design.loc[c, 'group'])]
                         * rng.lognormal(0, 0.15)) for g in profiles] for c in cols},
        index=pd.Index(list(profiles), name='Name'),
    ).astype(float)

    met_profiles = {TARGET: TARGET_PROFILE,
                    'MetA': np.array([5, 4, 3, 2, 3, 6], dtype=float) * 1e5,
                    'MetB': rng.uniform(1, 5, size=len(GROUPS)) * 1e5}
    metabo = pd.DataFrame(
        {c: [met_profiles[m][GROUPS.index(design.loc[c, 'group'])] * rng.lognormal(0, 0.05)
             for m in met_profiles] for c in cols},
        index=pd.Index(list(met_profiles), name='Name'),
    )
    annotation = pd.DataFrame(
        {'Description': [('capsaicin synthase' if g == TRUE_GENE else
                          'hypothetical protein' if i % 2 else 'putative kinase')
                         for i, g in enumerate(genes.index)]},
        index=genes.index,
    )

    # replicate-free versions: group means, columns named by group
    gene_mean = genes.T.groupby(design['group'].to_numpy(), sort=False).mean().T.round()
    metabo_mean = metabo.T.groupby(design['group'].to_numpy(), sort=False).mean().T

    paths = {}
    for name, df in [('gene', genes), ('metabolite', metabo), ('design', design),
                     ('annotation', annotation), ('gene_mean', gene_mean),
                     ('metabolite_mean', metabo_mean)]:
        p = tmp_path / f"{name}.tsv"
        df.to_csv(p, sep='\t')
        paths[name] = str(p)
    return paths


@pytest.fixture
def data(tmp_path):
    return _make_dataset(tmp_path)


def _fake_fc_comp(counts_df, metadata, contrast=None, shrink=True, n_cpus=None):
    """Stand-in for fc_comp (no pydeseq2): log2 of the group-mean ratio peak / comparison."""
    peak, other = contrast if contrast is not None else metadata.attrs['contrast']
    g = metadata.loc[list(counts_df.columns), 'group']
    m1 = counts_df.loc[:, (g == peak).to_numpy()].mean(axis=1)
    m0 = counts_df.loc[:, (g == other).to_numpy()].mean(axis=1)
    out = pd.DataFrame({'log2FoldChange': np.log2((m1 + 1) / (m0 + 1))})
    out.index.name = 'Name'
    return out


@pytest.fixture
def fake_fc(monkeypatch):
    calls = []

    def recorder(counts_df, metadata, **kw):
        calls.append((counts_df, metadata, kw))
        return _fake_fc_comp(counts_df, metadata, **kw)

    monkeypatch.setattr(pl, 'fc_comp', recorder)
    return calls


@pytest.fixture(scope='module')
def deseq_ok():
    """Skip where pydeseq2 is missing or broken (e.g. pydeseq2 0.5.x + pandas 3)."""
    pytest.importorskip('pydeseq2')
    from pydeseq2.dds import DeseqDataSet
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.poisson(100, size=(8, 12)),
                          index=[f's{i}' for i in range(8)],
                          columns=[f'g{i}' for i in range(12)])
    meta = pd.DataFrame({'condition': ['a'] * 4 + ['b'] * 4}, index=counts.index)
    try:
        kw = dict(counts=counts, metadata=meta)
        if 'design' in inspect.signature(DeseqDataSet.__init__).parameters:
            kw['design'] = '~condition'
        else:
            kw['design_factors'] = 'condition'
        DeseqDataSet(**kw).deseq2()
    except Exception as e:  # third-party incompatibility, not our bug
        pytest.skip(f"pydeseq2 does not run in this environment: {type(e).__name__}: {e}")


@pytest.fixture(scope='module')
def tslearn_ok():
    try:
        pytest.importorskip('tslearn')
        from catbridge.metrics import ts_clustering
        rng = np.random.default_rng(0)
        ts_clustering(pd.DataFrame(rng.normal(size=(6, 8))), 2)
    except pytest.skip.Exception:
        raise
    except Exception as e:
        pytest.skip(f"tslearn not usable in this environment: {type(e).__name__}: {e}")


def _args(d, design=True, annotation=True):
    """Positional arguments of the first four parameters."""
    if design:
        return (d['gene'], d['metabolite'], d['design'], d['annotation'] if annotation else None)
    return (d['gene_mean'], d['metabolite_mean'], None, d['annotation'] if annotation else None)


EXPECTED_METRIC_COLS = ['Granger', 'CCM', 'CCA', 'DTW', 'CCF', 'Spearman', 'Pearson']


# ------------------------------------------------------------------ signatures
def test_compute_corr_signature():
    sig = inspect.signature(compute_corr)
    params = list(sig.parameters.values())
    old = ['gene_file', 'metabo_file', 'design_file', 'annotation_file', 'target',
           'cluster_count', 'aggregation_func', 'lag', 'E', 'tau', 'n_components']
    assert [p.name for p in params[:11]] == old
    assert [p.default for p in params[6:11]] == [None, 1, 3, 1, 1]
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params[:11])
    kwonly = {p.name: p.default for p in params[11:]}
    assert all(p.kind is p.KEYWORD_ONLY for p in params[11:])
    assert kwonly == dict(metrics=None, fc_window='after', fc_offset=1, pseudocount=1.0,
                          shrink=True, n_cpus=None, random_state=0)


def test_pipeline_signature():
    params = list(inspect.signature(pipeline).parameters.values())
    assert [p.name for p in params[:8]] == [
        'gene_file', 'metabo_file', 'design_file', 'annotation_file', 'target',
        'cluster_count', 'max_lag', 'aggregation_func']
    assert params[6].default == 1 and params[7].default is None
    assert params[8].name == 'score_col' and params[8].kind is params[8].KEYWORD_ONLY
    assert params[8].default == 'Granger'


# ------------------------------------------------------------------ compute_corr, no design
def test_compute_corr_no_design(data):
    out = compute_corr(*_args(data, design=False), TARGET, None)
    assert len(out) == N_GENES and out['Name'].is_unique
    assert set(out['Name']) == set(pd.read_csv(data['gene_mean'], sep='\t', index_col=0).index)
    for c in EXPECTED_METRIC_COLS + ['log2FoldChange', 'Description', 'Description_Score']:
        assert c in out.columns, c
    assert 'Cluster' not in out.columns
    # 'log2FoldChange' of the planted gene: peak day30 (700) vs day50 (150) -> clearly positive
    lfc = out.set_index('Name')['log2FoldChange']
    assert lfc[TRUE_GENE] > 1.5
    assert out.set_index('Name').loc[TRUE_GENE, 'Description_Score'] == pytest.approx(0.2)


def test_compute_corr_positional_old_style_call(data):
    """Old callers passed target, cluster_count, aggregation_func, lag, E, tau, n_components positionally."""
    out = compute_corr(*_args(data, design=False), TARGET, 0, None, 1, 3, 1, 1)
    assert len(out) == N_GENES and 'Cluster' not in out.columns


def test_metrics_subset(data):
    out = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                       metrics=['Pearson'])
    assert 'Pearson' in out.columns
    for c in ['Granger', 'CCM', 'CCA', 'DTW', 'CCF', 'Spearman', 'CCM_p', 'Granger_padj']:
        assert c not in out.columns
    assert 'log2FoldChange' in out.columns
    assert 'Description' not in out.columns


def test_metrics_none_means_all(data):
    out = compute_corr(*_args(data, design=False, annotation=False), TARGET, None, metrics=None)
    for c in EXPECTED_METRIC_COLS:
        assert c in out.columns


@pytest.mark.parametrize("bad", [['Pearson', 'Foo'], ['pearson'], ['Granger_padj']])
def test_unknown_metric_raises(data, bad):
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_corr(*_args(data, design=False), TARGET, None, metrics=bad)


def test_lag_reaches_ccf_and_granger(data, monkeypatch):
    # 6 time points: Granger can only run at lag 1 (statsmodels needs nobs > 3*lag+1), so check
    # the CCF values and that Granger receives exactly the requested maxlag.
    seen = []
    real = pl.compute_granger

    def spy(df, t, maxlag=1):
        seen.append(maxlag)
        return real(df, t, maxlag=maxlag)

    monkeypatch.setattr(pl, 'compute_granger', spy)
    a = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                     lag=1, metrics=['CCF', 'Granger']).set_index('Name')
    with pytest.warns(UserWarning, match="largest usable lag is 1"):
        b = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                         lag=2, metrics=['CCF', 'Granger']).set_index('Name')
    assert seen == [1, 2]
    assert not np.allclose(a['CCF'], b['CCF'], equal_nan=True)
    assert a['Granger'].notna().any() and b['Granger'].isna().all()


def test_lag_changes_granger_values_with_enough_time_points(tmp_path):
    paths = _long_dataset(tmp_path)
    kw = dict(metrics=['Granger'])
    a = compute_corr(paths['gene'], paths['metabolite'], None, None, TARGET, None, lag=1, **kw)
    b = compute_corr(paths['gene'], paths['metabolite'], None, None, TARGET, None, lag=2, **kw)
    assert a['Granger'].notna().all() and b['Granger'].notna().all()
    assert not np.allclose(a['Granger'], b['Granger'])


def test_ccf_lag_value_is_the_requested_lag(data):
    from catbridge.metrics import compute_ccf
    gene = pd.read_csv(data['gene_mean'], sep='\t', index_col=0)
    met = pd.read_csv(data['metabolite_mean'], sep='\t', index_col=0)
    t = met.loc[TARGET].tolist()
    for lag in (1, 2):
        got = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                           lag=lag, metrics=['CCF']).set_index('Name')['CCF']
        want = compute_ccf(gene, t, lag=lag).set_index('Name')['CCF']
        pd.testing.assert_series_equal(got.reindex(want.index), want, check_names=False)


def test_n_components_reaches_cca(data, monkeypatch):
    """The value passed to compute_corr must arrive in compute_cca unchanged (a stand-in that
    always passes the default would not be caught by a test that only uses the default)."""
    seen = []
    real = pl.compute_cca

    def spy(df, t, n_components=1):
        seen.append(n_components)
        return real(df, t, n_components=n_components)

    monkeypatch.setattr(pl, 'compute_cca', spy)
    args = _args(data, design=False, annotation=False)
    out = compute_corr(*args, TARGET, None, n_components=1, metrics=['CCA'])
    assert seen == [1] and out['CCA'].notna().any()
    # n_components=2 is not possible (one gene vs one metabolite): it must raise, not give NaN
    with pytest.raises(ValueError, match="n_components"):
        compute_corr(*args, TARGET, None, n_components=2, metrics=['CCA'])
    assert seen == [1, 2]
    with pytest.raises(ValueError, match="n_components"):
        pipeline(*args, TARGET, None, score_col='CCA', n_components=3)
    assert seen == [1, 2, 3]


def test_n_components_not_checked_when_cca_not_requested(data):
    out = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                       n_components=2, metrics=['Pearson'])
    assert len(out) == N_GENES


def test_E_tau_reach_ccm(data):
    a = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                     E=2, tau=1, metrics=['CCM']).set_index('Name')['CCM']
    b = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                     E=3, tau=1, metrics=['CCM']).set_index('Name')['CCM']
    assert not np.allclose(a, b, equal_nan=True)


def _long_dataset(tmp_path, n_time=12, n_genes=15, seed=5):
    """Replicate-free tables with enough time points for CCM (E=3) and Granger lag 2-3."""
    rng = np.random.default_rng(seed)
    cols = [f't{i}' for i in range(n_time)]
    gene = pd.DataFrame(rng.normal(size=(n_genes, n_time)).cumsum(axis=1) + 50,
                        index=pd.Index([f'g{i}' for i in range(n_genes)], name='Name'), columns=cols)
    met = pd.DataFrame([rng.normal(size=n_time).cumsum() + 50], index=pd.Index([TARGET], name='Name'),
                       columns=cols)
    paths = {}
    for name, df in (('gene', gene), ('metabolite', met)):
        paths[name] = str(tmp_path / f'long_{name}.tsv')
        df.to_csv(paths[name], sep='\t')
    return paths


def test_default_compute_corr_ccm_column_has_values_matching_compute_ccm(tmp_path):
    """With 12 time points (E=3) CCM is defined: the column must hold numbers equal to a direct
    compute_ccm call (the 6-point fixture gives all NaN, so it never checked a value)."""
    import warnings
    from catbridge.metrics import compute_ccm
    paths = _long_dataset(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        warnings.filterwarnings('error', message='CCM skipped')
        out = compute_corr(paths['gene'], paths['metabolite'], None, None, TARGET, None)
    gene = pd.read_csv(paths['gene'], sep='\t', index_col=0)
    t = pd.read_csv(paths['metabolite'], sep='\t', index_col=0).loc[TARGET].to_numpy()
    want = compute_ccm(gene, t, E=3, tau=1).set_index('Name')
    got = out.set_index('Name').reindex(want.index)
    assert got['CCM'].notna().all()
    assert got['CCM'].between(-1, 1).all()
    np.testing.assert_allclose(got['CCM'].to_numpy(), want['CCM'].to_numpy())
    np.testing.assert_allclose(got['CCM_p'].to_numpy(), want['CCM_p'].to_numpy())


# ------------------------------------------------------------------ fc_offset
def test_fc_offset_reaches_find_noontide_no_design(data, monkeypatch):
    seen = []
    real = pl.find_noontide

    def spy(df, row_name, window='after', offset=1):
        seen.append((window, offset))
        return real(df, row_name, window=window, offset=offset)

    monkeypatch.setattr(pl, 'find_noontide', spy)
    kw = dict(metrics=['Pearson'])
    a = compute_corr(*_args(data, design=False, annotation=False), TARGET, None, **kw)
    b = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                     fc_offset=2, **kw)
    assert seen == [('after', 1), ('after', 2)]
    la, lb = a.set_index('Name')['log2FoldChange'], b.set_index('Name')['log2FoldChange']
    assert not np.allclose(la, lb.reindex(la.index))
    # planted gene: peak day30 (700) vs day50 (150) with offset 1, vs day60 (50) with offset 2
    assert la[TRUE_GENE] == pytest.approx(np.log2(701 / 151), abs=0.4)
    assert lb[TRUE_GENE] == pytest.approx(np.log2(701 / 51), abs=0.5)
    assert lb[TRUE_GENE] > la[TRUE_GENE] + 0.8


def test_fc_offset_reaches_df_for_fc_and_changes_contrast_with_design(data, fake_fc, monkeypatch):
    seen = []
    real = pl.df_for_fc

    def spy(df1, target, df2, design, window='after', offset=1):
        seen.append((window, offset))
        return real(df1, target, df2, design, window=window, offset=offset)

    monkeypatch.setattr(pl, 'df_for_fc', spy)
    compute_corr(*_args(data), TARGET, None, metrics=['Pearson'])
    compute_corr(*_args(data), TARGET, None, metrics=['Pearson'], fc_offset=2)
    compute_corr(*_args(data), TARGET, None, metrics=['Pearson'], fc_offset=2, fc_window='before')
    assert seen == [('after', 1), ('after', 2), ('before', 2)]
    contrasts = [kw['contrast'] for _, _, kw in fake_fc]
    assert contrasts == [('day30', 'day50'), ('day30', 'day60'), ('day30', 'day7')]
    assert set(fake_fc[1][0].columns) == {f'{g}_{r}' for g in ('day30', 'day60') for r in (1, 2, 3)}


def test_fc_offset_through_pipeline_kwargs(data, monkeypatch):
    seen = []
    real = pl.find_noontide
    monkeypatch.setattr(pl, 'find_noontide', lambda df, r, window='after', offset=1: (
        seen.append(offset), real(df, r, window=window, offset=offset))[1])
    pipeline(data['gene_mean'], data['metabolite_mean'], None, None, TARGET, None, fc_offset=2)
    assert seen == [2]


# ------------------------------------------------------------------ column alignment
def test_mismatched_column_names_raise(data, tmp_path):
    met = pd.read_csv(data['metabolite_mean'], sep='\t', index_col=0)
    met = met.rename(columns={'day16': 'day16x'})
    p = tmp_path / 'met_bad.tsv'
    met.to_csv(p, sep='\t')
    with pytest.raises(ValueError, match="day16x"):
        compute_corr(data['gene_mean'], str(p), None, None, TARGET, None)


def test_same_names_different_order_raise(data, tmp_path):
    met = pd.read_csv(data['metabolite_mean'], sep='\t', index_col=0)
    met = met[sorted(met.columns)]          # alphabetical order -> day16 before day7
    p = tmp_path / 'met_bad_order.tsv'
    met.to_csv(p, sep='\t')
    with pytest.raises(ValueError, match="order"):
        compute_corr(data['gene_mean'], str(p), None, None, TARGET, None)


def test_mismatched_groups_with_design_raise(data, tmp_path):
    """A whole group missing from the metabolite table -> aggregated columns differ."""
    met = pd.read_csv(data['metabolite'], sep='\t', index_col=0)
    met = met[[c for c in met.columns if not c.startswith('day60_')]]
    p = tmp_path / 'met_missing_group.tsv'
    met.to_csv(p, sep='\t')
    with pytest.raises(ValueError, match="day60"):
        compute_corr(data['gene'], str(p), data['design'], None, TARGET, None)


def test_missing_target_raises_keyerror(data):
    with pytest.raises(KeyError, match="Nope"):
        compute_corr(*_args(data, design=False), 'Nope', None)


# ------------------------------------------------------------------ fold change window
def test_fc_window_before_differs_from_after_no_design(data):
    kw = dict(metrics=['Pearson'])
    after = compute_corr(*_args(data, design=False, annotation=False), TARGET, None, **kw)
    before = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                          fc_window='before', **kw)
    a = after.set_index('Name')['log2FoldChange']
    b = before.set_index('Name')['log2FoldChange'].reindex(a.index)
    assert not np.allclose(a, b)
    # the planted gene: day30 vs day50 (after) is a big drop, day30 vs day16 (before) a smaller one
    assert a[TRUE_GENE] == pytest.approx(np.log2((700 + 1) / (150 + 1)), abs=0.4)
    assert b[TRUE_GENE] == pytest.approx(np.log2((700 + 1) / (300 + 1)), abs=0.4)


def test_pseudocount_reaches_no_repeat_fc(data):
    kw = dict(metrics=['Pearson'])
    a = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                     pseudocount=1.0, **kw).set_index('Name')['log2FoldChange']
    b = compute_corr(*_args(data, design=False, annotation=False), TARGET, None,
                     pseudocount=1000.0, **kw).set_index('Name')['log2FoldChange']
    assert not np.allclose(a, b.reindex(a.index))


# ------------------------------------------------------------------ design path (wiring, fake DESeq2)
def test_design_path_wiring_with_fake_fc(data, fake_fc):
    out = compute_corr(*_args(data), TARGET, None, shrink=False, n_cpus=2)
    assert len(out) == N_GENES and out['Name'].is_unique
    for c in EXPECTED_METRIC_COLS + ['log2FoldChange', 'Description_Score']:
        assert c in out.columns
    assert len(fake_fc) == 1
    counts, meta, kw = fake_fc[0]
    # contrast = (peak group of the target, the group after it); passed on explicitly and via attrs
    assert kw['contrast'] == ('day30', 'day50')
    assert meta.attrs['contrast'] == ('day30', 'day50')
    assert kw['shrink'] is False and kw['n_cpus'] == 2
    # raw (unaggregated) replicate columns of the two groups
    assert set(counts.columns) == {f'{g}_{r}' for g in ('day30', 'day50') for r in (1, 2, 3)}
    # direction: the planted gene is up at the peak group
    assert out.set_index('Name').loc[TRUE_GENE, 'log2FoldChange'] > 1.5


def test_design_path_fc_window_before_with_fake_fc(data, fake_fc):
    a = compute_corr(*_args(data), TARGET, None, metrics=['Pearson']).set_index('Name')
    b = compute_corr(*_args(data), TARGET, None, metrics=['Pearson'],
                     fc_window='before').set_index('Name')
    assert fake_fc[0][2]['contrast'] == ('day30', 'day50')
    assert fake_fc[1][2]['contrast'] == ('day30', 'day16')
    assert not np.allclose(a['log2FoldChange'], b['log2FoldChange'].reindex(a.index))


def test_aggregation_default_is_mean_and_time_order_kept(data, fake_fc, monkeypatch):
    """Replicates are averaged (default aggregation) and the time order is NOT alphabetical."""
    seen = {}
    real = pl.compute_pearson

    def spy(df, t):
        seen['cols'] = list(df.columns)
        seen['df'] = df.copy()
        seen['t'] = list(t)
        return real(df, t)

    monkeypatch.setattr(pl, 'compute_pearson', spy)
    compute_corr(*_args(data), TARGET, None, metrics=['Pearson'])
    assert seen['cols'] == GROUPS
    assert seen['cols'].index('day7') < seen['cols'].index('day16')
    # target series is in time order too (peak at day30 = position 3)
    assert int(np.argmax(seen['t'])) == PEAK
    # mean of the replicates
    raw = pd.read_csv(data['gene'], sep='\t', index_col=0)
    expected = raw[[f'day7_{r}' for r in (1, 2, 3)]].mean(axis=1)
    np.testing.assert_allclose(seen['df']['day7'].to_numpy(), expected.to_numpy())


def test_explicit_aggregation_func_is_used(data, fake_fc):
    calls = []

    def agg(df, design):
        calls.append(df.shape)
        from catbridge.preprocess import repeat_aggregation_mean
        return repeat_aggregation_mean(df, design)

    compute_corr(*_args(data), TARGET, None, aggregation_func=agg, metrics=['Pearson'])
    assert len(calls) == 2   # gene and metabolite


def test_clustering_uses_aggregated_genes(data, fake_fc, monkeypatch):
    """ts_clustering must see the 6 aggregated time points, not the 18 raw replicate columns."""
    seen = {}

    def fake_ts(df, n_clusters, random_state=0, **kw):
        seen.update(cols=list(df.columns), n=n_clusters, rs=random_state)
        return pd.Series(np.arange(len(df)) % n_clusters, index=df.index, name='Cluster')

    monkeypatch.setattr(pl, 'ts_clustering', fake_ts)
    out = compute_corr(*_args(data), TARGET, 4, metrics=['Pearson'], random_state=7)
    assert seen == dict(cols=GROUPS, n=4, rs=7)
    assert 'Cluster' in out.columns and out['Cluster'].notna().all()
    assert len(out) == N_GENES


@pytest.mark.parametrize("cc", [None, 0])
def test_no_clustering_when_cluster_count_falsy(data, fake_fc, monkeypatch, cc):
    monkeypatch.setattr(pl, 'ts_clustering',
                        lambda *a, **k: pytest.fail("ts_clustering must not be called"))
    out = compute_corr(*_args(data), TARGET, cc, metrics=['Pearson'])
    assert 'Cluster' not in out.columns
    # compute_score copes without a Cluster column
    assert 'Score' in compute_score(out, 'Pearson').columns


def test_annotation_only_genes_do_not_add_rows(data, tmp_path):
    ann = pd.read_csv(data['annotation'], sep='\t', index_col=0)
    ann.loc['NOT_IN_GENES'] = 'capsaicin synthase'
    p = tmp_path / 'ann_extra.tsv'
    ann.to_csv(p, sep='\t')
    out = compute_corr(data['gene_mean'], data['metabolite_mean'], None, str(p), TARGET, None,
                       metrics=['Pearson'])
    assert len(out) == N_GENES and 'NOT_IN_GENES' not in set(out['Name'])


# ------------------------------------------------------------------ pipeline()
def _planted_rank(res):
    return int(res.index[res['Name'] == TRUE_GENE][0])


def test_pipeline_no_design(data):
    res = pipeline(data['gene_mean'], data['metabolite_mean'], None, data['annotation'],
                   TARGET, None)
    assert res.index.name == 'Rank'
    assert list(res.index) == list(range(1, N_GENES + 1))
    assert {'Name', 'Granger', 'log2FoldChange', 'Score'} <= set(res.columns)
    assert res['Score'].is_monotonic_decreasing
    # Granger with only 6 time points is weak (and G00-G02 are, by construction, genuine
    # one-step leaders too): the planted gene ranks 4-5 of 40 over 20 seeds -> top 25 %
    assert _planted_rank(res) <= 0.25 * N_GENES
    # Pearson is the sharper metric here: rank 1 for all 20 seeds -> top 10 %
    res_p = pipeline(data['gene_mean'], data['metabolite_mean'], None, data['annotation'],
                     TARGET, None, score_col='Pearson')
    assert _planted_rank(res_p) <= 0.10 * N_GENES


def test_pipeline_with_design_fake_fc(data, fake_fc):
    res = pipeline(data['gene'], data['metabolite'], data['design'], data['annotation'],
                   TARGET, None)
    assert res.index.name == 'Rank' and res.index[0] == 1
    assert len(res) == N_GENES
    assert _planted_rank(res) <= 0.25 * N_GENES
    res_p = pipeline(data['gene'], data['metabolite'], data['design'], data['annotation'],
                     TARGET, None, score_col='Pearson')
    assert _planted_rank(res_p) <= 0.10 * N_GENES


def test_pipeline_with_design_real_deseq2(data, deseq_ok):
    res = pipeline(data['gene'], data['metabolite'], data['design'], data['annotation'],
                   TARGET, None)
    assert res.index.name == 'Rank' and res.index[0] == 1
    assert _planted_rank(res) <= 0.25 * N_GENES
    # direction of the DESeq2 fold change: planted gene is up at day30 vs day50
    assert res.set_index('Name').loc[TRUE_GENE, 'log2FoldChange'] > 0


def test_compute_corr_with_design_real_deseq2(data, deseq_ok):
    out = compute_corr(*_args(data), TARGET, None)
    assert len(out) == N_GENES and out['Name'].is_unique
    for c in EXPECTED_METRIC_COLS + ['log2FoldChange', 'Description', 'Description_Score']:
        assert c in out.columns
    assert 'Cluster' not in out.columns
    assert out.set_index('Name').loc[TRUE_GENE, 'log2FoldChange'] > 0


def test_design_path_with_integer_gene_names(data, tmp_path, deseq_ok):
    """pydeseq2 turns gene names into str; the merge on 'Name' must still work for int ids."""
    gene = pd.read_csv(data['gene'], sep='\t', index_col=0)
    gene.index = pd.Index(np.arange(1000, 1000 + len(gene)), name='Name')
    gp = tmp_path / 'gene_int.tsv'
    gene.to_csv(gp, sep='\t')
    out = compute_corr(str(gp), data['metabolite'], data['design'], None, TARGET, None,
                       metrics=['Pearson'])
    assert len(out) == N_GENES and out['Name'].is_unique
    assert out['Name'].map(type).eq(int).all() or pd.api.types.is_integer_dtype(out['Name'])
    assert out['log2FoldChange'].notna().sum() > N_GENES // 2
    assert out['Pearson'].notna().all()           # not shifted into separate (all-NaN) rows


def test_pipeline_with_clustering(data, tslearn_ok):
    res = pipeline(data['gene_mean'], data['metabolite_mean'], None, data['annotation'],
                   TARGET, 3)
    assert 'Cluster' in res.columns and res['Cluster'].between(0, 2).all()


def test_pipeline_granger_small_p_ranks_higher(data):
    res = pipeline(data['gene_mean'], data['metabolite_mean'], None, None, TARGET, None,
                   score_col='Granger')
    ok = res.dropna(subset=['Granger'])
    assert len(ok) > 10
    # Score must increase with evidence = -log10(p) (the old code had the opposite direction)
    from scipy.stats import spearmanr
    rho = spearmanr(ok['Score'], -np.log10(ok['Granger'].clip(lower=1e-300)))[0]
    assert rho > 0
    # and not the other way round
    assert spearmanr(ok['Score'], ok['Granger'])[0] < rho


def test_pipeline_granger_nan_is_not_best(data, monkeypatch):
    """An untestable gene (NaN p-value) must not be filled with p = 0 (= most significant)."""
    gene = pd.read_csv(data['gene_mean'], sep='\t', index_col=0)
    gene.loc['G10'] = 5.0                       # constant series: Granger cannot be tested
    p = data['gene_mean'].replace('gene_mean', 'gene_const')
    gene.to_csv(p, sep='\t')
    res = pipeline(p, data['metabolite_mean'], None, None, TARGET, None, score_col='Granger')
    row = res[res['Name'] == 'G10'].iloc[0]
    assert np.isnan(row['Granger'])
    assert int(res.index[res['Name'] == 'G10'][0]) > N_GENES // 2


def test_pipeline_other_score_col_and_kwargs(data):
    res = pipeline(data['gene_mean'], data['metabolite_mean'], None, data['annotation'],
                   TARGET, None, max_lag=2, score_col='CCM', E=2)
    assert res.index.name == 'Rank' and 'CCM' in res.columns
    with pytest.raises(ValueError):
        pipeline(data['gene_mean'], data['metabolite_mean'], None, None, TARGET, None,
                 score_col='NoSuchMetric')
    with pytest.raises(TypeError):
        pipeline(data['gene_mean'], data['metabolite_mean'], None, None, TARGET, None, lag=3)


def test_pipeline_matches_compute_score_of_compute_corr(data):
    corr = compute_corr(data['gene_mean'], data['metabolite_mean'], None, data['annotation'],
                        TARGET, None, lag=1, metrics=['Granger', 'Pearson'])
    want = compute_score(corr, 'Granger')
    got = pipeline(data['gene_mean'], data['metabolite_mean'], None, data['annotation'],
                   TARGET, None)
    pd.testing.assert_frame_equal(got, want.rename_axis('Rank'))
