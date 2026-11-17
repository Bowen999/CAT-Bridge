"""Dirty inputs: each one is either handled (with a warning saying what was done) or rejected
with a message that names the problem. None of them may crash deep inside pandas / numpy /
a third-party library, or silently give a wrong result."""
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from catbridge import data_io, foldchange as fcm, metrics, pipeline as pl
from catbridge.data_io import read_upload, get_target
from catbridge.pipeline import compute_corr

pytestmark = pytest.mark.filterwarnings('ignore:CCM skipped', 'ignore:Granger')

GROUPS = ['day0', 'day7', 'day16', 'day30', 'day50', 'day60']
COLS = [f"{g}_{r}" for g in GROUPS for r in (1, 2, 3)]


def _tables(seed=0):
    rng = np.random.default_rng(seed)
    design = pd.DataFrame({'group': [c.rsplit('_', 1)[0] for c in COLS]},
                          index=pd.Index(COLS, name='sample'))
    genes = pd.DataFrame(rng.poisson(200, size=(20, len(COLS))).astype(float),
                         index=pd.Index([f'G{i:02d}' for i in range(20)], name='Name'), columns=COLS)
    prof = np.repeat([1, 2, 4, 10, 5, 2.0], 3) * 1e5
    met = pd.DataFrame([prof * rng.lognormal(0, .05, len(COLS)) for _ in range(3)],
                       index=pd.Index(['Target', 'MetA', 'MetB'], name='Name'), columns=COLS)
    ann = pd.DataFrame({'Description': ['putative kinase'] * 20}, index=genes.index)
    return genes, met, design, ann


def _means(genes, met, design):
    g = design['group'].to_numpy()
    return (genes.T.groupby(g, sort=False).mean().T.round(), met.T.groupby(g, sort=False).mean().T)


def _write(tmp_path, name, df, sep='\t', ext='tsv', **kw):
    p = tmp_path / f'{name}.{ext}'
    df.to_csv(p, sep=sep, **kw)
    return str(p)


def _fake_fc(counts_df, metadata, contrast=None, shrink=True, n_cpus=None):
    peak, other = contrast
    g = metadata.loc[list(counts_df.columns), 'group']
    m1 = counts_df.loc[:, (g == peak).to_numpy()].mean(axis=1)
    m0 = counts_df.loc[:, (g == other).to_numpy()].mean(axis=1)
    out = pd.DataFrame({'log2FoldChange': np.log2((m1 + 1) / (m0 + 1))})
    out.index.name = 'Name'
    return out


@pytest.fixture
def files(tmp_path):
    genes, met, design, ann = _tables()
    gm, mm = _means(genes, met, design)
    return dict(gene=_write(tmp_path, 'gene', genes), met=_write(tmp_path, 'met', met),
                design=_write(tmp_path, 'design', design), ann=_write(tmp_path, 'ann', ann),
                gene_mean=_write(tmp_path, 'gene_mean', gm), met_mean=_write(tmp_path, 'met_mean', mm),
                tables=(genes, met, design, ann), means=(gm, mm), dir=tmp_path)


@pytest.fixture
def fake_fc(monkeypatch):
    monkeypatch.setattr(pl, 'fc_comp', _fake_fc)


def _nodesign(files, gene=None, met=None, ann=None, **kw):
    return compute_corr(gene or files['gene_mean'], met or files['met_mean'], None, ann,
                        'Target', None, **kw)


# ================================================================== read_upload
def test_comment_trailer_rows_are_dropped(tmp_path):
    p = tmp_path / 'a.tsv'
    p.write_text('Name\tDescription\nG1\tkinase\nG2\t\n## 32703 queries scanned\t\n'
                 '## Total time (seconds): 108.3\t\n')
    with pytest.warns(UserWarning, match=r"dropped 2 comment/empty row"):
        a = read_upload(p)
    assert list(a.index) == ['G1', 'G2']            # G2 (no description) is kept


def test_pepper_style_annotation_trailer_not_a_gene(files, tmp_path):
    genes, met, design, ann = files['tables']
    p = _write(tmp_path, 'ann2', ann)
    with open(p, 'a') as f:
        f.write('## 32703 queries scanned\t\n')
    with pytest.warns(UserWarning, match='comment'):
        res = _nodesign(files, ann=p)
    assert not res['Name'].astype(str).str.startswith('#').any() and len(res) == 20


def test_path_object_and_uppercase_extension(files, tmp_path):
    gm, mm = files['means']
    gp = tmp_path / 'G.TSV'
    gm.to_csv(gp, sep='\t')
    res = compute_corr(gp, pathlib.Path(files['met_mean']), None, None, 'Target', None)
    assert len(res) == 20


def test_compressed_file(files, tmp_path):
    gm, _ = files['means']
    p = tmp_path / 'g.tsv.gz'
    gm.to_csv(p, sep='\t')
    pd.testing.assert_frame_equal(read_upload(p), read_upload(files['gene_mean']))


@pytest.mark.parametrize("ext,sep", [('csv', '\t'), ('tsv', ','), ('tsv', ';'), ('txt', ',')])
def test_delimiter_that_does_not_match_the_extension(files, tmp_path, ext, sep):
    gm, _ = files['means']
    p = tmp_path / f'g.{ext}'
    gm.to_csv(p, sep=sep)
    with pytest.warns(UserWarning, match='separated'):
        df = read_upload(p)
    pd.testing.assert_frame_equal(df, gm, check_names=False)


def test_empty_file_and_header_only(files, tmp_path):
    p = tmp_path / 'empty.tsv'
    p.write_text('')
    with pytest.raises(ValueError, match='empty'):
        read_upload(p)
    gm, _ = files['means']
    h = _write(tmp_path, 'header', gm.iloc[:0])
    with pytest.raises(ValueError, match='no rows'):
        _nodesign(files, gene=h)


def test_unsupported_and_missing_file(tmp_path):
    with pytest.raises(ValueError, match='Unsupported'):
        read_upload(tmp_path / 'x.json')
    with pytest.raises(FileNotFoundError):
        read_upload(tmp_path / 'missing.tsv')


def test_duplicated_column_names_are_an_error(files, tmp_path):
    gm, _ = files['means']
    g = gm.copy()
    g.columns = ['day0', 'day0'] + list(g.columns[2:])
    p = _write(tmp_path, 'dup', g)
    with pytest.raises(ValueError, match=r"duplicated column names: \['day0'\]"):
        read_upload(p)


def test_spaces_around_names_are_removed(files, tmp_path, fake_fc):
    genes, met, design, ann = files['tables']
    g = genes.copy()
    g.columns = [' ' + c if i % 2 else c + ' ' for i, c in enumerate(g.columns)]
    g.index = [n + ' ' for n in g.index]
    p = _write(tmp_path, 'g_sp', g)
    with pytest.warns(UserWarning, match='spaces'):
        res = compute_corr(p, files['met'], files['design'], files['ann'], 'Target', None)
    assert res['Description'].notna().all()          # names match the annotation again


def test_trailing_delimiters_give_no_extra_column(files, tmp_path, fake_fc):
    p = tmp_path / 'g_trail.tsv'
    files['tables'][0].to_csv(p, sep='\t')
    p.write_text('\n'.join(line + '\t' for line in p.read_text().splitlines()) + '\n')
    with pytest.warns(UserWarning, match='empty column'):
        res = compute_corr(str(p), files['met'], files['design'], None, 'Target', None)
    assert len(res) == 20


def test_integer_headers_from_excel_match_text_headers(files, tmp_path):
    pytest.importorskip('openpyxl')
    gm, mm = files['means']
    gm, mm = gm.copy(), mm.copy()
    gm.columns = range(6)
    mm.columns = range(6)
    gp = tmp_path / 'g.xlsx'
    gm.to_excel(gp)
    mp = _write(tmp_path, 'm_int', mm)
    assert len(compute_corr(str(gp), mp, None, None, 'Target', None)) == 20


# ================================================================== table contents
@pytest.mark.parametrize("design", [True, False])
def test_non_numeric_cell_names_the_cell(files, tmp_path, design):
    genes = (files['tables'][0] if design else files['means'][0]).astype(object)
    genes.iloc[3, 4] = '1,5'
    p = _write(tmp_path, 'g_txt', genes)
    with pytest.raises(ValueError, match=r"non-numeric.*row 'G03', column '[^']+': '1,5'"):
        if design:
            compute_corr(p, files['met'], files['design'], None, 'Target', None)
        else:
            _nodesign(files, gene=p)


def test_non_numeric_target_value(files, tmp_path):
    met = files['means'][1].astype(object)
    met.iloc[0, 2] = 'n.d.'
    p = _write(tmp_path, 'm_txt', met)
    with pytest.raises(ValueError, match=r"metabolite table .*'n\.d\.'"):
        _nodesign(files, met=p)


def test_infinite_value(files, tmp_path):
    gm = files['means'][0].copy()
    gm.iloc[1, 1] = np.inf
    with pytest.raises(ValueError, match='infinite'):
        _nodesign(files, gene=_write(tmp_path, 'g_inf', gm))


@pytest.mark.parametrize("design", [True, False])
def test_missing_name_column(files, tmp_path, design):
    genes = files['tables'][0] if design else files['means'][0]
    p = _write(tmp_path, 'g_noname', genes, index=False)
    with pytest.raises(ValueError, match='no name column'):
        if design:
            compute_corr(p, files['met'], files['design'], None, 'Target', None)
        else:
            _nodesign(files, gene=p)


def test_missing_name_column_with_integer_counts(files, tmp_path):
    gm = files['means'][0].astype(int)
    p = _write(tmp_path, 'g_noname_int', gm, index=False)
    with pytest.raises(ValueError, match='no name column'):
        _nodesign(files, gene=p)


def test_name_column_with_another_header_works(files, tmp_path):
    gm = files['means'][0].copy()
    gm.index.name = 'gene_id'
    res = _nodesign(files, gene=_write(tmp_path, 'g_id', gm), ann=files['ann'])
    assert res['Description'].notna().all()


def test_blank_gene_name(files, tmp_path):
    gm = files['means'][0].copy()
    gm.index = [''] + list(gm.index[1:])
    with pytest.raises(ValueError, match='without a name'):
        _nodesign(files, gene=_write(tmp_path, 'g_blank', gm))


def test_duplicated_gene_names(files, tmp_path):
    gm = files['means'][0].copy()
    gm.index = list(gm.index[:-1]) + [gm.index[0]]
    with pytest.raises(ValueError, match='unique'):
        _nodesign(files, gene=_write(tmp_path, 'g_dup', gm))


def test_all_nan_gene_without_design_is_nan_row(files, tmp_path):
    gm = files['means'][0].copy()
    gm.iloc[5] = np.nan
    res = _nodesign(files, gene=_write(tmp_path, 'g_nan', gm)).set_index('Name')
    assert res.loc['G05', ['Pearson', 'DTW', 'log2FoldChange']].isna().all()
    assert res.drop('G05')['Pearson'].notna().all()


def test_missing_counts_for_deseq2_name_the_genes():
    counts = pd.DataFrame(np.arange(24.).reshape(4, 6) + 10, index=list('abcd'),
                          columns=[f's{i}' for i in range(6)])
    counts.iloc[2, 1] = np.nan
    meta = pd.DataFrame({'group': ['x'] * 3 + ['y'] * 3}, index=counts.columns)
    pytest.importorskip('pydeseq2')
    with pytest.raises(ValueError, match=r"1 gene\(s\) have missing counts.*\['c'\]"):
        fcm.fc_comp(counts, meta, contrast=('x', 'y'))


def test_negative_values_fold_change_message(files, tmp_path):
    gm = np.log2(files['means'][0] + 1) - 8
    with pytest.raises(ValueError, match='not log-transformed'):
        _nodesign(files, gene=_write(tmp_path, 'g_log', gm))


@pytest.mark.parametrize("n", [2, 3])
def test_few_time_points_warn(files, tmp_path, n):
    gm, mm = files['means']
    gp = _write(tmp_path, 'g_few', gm.iloc[:, 2:2 + n])
    mp = _write(tmp_path, 'm_few', mm.iloc[:, 2:2 + n])
    with pytest.warns(UserWarning, match=f'Only {n} time points'):
        res = compute_corr(gp, mp, None, None, 'Target', None)
    assert len(res) == 20


# ================================================================== target
def test_target_duplicated(files, tmp_path):
    mm = files['means'][1].copy()
    mm.index = ['Target', 'Target', 'MetB']
    with pytest.raises(ValueError, match="occurs 2 times"):
        _nodesign(files, met=_write(tmp_path, 'm_dup', mm))


def test_target_missing_suggests_names(files):
    with pytest.raises(KeyError, match="Similar names: \\['Target'\\]"):
        compute_corr(files['gene_mean'], files['met_mean'], None, None, 'target', None)


def test_target_constant(files, tmp_path):
    mm = files['means'][1].copy()
    mm.iloc[0] = 5.0
    with pytest.raises(ValueError, match='constant'):
        _nodesign(files, met=_write(tmp_path, 'm_const', mm))


def test_target_missing_time_point(files, tmp_path):
    mm = files['means'][1].copy()
    mm.iloc[0, 1] = np.nan
    with pytest.raises(ValueError, match=r"no value at time point\(s\) \['day7'\]"):
        _nodesign(files, met=_write(tmp_path, 'm_nan', mm))


# ================================================================== design
def test_design_sample_names_do_not_match(files, tmp_path):
    design = files['tables'][2].copy()
    design.index = [c.replace('day', 'Day') for c in design.index]
    with pytest.raises(ValueError, match=r"not samples of the design table.*similar design samples"):
        compute_corr(files['gene'], files['met'], _write(tmp_path, 'd_bad', design), None,
                     'Target', None)


def test_design_group_column_capitalised_is_accepted(files, tmp_path, fake_fc):
    design = files['tables'][2].rename(columns={'group': 'Group'})
    with pytest.warns(UserWarning, match="'Group' as 'group'"):
        res = compute_corr(files['gene'], files['met'], _write(tmp_path, 'd_G', design), None,
                           'Target', None)
    assert len(res) == 20


def test_design_without_group_column(files, tmp_path):
    design = files['tables'][2].rename(columns={'group': 'condition'})
    with pytest.raises(ValueError, match=r"needs a column named 'group'.*\['condition'\]"):
        compute_corr(files['gene'], files['met'], _write(tmp_path, 'd_nog', design), None,
                     'Target', None)


def test_design_sample_without_group(files, tmp_path):
    design = files['tables'][2].copy()
    design.iloc[4, 0] = np.nan
    with pytest.raises(ValueError, match=r"without a group: \['day7_2'\]"):
        compute_corr(files['gene'], files['met'], _write(tmp_path, 'd_nan', design), None,
                     'Target', None)


def test_design_duplicated_sample(files, tmp_path):
    design = files['tables'][2]
    design = pd.concat([design, design.iloc[[0]]])
    with pytest.raises(ValueError, match=r"more than once: \['day0_1'\]"):
        compute_corr(files['gene'], files['met'], _write(tmp_path, 'd_dup', design), None,
                     'Target', None)


def test_design_peak_group_with_one_replicate(files, tmp_path):
    pytest.importorskip('pydeseq2')
    genes, met, design, _ = files['tables']
    drop = ['day30_2', 'day30_3']
    with pytest.raises(ValueError, match="fewer than 2 samples"):
        compute_corr(_write(tmp_path, 'g1', genes.drop(columns=drop)),
                     _write(tmp_path, 'm1', met.drop(columns=drop)),
                     _write(tmp_path, 'd1', design.drop(index=drop)), None, 'Target', None)


# ================================================================== annotation
def test_annotation_names_that_match_no_gene_warn(files, tmp_path):
    ann = files['tables'][3].copy()
    ann.index = [n + '.1' for n in ann.index]
    with pytest.warns(UserWarning, match='None of the 20 annotation names matches a gene name'):
        res = _nodesign(files, ann=_write(tmp_path, 'a_bad', ann))
    assert (res['Description_Score'] == 0.1).all()


def test_annotation_integer_ids_match_text_ids(tmp_path):
    rng = np.random.default_rng(1)
    gm = pd.DataFrame(rng.poisson(100, (8, 6)).astype(float), index=pd.Index(range(100, 108), name='Name'),
                      columns=GROUPS)
    mm = pd.DataFrame([[1, 2, 4, 10, 5, 2.0]], index=['Target'], columns=GROUPS)
    ann = pd.DataFrame({'Description': ['kinase'] * 8}, index=[f'{i}' for i in range(100, 108)])
    gp = _write(tmp_path, 'g', gm)
    mp = _write(tmp_path, 'm', mm)
    # integer gene ids in the gene table, the same ids as text in the annotation (earlier development version: no
    # match at all -> every gene silently got the 'missing description' score)
    ann_text = ann.copy()
    ann_text.index = ann_text.index.astype(object)
    res = pl._merge_annotation(pd.DataFrame({'Name': gm.index.to_numpy()}), ann_text)
    assert res['Description'].notna().all() and res['Name'].tolist() == list(range(100, 108))
    # through compute_corr, with ids pandas reads as text ('id100')
    ap = tmp_path / 'a.tsv'
    ap.write_text('Name\tDescription\n' + ''.join(f'id{i}\tkinase\n' for i in range(100, 108)))
    with pytest.warns(UserWarning, match='None of the 8 annotation names'):
        compute_corr(gp, mp, None, str(ap), 'Target', None)


def test_annotation_description_column(files, tmp_path):
    ann = files['tables'][3].rename(columns={'Description': 'description'})
    with pytest.warns(UserWarning, match="'description' as 'Description'"):
        res = _nodesign(files, ann=_write(tmp_path, 'a_lc', ann))
    assert res['Description'].notna().all()
    ann = files['tables'][3].rename(columns={'Description': 'Annotation'})
    with pytest.raises(ValueError, match=r"needs a column named 'Description'.*\['Annotation'\]"):
        _nodesign(files, ann=_write(tmp_path, 'a_none', ann))


# ================================================================== clustering
def test_more_clusters_than_genes(files):
    with pytest.raises(ValueError, match=r"20 usable series for n_clusters=50"):
        compute_corr(files['gene_mean'], files['met_mean'], None, None, 'Target', 50)


def test_gene_with_missing_value_is_not_clustered(files, tmp_path):
    gm = files['means'][0].copy()
    gm.iloc[3, 2] = np.nan
    with pytest.warns(UserWarning, match='not clustered'):
        res = compute_corr(_write(tmp_path, 'g_nan1', gm), files['met_mean'], None, None,
                           'Target', 3)
    res = res.set_index('Name')
    assert np.isnan(res.loc['G03', 'Cluster'])
    assert res.drop('G03')['Cluster'].notna().all()
    assert res.drop('G03')['Cluster'].nunique() == 3     # (earlier development version: NaN rows broke the clustering)


def test_clustering_without_tslearn_causal_ccm_fastdtw(files):
    """The metrics and the clustering run with these packages blocked."""
    import subprocess
    code = (
        "import sys\n"
        "for m in ['tslearn', 'causal_ccm', 'fastdtw']:\n"
        "    sys.modules[m] = None\n"
        "from catbridge import catbridge as cat\n"
        f"r = cat.compute_corr({files['gene_mean']!r}, {files['met_mean']!r}, None, None, 'Target', 3)\n"
        "assert r['Cluster'].nunique() == 3 and r['DTW'].notna().all()\n"
        "print('ok')\n"
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=repo,
                       capture_output=True, text=True)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr


# ================================================================== environment problems
def test_pydeseq2_missing_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pydeseq2', None)
    monkeypatch.setitem(sys.modules, 'pydeseq2.dds', None)
    with pytest.raises(ImportError, match='pip install pydeseq2'):
        fcm._import_pydeseq2()


def _counts():
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(rng.poisson(100, (30, 6)), index=[f'g{i}' for i in range(30)],
                          columns=[f's{i}' for i in range(6)])
    meta = pd.DataFrame({'group': ['x'] * 3 + ['y'] * 3}, index=counts.columns)
    return counts, meta


def test_pandas3_incompatibility_is_explained(monkeypatch):
    """Simulated: pydeseq2 fails inside DeseqDataSet under pandas >= 3."""
    class Boom:
        def __init__(self, *a, **k):
            raise TypeError("Invalid value '[ True ...]' for dtype 'float64'")

    monkeypatch.setattr(fcm, '_import_pydeseq2', lambda: (Boom, Boom))
    monkeypatch.setattr(fcm.pd, '__version__', '3.0.0')
    counts, meta = _counts()
    with pytest.raises(RuntimeError, match=r'pandas<3.*NOT computed; no other method'):
        fcm.fc_comp(counts, meta, contrast=('x', 'y'))
    # with pandas < 3 the original error is not reworded
    monkeypatch.setattr(fcm.pd, '__version__', '2.2.3')
    with pytest.raises(TypeError, match='Invalid value'):
        fcm.fc_comp(counts, meta, contrast=('x', 'y'))


def test_pandas3_incompatibility_real():
    """In an environment with pandas >= 3 and pydeseq2 0.5.x the real failure is explained."""
    if int(pd.__version__.split('.')[0]) < 3:
        pytest.skip('pandas < 3: DESeq2 works here')
    pytest.importorskip('pydeseq2')
    counts, meta = _counts()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fcm.fc_comp(counts, meta, contrast=('x', 'y'))
    except RuntimeError as e:
        assert 'pandas<3' in str(e)
    else:
        pytest.skip('this pydeseq2 works with pandas 3')


# ================================================================== progress / options
def test_verbose_prints_steps(files, capsys):
    _nodesign(files, verbose=True)
    err = capsys.readouterr().err
    for step in ('reading the gene table', 'Granger', 'DTW', 'Pearson', 'done'):
        assert step in err
    _nodesign(files)
    assert capsys.readouterr().err == ''


def test_dtw_method_option(files):
    a = _nodesign(files)['DTW']
    b = _nodesign(files, dtw_method='exact')['DTW']
    assert (b <= a + 1e-12).all()
    with pytest.raises(ValueError, match='dtw_method'):
        _nodesign(files, dtw_method='fast')


def test_metric_inputs_with_text_give_clear_error():
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': ['x', 3.0], 'c': [1.0, 0.5]}, index=['g1', 'g2'])
    with pytest.raises(ValueError, match=r"non-numeric.*\['b'\]"):
        metrics.compute_pearson(df, [1, 2, 3])
