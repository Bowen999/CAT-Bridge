"""Smoke / direction tests for catbridge.plotting (Agg backend, nothing is shown)."""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from catbridge.plotting import plot_line_heatmap, plot_result  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _result(**cols):
    n = 6
    df = pd.DataFrame({'Name': [f'g{i}' for i in range(n)],
                       'Score': np.linspace(1, 0, n),
                       'log2FoldChange': np.linspace(-1, 2, n)})
    for k, v in cols.items():
        df[k] = v
    return df


def _colour_norm_and_label(df, col, tmp_path):
    out = tmp_path / 'p.png'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plot_result(df, 'Score', 'log2FoldChange', col, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0
    fig = plt.gcf()
    sc = fig.axes[0].collections[0]
    label = fig.axes[1].get_ylabel()          # colour bar axis
    return np.asarray(sc.get_array(), dtype=float), label


def test_plot_result_granger_pvalue_small_p_gets_highest_colour(tmp_path):
    df = _result(Granger=[0.5, 0.2, 0.05, 0.01, 0.001, 1.0])
    c, label = _colour_norm_and_label(df, 'Granger', tmp_path)
    assert int(np.argmax(c)) == 4 and int(np.argmin(c)) == 5      # p=0.001 strongest, p=1 weakest
    assert 'log10' in label


def test_plot_result_dtw_small_distance_gets_highest_colour(tmp_path):
    df = _result(DTW=[3.0, 1.0, 2.0, 0.5, 4.0, 2.5])
    c, label = _colour_norm_and_label(df, 'DTW', tmp_path)
    assert int(np.argmax(c)) == 3 and int(np.argmin(c)) == 4      # 0.5 strongest, 4.0 weakest
    assert 'DTW' in label and 'distance' in label


def test_plot_result_higher_is_better_column_unchanged(tmp_path):
    df = _result(Pearson=[0.1, -0.5, 0.9, 0.3, 0.0, 0.2])
    c, label = _colour_norm_and_label(df, 'Pearson', tmp_path)
    assert int(np.argmax(c)) == 2 and int(np.argmin(c)) == 1
    assert label == 'Pearson'


def test_plot_result_does_not_modify_input_and_missing_column_raises(tmp_path):
    df = _result(DTW=[3.0, 1.0, 2.0, 0.5, 4.0, np.nan])
    before = df.copy()
    _colour_norm_and_label(df, 'DTW', tmp_path)
    pd.testing.assert_frame_equal(df, before)
    with pytest.raises(ValueError):
        plot_result(df, 'Score', 'log2FoldChange', 'Nope')


def test_plot_line_heatmap_runs_with_wide_table(tmp_path):
    # input = a wide table: Name, the four metric columns, then the time-point columns
    df = pd.DataFrame({'Name': ['a', 'b'], 'Granger': [0.01, 0.4], 'log2FoldChange': [2.0, -1.0],
                       'Pearson': [0.9, 0.1], 'Spearman': [0.8, 0.2],
                       't0': [1.0, 2.0], 't1': [3.0, 2.0], 't2': [5.0, 1.0]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plot_line_heatmap(df, 'a')
    assert len(plt.get_fignums()) == 1
