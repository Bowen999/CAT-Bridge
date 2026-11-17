"""API-compat tests for the split of catbridge/catbridge.py into sub-modules.

The baseline commit (deebb82) is the reference for the public API: every
top-level ``def`` and every top-level assigned constant it defined must still
be reachable as ``catbridge.catbridge.<name>``.
"""
import ast
import os
import subprocess
import sys

import pytest

BASELINE = "deebb82"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _baseline_public_names():
    try:
        src = subprocess.run(
            ["git", "show", f"{BASELINE}:catbridge/catbridge.py"],
            cwd=REPO, check=True, capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("baseline commit not available in this checkout")
    tree = ast.parse(src)
    funcs, consts = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            funcs.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    consts.append(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            consts.append(node.target.id)
    funcs = sorted({n for n in funcs if not n.startswith("_")})
    consts = sorted({n for n in consts if not n.startswith("_")})
    return funcs, consts


def test_import_package():
    import catbridge
    from catbridge import catbridge as cat  # noqa: F401


def test_all_old_functions_available_and_callable():
    from catbridge import catbridge as cat
    funcs, _ = _baseline_public_names()
    assert funcs, "no functions parsed from baseline"
    missing = [n for n in funcs if not callable(getattr(cat, n, None))]
    assert not missing, f"missing/not callable: {missing}"


def test_all_old_constants_available():
    from catbridge import catbridge as cat
    _, consts = _baseline_public_names()
    assert "keywords_scores" in consts
    missing = [n for n in consts if not hasattr(cat, n)]
    assert not missing, f"missing constants: {missing}"
    assert cat.keywords_scores == {'ase': 0.2, 'enzyme': 0.2, 'synthase': 0.2}


def test_function_homes():
    """Each function lives in the module the split plan assigned it to."""
    import catbridge.data_io as d, catbridge.preprocess as p, catbridge.metrics as m
    import catbridge.foldchange as f, catbridge.scoring as s, catbridge.plotting as pl
    import catbridge.llm as l, catbridge.pipeline as pp
    homes = {
        d: ["read_upload", "get_target"],
        p: ["normalize", "scaling", "scale_df", "scale_column", "repeat_aggregation_max",
            "repeat_aggregation_mean", "merge_dataframes", "calculate_trend_strength", "merge_and_reduce"],
        m: ["compute_granger", "compute_reverse_granger", "granger_list", "compute_ccm", "compute_cca",
            "compute_dtw", "compute_ccf", "compute_spearman", "compute_pearson", "clustering", "ts_clustering"],
        f: ["find_noontide", "df_for_fc", "no_repeat_fc", "fc_comp", "compute_fc"],
        s: ["score", "compute_score", "keywords_scores", "add_annotation_score", "trend_strength",
            "top_important_features", "compute_modulus"],
        l: ["Yuanfang"],
        pp: ["pipeline", "compute_corr"],
    }
    for mod, names in homes.items():
        for n in names:
            assert n in mod.__all__ and hasattr(mod, n), (mod.__name__, n)
    for n in pl.__all__:
        assert n.startswith("plot_") or n == "save_table_as_svg"


def test_import_without_heavy_optional_deps():
    """`import catbridge` must not need openai/datashader/bioinfokit/pydeseq2/tslearn/skfuzzy/adjustText.

    (PIL is not blocked: matplotlib itself requires it; catbridge no longer imports it directly.)
    """
    code = (
        "import sys\n"
        "for m in ['openai','datashader','bioinfokit','pydeseq2','tslearn','skfuzzy','adjustText']:\n"
        "    sys.modules[m] = None\n"
        "import catbridge\n"
        "from catbridge import catbridge as cat\n"
        "assert callable(cat.compute_corr)\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
