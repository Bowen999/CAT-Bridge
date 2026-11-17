"""Data input helpers: reading uploaded tables and extracting a target row."""

import difflib
import os
import warnings

import numpy as np
import pandas as pd

__all__ = [
    'read_upload',
    'get_target',
]

_TEXT_SEP = {'csv': ',', 'txt': '\t', 'tsv': '\t'}
_COMPRESSION = ('.gz', '.bz2', '.xz', '.zip', '.zst')


def _extension(path):
    """Lower-case extension, ignoring a compression suffix (gene.tsv.gz -> 'tsv')."""
    low = path.lower()
    for suffix in _COMPRESSION:
        if low.endswith(suffix):
            low = low[:-len(suffix)]
            break
    return os.path.splitext(low)[1].lstrip('.')


def _first_line(path):
    """First non-empty line of a (possibly compressed) text file, or '' ."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            head = pd.read_csv(path, sep='\x01', header=None, nrows=1, dtype=str,
                               engine='python', skip_blank_lines=True)
        return '' if head.empty else str(head.iloc[0, 0])
    except Exception:
        return ''


def _is_text(dtype):
    """object or string dtype (pandas 3 reads text as the 'str' dtype)."""
    return dtype == object or pd.api.types.is_string_dtype(dtype)


def _read_text(path, sep):
    return pd.read_csv(path, sep=sep, index_col=0)


def _header(path, ext, sep):
    """Raw header names (pandas renames duplicates to 'x.1', so read them unparsed)."""
    if ext in _TEXT_SEP:
        h = pd.read_csv(path, sep=sep, header=None, nrows=1, dtype=str)
    else:
        h = pd.read_excel(path, header=None, nrows=1, dtype=str)
    return ['' if pd.isna(v) else str(v) for v in h.iloc[0].tolist()] if len(h) else []


def read_upload(file_name):
    """
    Read the uploaded file and return a dataframe (first column = row names / index).

    Supported formats: csv, txt / tsv (tab-separated), xls, xlsx; the extension is
    case-insensitive and may be followed by a compression suffix (``gene.tsv.gz``).
    ``file_name`` may be a str or a path object.

    Clean-ups (each one issues a warning saying what was done):
      * a text file whose content uses another delimiter than its extension says (e.g. a
        tab-separated ``.csv``) is read with the delimiter that is actually used;
      * leading / trailing spaces are removed from column names and row names;
      * empty columns without a header (trailing delimiters, ``Unnamed: n``) are dropped;
      * comment / trailer rows -- a row name starting with ``#`` (e.g. the
        ``## 32703 queries scanned`` lines at the end of eggNOG-mapper output) or no row name,
        and no value in any column -- are dropped.

    Raises:
        ValueError: unsupported extension, empty file, or duplicated column names (pandas
            would silently rename the second one to ``name.1``).
        FileNotFoundError: the file does not exist.
    """
    path = os.fspath(file_name)
    ext = _extension(path)
    if ext not in ('csv', 'txt', 'tsv', 'xls', 'xlsx'):
        raise ValueError(
            f"Unsupported file format: '{path}'. "
            "Supported extensions: csv, txt, tsv, xls, xlsx."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: '{path}'.")
    if os.path.getsize(path) == 0:
        raise ValueError(f"'{path}' is empty (0 bytes).")

    sep = _TEXT_SEP.get(ext)
    try:
        if sep is not None:
            df = _read_text(path, sep)
            if df.shape[1] == 0:
                # one column only: is another delimiter used?
                line = _first_line(path)
                counts = {s: line.count(s) for s in ('\t', ',', ';') if s != sep}
                other = max(counts, key=counts.get) if counts else None
                if other is not None and counts[other] > 0:
                    df = _read_text(path, other)
                    names = {'\t': 'tab', ',': 'comma', ';': 'semicolon'}
                    warnings.warn(
                        f"'{path}': the extension '.{ext}' means {names[sep]}-separated, but the "
                        f"file is {names[other]}-separated; it was read as {names[other]}-separated.",
                        UserWarning, stacklevel=2)
                    sep = other
        else:
            df = pd.read_excel(path, index_col=0)
    except pd.errors.EmptyDataError:
        raise ValueError(f"'{path}' contains no data (no header line).") from None

    # duplicated column names (pandas has already renamed them to 'name.1', ...)
    header = _header(path, ext, sep)[1:]
    stripped = [h.strip() for h in header]
    dup = sorted({h for h in stripped if h and stripped.count(h) > 1})
    if dup:
        raise ValueError(
            f"'{path}' has duplicated column names: {dup[:10]}. Every sample / time point needs "
            "a unique column name (pandas would silently rename the copy to '<name>.1').")

    # empty unnamed columns (trailing delimiters)
    unnamed = [c for c, h in zip(df.columns, stripped + [''] * len(df.columns))
               if h == '' and df[c].isna().all()]
    if unnamed:
        df = df.drop(columns=unnamed)
        warnings.warn(f"'{path}': dropped {len(unnamed)} empty column(s) without a header "
                      "(trailing delimiters?).", UserWarning, stacklevel=2)

    # whitespace around names
    def _strip(labels):
        if not _is_text(labels.dtype):
            return labels, 0
        values = list(labels)
        new_values = [v.strip() if isinstance(v, str) else v for v in values]
        changed = sum(1 for a, b in zip(values, new_values) if isinstance(a, str) and a != b)
        if not changed:
            return labels, 0
        return pd.Index(new_values, name=labels.name), changed
    cols, n_cols = _strip(df.columns)
    idx, n_idx = _strip(df.index)
    if n_cols or n_idx:
        df.columns, df.index = cols, idx
        warnings.warn(f"'{path}': removed leading/trailing spaces from {n_cols} column name(s) "
                      f"and {n_idx} row name(s).", UserWarning, stacklevel=2)

    # comment / trailer rows: '#...' or no name, and nothing in any column
    if len(df):
        empty_row = np.ones(len(df), dtype=bool)
        for c in range(df.shape[1]):
            col = df.iloc[:, c]
            blank = col.isna().to_numpy()
            if _is_text(col.dtype):
                blank = blank | col.astype(str).str.strip().eq("").to_numpy()
            empty_row &= blank
        labels = df.index
        is_comment = np.array([(isinstance(v, str) and v.startswith('#')) or pd.isna(v)
                               or (isinstance(v, str) and v == '') for v in labels])
        drop = empty_row & is_comment
        if drop.any():
            shown = [str(v) for v in labels[drop][:3]]
            df = df.loc[~drop]
            warnings.warn(f"'{path}': dropped {int(drop.sum())} comment/empty row(s) without data, "
                          f"e.g. {shown}.", UserWarning, stacklevel=2)
    return df


# ************* input checks (used by compute_corr / compute_fc) *************
def _check_numeric_table(df, what, path, feature):
    """The gene / metabolite table must have named rows and finite numeric values only."""
    if df.shape[0] == 0:
        raise ValueError(f"The {what} table ('{path}') has no rows.")
    if df.shape[1] == 0:
        raise ValueError(
            f"The {what} table ('{path}') has no data columns, only the name column. "
            "Check the delimiter of the file (csv = comma, tsv/txt = tab).")
    idx = df.index
    blank = [i + 1 for i, v in enumerate(idx)
             if (not isinstance(v, str) and pd.isna(v)) or (isinstance(v, str) and not v.strip())]
    if blank:
        raise ValueError(
            f"The {what} table ('{path}') has {len(blank)} row(s) without a name in the first "
            f"column (data rows {blank[:5]}, counting from 1 after the header).")
    if pd.api.types.is_float_dtype(idx.dtype):
        raise ValueError(
            f"The first column of the {what} table ('{path}') is used as the {feature} names, but "
            f"it contains numbers (e.g. {list(idx[:3])}): the table seems to have no name column. "
            f"Add a first column with the {feature} names.")
    bad_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    if bad_cols:
        cells = []
        for c in bad_cols:
            col = df[c]
            bad = (pd.to_numeric(col, errors='coerce').isna() & col.notna()).to_numpy()
            for pos in np.flatnonzero(bad)[:3]:
                cells.append(f"row '{idx[pos]}', column '{c}': {col.iloc[pos]!r}")
        raise ValueError(
            f"The {what} table ('{path}') has non-numeric values in {len(bad_cols)} column(s), "
            f"e.g. {'; '.join(cells[:5])}. All values must be numbers; leave missing values empty "
            "(or NA). Decimal commas ('1,5') and thousands separators are not supported.")
    values = df.to_numpy(dtype=float)
    inf = np.isinf(values)
    if inf.any():
        r, c = np.argwhere(inf)[0]
        raise ValueError(
            f"The {what} table ('{path}') has {int(inf.sum())} infinite value(s), e.g. row "
            f"'{idx[r]}', column '{df.columns[c]}'. Replace them by numbers or leave them empty.")


def _check_design(design, path):
    """Return the design with a 'group' column and str sample names; ValueError otherwise."""
    cols = list(design.columns)
    if 'group' not in cols:
        alt = [c for c in cols if str(c).strip().lower() == 'group']
        if len(alt) != 1:
            raise ValueError(
                f"The design table ('{path}') needs a column named 'group' (the time point of "
                f"each sample); its columns are {cols}. The first column must hold the sample names "
                "(the column names of the gene and metabolite tables).")
        warnings.warn(f"design table: using column {alt[0]!r} as 'group'.", UserWarning, stacklevel=4)
        design = design.rename(columns={alt[0]: 'group'})
    design = design.copy()
    design.index = design.index.map(str)
    if design.index.has_duplicates:
        dup = list(design.index[design.index.duplicated()].unique()[:10])
        raise ValueError(f"The design table ('{path}') lists the same sample more than once: {dup}.")
    missing = design.index[design['group'].isna()]
    if len(missing):
        raise ValueError(
            f"The design table ('{path}') has samples without a group: {list(missing[:10])}.")
    return design


def _check_samples_in_design(columns, design, what):
    """Every column of the data table must be a sample of the design table."""
    missing = [c for c in columns if c not in design.index]
    if not missing:
        return
    hints = []
    for c in missing[:3]:
        close = difflib.get_close_matches(c, list(design.index), n=1, cutoff=0.6)
        if close:
            hints.append(f"'{c}' -> '{close[0]}'?")
    raise ValueError(
        f"{len(missing)} column(s) of the {what} table are not samples of the design table: "
        f"{missing[:10]}" + (f" (similar design samples: {', '.join(hints)})" if hints else "")
        + f". Design samples: {list(design.index[:10])}{' ...' if len(design) > 10 else ''}. "
        "The first column of the design table must hold exactly the column names of the data tables.")


def _check_target(values, target, columns):
    v = np.asarray(values, dtype=float)
    bad = [c for c, x in zip(columns, v) if not np.isfinite(x)]
    if bad:
        raise ValueError(
            f"The target '{target}' has no value at time point(s) {bad}; every metric would be NaN. "
            "Fill in the missing value(s) or remove these time points.")
    if np.ptp(v) == 0:
        raise ValueError(
            f"The target '{target}' is constant over time ({v[0]!r} at every time point); no "
            "association with it can be computed.")



#get target compounds
def get_target(name, df):
    """
    find the target compound in the dataframe and return the row of the target

    Raises KeyError (with the target name in the message, and similar names if there are any)
    if the target is not in df.index, and ValueError if it occurs more than once.
    """
    if name in df.index:
        row = df.loc[name]
        if isinstance(row, pd.DataFrame):
            raise ValueError(
                f"Target '{name}' occurs {len(row)} times in the metabolite table; "
                "metabolite names must be unique (at least the target's).")
        return row.tolist()
    close = difflib.get_close_matches(str(name), [str(i) for i in df.index], n=3, cutoff=0.6)
    hint = f" Similar names: {close}." if close else ""
    raise KeyError(f"Target '{name}' is not in the index of the dataframe.{hint}")
