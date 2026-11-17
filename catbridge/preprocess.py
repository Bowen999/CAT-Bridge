"""Pre-processing: normalisation, scaling, replicate aggregation, merging, trend strength, PCA reduction."""

import warnings

import pandas as pd
import numpy as np

from ._lazy import lazy_callable

# imported on first use (keeps `import catbridge` fast)
linregress = lazy_callable('scipy.stats', 'linregress')  # linear regression
MinMaxScaler = lazy_callable('sklearn.preprocessing', 'MinMaxScaler')
StandardScaler = lazy_callable('sklearn.preprocessing', 'StandardScaler')
IncrementalPCA = lazy_callable('sklearn.decomposition', 'IncrementalPCA')

__all__ = [
    'normalize',
    'scaling',
    'scale_df',
    'scale_column',
    'repeat_aggregation_max',
    'repeat_aggregation_mean',
    'merge_dataframes',
    'calculate_trend_strength',
    'merge_and_reduce',
]


#def normalize function, that allow log2, log10, and z-score
def normalize(df, method):
    """
    Normalize the data using the specified method.
    """
    if method == 'log2':
        df = np.log2(df + 1)
    elif method == 'log10':
        df = np.log10(df + 1)
    elif method == 'z-score':
        df = (df - df.mean()) / df.std()
    return df


#  ************* Scaling ***************
def scaling(df, method):
    """
    Scale the data using the specified method and return the scaled data.

    'min-max' scales each column to 0-1 (index and columns are preserved when
    ``df`` is a DataFrame). 'pareto' is (x - mean) / sqrt(std) per column.
    Any other method returns ``df`` unchanged (same as ``normalize``).
    The input is never modified.
    """
    if method == 'min-max':
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
        if isinstance(df, pd.DataFrame):
            df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
        else:
            df = scaled
    elif method == 'pareto':
        df = (df - df.mean()) / np.sqrt(df.std())
    return df


def scale_df(df):
    """Apply a log10 transformation and scale all values in a dataframe to the range 0-1.

    Rows = genes, columns = time points. Missing values are forward-filled along the columns
    (previous time point of the same gene); a missing first time point stays NaN, which
    MinMaxScaler keeps as NaN.
    """
    
    # Apply a log10 transformation, add a small constant to avoid log(0)
    df = np.log10(df + 1e-10)
    
    # Check for NaNs and fill or remove them before scaling
    if df.isnull().values.any():
        # fill along the time axis (columns): a missing value takes the previous time point of the
        # SAME gene. (axis=0 would copy the value of the previous gene.)
        df = df.ffill(axis=1)
        
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    
    return df_scaled


def scale_column(df, column_name):
    """
    Scale the values of a column using the MinMaxScaler.
    """
    # Create a scaler object
    scaler = MinMaxScaler()

    # Create a copy of the DataFrame to avoid modifying the original one
    df_scaled = df.copy()

    # Reshape the data to fit the scaler
    data = df[column_name].values.reshape(-1, 1)

    # Fit and transform the data
    df_scaled[column_name] = scaler.fit_transform(data)

    return df_scaled


# ************* Biological Replicates ***************
# ************** Max *******************************
def repeat_aggregation_max(df, design):
    """
    Aggregate biological replicates by taking the maximum value run for each row (gene/compound).

    For every row, the replicate (suffix ``_1``, ``_2``, ...) with the highest mean over all
    time points is selected, and that replicate's values are used for every time point.
    Output columns keep the order of first appearance in ``df`` (time order of the data file;
    the row order of ``design`` is irrelevant) and are named by the group of ``<base>_1`` in
    ``design``.

    Parameters:
        df (pandas DataFrame): The DataFrame to be aggregated.
        design (pandas DataFrame): The experimental design DataFrame.
    
    Returns:
        new_df (pandas DataFrame): The aggregated DataFrame.
    """
    # Calculate the number of suffixes
    num_suffixes = int(len(design.index) / len(design['group'].unique()))

    # Extract base column names and maintain their order
    base_cols = [col.rsplit('_', 1)[0] for col in df.columns if '_' in col]
    base_cols = list(dict.fromkeys(base_cols))

    # Mean of each replicate over all time points: rows x suffixes
    suffixes = [str(i) for i in range(1, num_suffixes + 1)]
    sub_dfs = {}
    mean_dfs = {}
    for suffix in suffixes:
        sub_df = df.filter(regex=f'_{suffix}$')
        sub_dfs[suffix] = sub_df
        mean_dfs[suffix] = sub_df.mean(axis=1)
    mean_mat = pd.DataFrame(mean_dfs, index=df.index).to_numpy(dtype=float)

    # Index of the replicate with the highest mean for each row (first one wins ties;
    # rows without any valid mean fall back to the first suffix -> values stay NaN/as is)
    best = np.argmax(np.where(np.isnan(mean_mat), -np.inf, mean_mat), axis=1)

    # Build the output column by column (vectorised over rows)
    new_cols = {}
    missing = []
    for base_col in base_cols:
        out = np.full(len(df.index), np.nan)
        for k, suffix in enumerate(suffixes):
            col = f"{base_col}_{suffix}"
            if col not in df.columns:
                if (best == k).any():
                    missing.append(col)
                continue
            mask = best == k
            out[mask] = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)[mask]
        new_cols[base_col] = out
    if missing:
        warnings.warn(
            f"repeat_aggregation_max: columns {missing} are missing from df; "
            "the corresponding values are set to NaN.", UserWarning, stacklevel=2)

    new_df = pd.DataFrame(new_cols, index=df.index)
    mapping_dict = design['group'].to_dict()
    new_df.columns = [mapping_dict.get(f"{c}_1", f"{c}_1") for c in new_df.columns]

    return new_df


# **************************** Mean **********************************
def repeat_aggregation_mean(df: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate biological replicates by taking the mean value run for each row (gene/compound).

    The output columns are the groups in order of first appearance in the columns of ``df``.
    The time order is that of the data file's columns -- the row order of ``design`` (which may
    be sorted alphabetically by sample name) does not matter, and the groups are NOT sorted
    alphabetically. Groups of ``design`` that have no column in ``df`` are dropped.
    ``repeat_aggregation_max`` gives the same column order.

    Parameters:
        df (pandas DataFrame): The DataFrame to be aggregated.
        design (pandas DataFrame): The experimental design DataFrame (index = sample names,
            column 'group').
        
    Returns:
        new_df (pandas DataFrame): The aggregated DataFrame.

    Raises:
        KeyError: if a column of ``df`` is not a sample in ``design.index``.
    """
    missing = [c for c in df.columns if c not in design.index]
    if missing:
        raise KeyError(f"Columns of df not found in design.index: {missing}")

    groups = design.loc[list(df.columns), 'group']
    group_values = groups.to_numpy()

    # Group order = order of first appearance in the DATA columns (time order of the data file)
    order = list(pd.unique(group_values))

    result = df.T.groupby(group_values, sort=False).mean().T
    result = result.loc[:, order]
    result.columns.name = None
    return result


# ************* Merge ***************
def merge_dataframes(dataframes):
    """
    Merge multiple dataframes based on the 'Name' column.

    Parameters:
        dataframes (list): A list of pandas DataFrames to be merged.

    Returns:
        merged_dataframe (pandas DataFrame): The merged DataFrame.
    """
    # Check if the input list is not empty
    if not dataframes:
        raise ValueError("The input list of dataframes is empty.")

    # Merge the dataframes one by one
    merged_dataframe = dataframes[0]
    for df in dataframes[1:]:
        merged_dataframe = merged_dataframe.merge(df, on='Name', how='outer')

    return merged_dataframe 


# ************ Find the increasing trend ***************
def calculate_trend_strength(df):
    """
    Function to calculate and sort the strength of the upward trend in each row of a DataFrame.

    This function calculates the trend strength (r-value) for each row in the DataFrame 
    using linear regression. It then adds a new column to the DataFrame with these r-values. 
    Finally, it sorts the DataFrame by these r-values in descending order and returns the 
    sorted DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be processed. 

    Returns:
    df_sorted_by_trend_strength (pandas.DataFrame): The input DataFrame, but with an added 
    column for trend strength and sorted by this column in descending order.
    """

    # Function to calculate the trend strength (r-value)
    def upward_trend_strength(row):
        _, _, rvalue, _, _ = linregress(x=range(len(row)), y=row)
        return rvalue

    # Apply function to each row to get trend strengths
    trend_strength = df.apply(upward_trend_strength, axis=1)

    # Add a new column to df with trend strengths
    df_with_trend_strength = df.assign(trend_strength=trend_strength)

    # Sort df by trend strength (r-value) in descending order
    df_sorted_by_trend_strength = df_with_trend_strength.sort_values(by='trend_strength', ascending=False)

    # Return sorted dataframe
    return df_sorted_by_trend_strength


# *************************** PCA (two dataframes) ***************************
def merge_and_reduce(df1, df2, n_components):
    """
    This function takes two dataframes with the same columns (sample names), performs Incremental PCA to reduce 
    the dimension of their index (IDs), and then merges them together. It ensures that both dataframes 
    have the same contribution to the final result by normalizing and reducing the dimension of each dataframe 
    separately before merging.

    Parameters:
    df1 (pandas.DataFrame): The first dataframe.
    df2 (pandas.DataFrame): The second dataframe.
    n_components (int): The number of dimensions for the output dataframe.

    Returns:
    pandas.DataFrame: The merged dataframe with reduced dimensions.
    """
    
    # Normalize each dataframe by their L2 norm
    scaler = StandardScaler()
    df1_norm = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns, index=df1.index)
    df2_norm = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns, index=df2.index)
    
    # Function to perform Incremental PCA and reduce dimensions
    def ipca_reduce(df, n_components):
        # Transpose the dataframe to reduce dimension of rows
        df_t = df.T

        # Apply Incremental PCA
        ipca = IncrementalPCA(n_components=n_components, batch_size=10)
        df_reduced = ipca.fit_transform(df_t)

        # Create a dataframe with the reduced dimension data
        df_reduced = pd.DataFrame(df_reduced, index=df_t.index, columns=[f'feature_{i}' for i in range(n_components)])

        # Transpose back to original form
        df_reduced_t = df_reduced.T

        return df_reduced_t
    
    # Reduce dimensions of each dataframe
    df1_reduced = ipca_reduce(df1_norm, n_components)
    df2_reduced = ipca_reduce(df2_norm, n_components)

    # Concatenate the two reduced dataframes
    df_merged = pd.concat([df1_reduced, df2_reduced])

    return df_merged
