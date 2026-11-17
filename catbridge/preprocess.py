"""Pre-processing: normalisation, scaling, replicate aggregation, merging, trend strength, PCA reduction."""

import pandas as pd
import numpy as np

from scipy.stats import linregress  # linear regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

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
    Scale the data using the specified method.
    """
    if method == 'min-max':
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
    elif method == 'pareto':
        df = (df - df.mean()) / np.sqrt(df.std())


def scale_df(df):
    """Apply a log10 transformation and scale all values in a dataframe to the range 0-1."""
    
    # Apply a log10 transformation, add a small constant to avoid log(0)
    df = np.log10(df + 1e-10)
    
    # Check for NaNs and fill or remove them before scaling
    if df.isnull().values.any():
        df = df.fillna(method='ffill')  # you could also use 'bfill' or df.dropna()
        
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
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
    base_cols = sorted(set(base_cols), key=base_cols.index)

    # Create sub DataFrames and store their means
    mean_dfs = {}
    sub_dfs = {}
    for suffix in np.arange(1, num_suffixes + 1).astype(str):
        sub_df = df.filter(regex=f'_{suffix}$')
        sub_dfs[suffix] = sub_df
        mean_dfs[suffix] = sub_df.mean(axis=1)

    # Create a DataFrame to store which sub DataFrame has the highest mean for each row
    max_mean_df = pd.DataFrame(mean_dfs).idxmax(axis=1)

    # Create a new DataFrame, preserving the original index
    new_df = pd.DataFrame(index=df.index)

    # For each base column name
    for base_col in base_cols:
        for idx in new_df.index:
            # Determine which sub DataFrame to pull from for this row
            sub_df_idx = max_mean_df.loc[idx]

            # Get value from the appropriate sub DataFrame
            new_df.loc[idx, base_col] = sub_dfs[sub_df_idx].loc[idx, f"{base_col}_{sub_df_idx}"]


    new_df.columns = [str(col) + '_1' for col in new_df]
    mapping_dict = design['group'].to_dict()
    new_df.columns = new_df.columns.map(lambda x: mapping_dict[x] if x in mapping_dict else x)
    
    return new_df


# **************************** Mean **********************************
def repeat_aggregation_mean(df: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate biological replicates by taking the mean value run for each row (gene/compound).
    
    Parameters:
        df (pandas DataFrame): The DataFrame to be aggregated.
        design (pandas DataFrame): The experimental design DataFrame.
        
    Returns:
        new_df (pandas DataFrame): The aggregated DataFrame.
    """
    # Generate new column names from design DataFrame
    new_column_names = {i: design.loc[i]['group'] for i in df.columns}

    # Rename columns and average by new column names
    df_copy = df.rename(columns=new_column_names)
    df_copy = df_copy.T.groupby(level=0).mean().T
    
    return df_copy


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
