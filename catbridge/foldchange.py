"""Fold-change computation: noontide detection, no-replicate FC, DESeq2-based FC."""

import math

import pandas as pd
import numpy as np

from .data_io import read_upload

__all__ = [
    'find_noontide',
    'df_for_fc',
    'no_repeat_fc',
    'fc_comp',
    'compute_fc',
]


# ********* Noontide ************
def find_noontide(df, row_name):
    """
    Find the column with the highest value in the row, and return the column, and the column after it (name).
    If the column with the highest value is the last column, then return the column with the second highest value, and the column after it (name).
    
    Parameters:
        df (pandas DataFrame): The DataFrame has been aggregated.
        row_name (str): The name of the row to be detected.
        
    Returns:
        column_n (str): The name of the column with the highest value.
    """
    # Get the row with the specified name
    row = df.loc[row_name].copy()

    # Convert to numeric and handle errors by replacing non-numeric values with NaN
    row = pd.to_numeric(row, errors='coerce')

    # Handle rows that are all NaN
    if row.isna().all():
        raise Exception(f"The row '{row_name}' contains only non-numeric or missing values.")

    # Identify the column with the highest value
    column_n = row.idxmax()
    col_idx = list(df.columns).index(column_n)

    # If column with max value is last column, then find the second highest column
    if col_idx == len(df.columns) - 1:
        row[column_n] = np.nan # set value in column_n to NaN
        column_n = row.idxmax() # get column with max value now

    # Find column after column_n
    col_idx = list(df.columns).index(column_n)
    if col_idx < len(df.columns) - 1: # Make sure it's not the last column
        column_n_plus_1 = df.columns[col_idx + 1]
    else:
        raise Exception("There is no column after the column with the maximum value.")

    # Keep only column n and column n+1
    df_filtered = df[[column_n, column_n_plus_1]]

    return df_filtered.columns


def df_for_fc(df1, target, df2, design):
    """
    Generate the design matrix and matrix for computing the FC score.
    
    Parameters:
        df1 (pandas DataFrame): The DataFrame has been aggregated (processed_metabo).
        target (str): The name of the row to be detected (Capsaicin).
        df2 (pandas DataFrame): The DataFrame for fc computing (gene).
        design (pandas DataFrame): study design, the sample and group information.
    
    Returns:
        volcano_df (pandas DataFrame): Modified matrix for computing FC score.
        volcano_design (pandas DataFrame): Modified design matrix.
    """
    noontide = find_noontide(df1, target)
    volcano_design = design[design['group'].isin(noontide)]
    volcano_df = df2[volcano_design.index]

    return volcano_df, volcano_design


def no_repeat_fc(df, noontide):
    """
    Compute the FC score for each metabolite. This function is used for Study Design that has no biological repeats.
    
    Parameters:
        df (pandas DataFrame): The DataFrame for fc computing (gene).
        noontide (list): The list of the column names of the noontide samples. (can be obtained from find_noontide function)
    
    Returns:
        new_df (pandas DataFrame): A new DataFrame with the FC score.
    """
    
    # Create a new DataFrame as a copy of the original
    new_df = df.copy()
    
    # Compute log2FoldChange
    new_df['log2FoldChange'] = new_df[noontide[0]] / new_df[noontide[1]]
    new_df['log2FoldChange'] = new_df['log2FoldChange'].apply(lambda x: math.log2(x if x > 0 else np.finfo(float).eps))
    
    # Replace infinities with 0
    new_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Replace NaNs with 0
    new_df['log2FoldChange'].fillna(0, inplace=True)
    
    # Make the range of log2FoldChange from 0-1
    # scaler = MinMaxScaler()
    # new_df['log2FoldChange'] = scaler.fit_transform(new_df[['log2FoldChange']])
    
    # Keep only the 'log2FoldChange' column in the new DataFrame
    new_df = new_df[['log2FoldChange']]
    
    return new_df


def fc_comp(counts_df, metadata):
    """
    Perform a DESeq2 analysis given counts data and metadata.
    
    Parameters:
    - counts_df (DataFrame): The dataframe containing gene counts.
    - metadata (DataFrame): The dataframe containing sample metadata.
    
    Returns:
    - DataFrame: The results dataframe from the DESeq2 analysis.
    """
    from pydeseq2.dds import DeseqDataSet  # lazy import (heavy optional dependency)
    from pydeseq2.ds import DeseqStats

    
    # Filter genes based on counts
    counts_df = counts_df.T
    genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
    counts_df = counts_df[genes_to_keep]
    counts_df = counts_df.round().astype(int)

    # Adjust metadata
    metadata['condition'] = metadata['group']

    # Create and run DESeq2 analysis
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors="condition",
        refit_cooks=True,
        n_cpus=8,
    )
    dds.deseq2()

    # Get and summarize statistics
    stat_res = DeseqStats(dds, n_cpus=8)
    stat_res.summary(lfc_null=0.1, alt_hypothesis="greaterAbs")
    stat_res.results_df['log2FoldChange'] = -1 * stat_res.results_df['log2FoldChange']
    
    return stat_res.results_df


def compute_fc(
    gene_file, metabo_file, design_file=None, aggregation_func=None, target=None
):
    """
    ... [rest of the docstring and function]
    """
    # Read data
    gene = read_upload(gene_file)
    metabo = read_upload(metabo_file)
    
    if design_file is not None:
        design = read_upload(design_file)
        # Process data
        processed_gene = aggregation_func(gene, design)
        processed_metabo = aggregation_func(metabo, design)
        fc_data = df_for_fc(processed_metabo, target, gene, design)  # Added target
        fc = fc_comp(fc_data[0], fc_data[1])
        
    else:
        noontide = find_noontide(metabo, target)  # Passed target as row_name
        fc = no_repeat_fc(gene, noontide)
    
    return fc
