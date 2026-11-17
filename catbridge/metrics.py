"""Similarity / causality metrics between each feature (row) and a target series, plus clustering."""

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import ccf
from causal_ccm.causal_ccm import ccm
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import MinMaxScaler
from fastdtw import fastdtw
from scipy.stats import spearmanr
from scipy.stats import pearsonr

__all__ = [
    'compute_granger',
    'compute_reverse_granger',
    'granger_list',
    'compute_ccm',
    'compute_cca',
    'compute_dtw',
    'compute_ccf',
    'compute_spearman',
    'compute_pearson',
    'clustering',
    'ts_clustering',
]


# ************* 2.1 Granger **************************
def compute_granger(df, target, maxlag=1):
    """
    Compute the Granger causality score for each row in the dataframe.
    The X is each row in the dataframe and the Y is the target list.
    The returd value is the p-value of the F test for the maximum lag.
    
    Parameters:
        df (pandas DataFrame): The DataFrame to be aggregated.
        target (list): The target list.
    
    Returns:
        index_list (list): The list of row indices.
    
    """
    # Initialize lists to store results
    index_list = []
    p_value_list = []

    # Loop through each row in the dataframe
    for idx, row in df.iterrows():
        # Skip if the row or the target list has constant values
        if np.std(row.values) == 0 or np.std(target) == 0:
            continue

        # Combine the row and target into a DataFrame
        data = pd.concat([pd.Series(target), pd.Series(row.values)], axis=1)

        # Perform the Granger causality test
        try:
            result = grangercausalitytests(data, maxlag=maxlag, verbose=False)

            # Extract the p-value of the F test for the maximum lag
            p_value = result[maxlag][0]['ssr_ftest'][1]
            # p_value = 1-p_value

            # Append results to the lists
            index_list.append(idx)
            p_value_list.append(p_value)
        except Exception as e:
            #add NA if there is an error
            index_list.append(idx)
            p_value_list.append(np.nan)
            
            # print(f"Error at index {idx}: {str(e)}")

    # Create a new dataframe to store results
    results_df = pd.DataFrame({
        'Name': index_list,
        'Granger': p_value_list
    })

    return results_df


def compute_reverse_granger(df, target, maxlag=1):
    """
    X is target and Y is each row in the dataframe (df).
    """
    
     # Initialize lists to store results
    index_list = []
    p_value_list = []

    # Loop through each row in the dataframe
    for idx, row in df.iterrows():
        # Skip if the row or the target list has constant values
        if np.std(row.values) == 0 or np.std(target) == 0:
            continue

        # Combine the row and target into a DataFrame
        data = pd.concat([pd.Series(row.values), pd.Series(target)], axis=1)

        # Perform the Granger causality test
        try:
            result = grangercausalitytests(data, maxlag=maxlag, verbose=False)

            # Extract the p-value of the F test for the maximum lag
            p_value = result[maxlag][0]['ssr_ftest'][1]
            p_value = 1-p_value

            # Append results to the lists
            index_list.append(idx)
            p_value_list.append(p_value)
        except Exception as e:
            #add NA if there is an error
            index_list.append(idx)
            p_value_list.append(np.nan)
            
            # print(f"Error at index {idx}: {str(e)}")

    # Create a new dataframe to store results
    results_df = pd.DataFrame({
        'Name': index_list,
        'Granger': p_value_list
    })

    return results_df


def granger_list(A, B, maxlag):
    """
    Compute granger test P value for listA (X), and listB (y) 
    """
    
    # Combine the two lists into a 2D array
    data = np.column_stack((A, B))
    
    # Perform the Granger causality test and return the result
    result = grangercausalitytests(data, maxlag, verbose=False)
    #result = grangercausalitytests(data, maxlag)

    # In this function, we'll return just the p-value for each lag.
    # You may want to modify this to return other information.
    p_values = {lag: result[lag][0]['ssr_ftest'][1] for lag in result.keys()}
    
    return p_values


# ****************** 2.2 CCM (Convergent cross mapping)*******************************
def compute_ccm(df, target, E=3, tau=1):
    # Placeholder for result list
    results = [] 

    for name, data in df.iterrows():
        # Get data from the row (as the name is now the index)
        data_values = data.values
        
        if np.any(np.isnan(data_values)) or np.any(np.isnan(target)):
            print("data_values:", data_values)
            print("target:", target)
            continue  # Skip the current iteration if NaN is found


        # Calculate ccm
        ccm1 = ccm(data_values, target, tau, E)
        
        # Get causality value
        causality_val = ccm1.causality()[0]
        
        # Append to results list
        results.append([name, causality_val])

    # Convert results list to DataFrame
    result_df = pd.DataFrame(results, columns=['Name', 'CCM'])
    
    # Sort by Causality in descending order and reset the index
    result_df = result_df.sort_values(by='CCM', ascending=False).reset_index(drop=True)
    
    return result_df


# ******************** 2.3 CCA (Canonical correlation) ********************
def compute_cca(df, target, n_components=1):
    # Ensure the target data is 2D (samples x features)
    metabolite_concentration = np.array(target).reshape(-1, 1)
    results_dict = {"Name": [], "CCA": []}  # Initialize a dictionary to store the results

    for index, row in df.iterrows():
        # Reshape the gene_expression data to be 2D (samples x features)
        gene_expression = np.array(row).reshape(-1, 1)

        # It's essential to have more than one sample to perform CCA,
        # so if there's only one sample, append NaN and continue
        if gene_expression.shape[0] <= 1 or metabolite_concentration.shape[0] <= 1:
            results_dict["Name"].append(index)
            results_dict["CCA"].append(np.nan)
            continue

        try:
            # Instantiate CCA
            cca = CCA(n_components=n_components)
            cca.fit(gene_expression, metabolite_concentration)

            # Transform the data based on the canonical correlation vectors
            gene_c, metabolite_c = cca.transform(gene_expression, metabolite_concentration)

            # Compute the correlation between the transformed data
            correlation = np.corrcoef(gene_c.T, metabolite_c.T)[0, 1]
            results_dict["Name"].append(index)
            results_dict["CCA"].append(correlation)

        except ValueError as e:
            results_dict["Name"].append(index)
            results_dict["CCA"].append(np.nan)  # Append NaN for rows causing errors

    return pd.DataFrame(results_dict)  # Convert the results dictionary to a DataFrame and return it


# ******************* 2.4 DTW (Dynamic Time Warping) *************************
def compute_dtw(df, target):
    results_dict = {"Name": [], "DTW": []}  # Initialize a dictionary to store the results
    
    # Prepare the MinMax scaler
    scaler = MinMaxScaler()
    
    for index, row in df.iterrows():
        # Convert the row data and target to numpy arrays
        array1, array2 = np.array(row).reshape(-1, 1), np.array(target).reshape(-1, 1)
        
        # Normalize the arrays
        array1_norm = scaler.fit_transform(array1)
        array2_norm = scaler.fit_transform(array2)
        
        # Compute the DTW distance using the fastdtw function
        distance, path = fastdtw(array1_norm, array2_norm)
        
        # Append the index and DTW distance to the results dictionary
        results_dict["Name"].append(index)
        results_dict["DTW"].append(distance)
    
    return pd.DataFrame(results_dict)  # Convert the results dictionary to a DataFrame and return it


# ************************* 2.5 CCF *****************************
def compute_ccf(df, target, lag=1):
    results_dict = {"Name": [], "CCF": []}  # Initialize a dictionary to store the results

    for index, row in df.iterrows():
        gene_expression = np.array(row)

        # Ensure gene_expression and target have equal lengths, or handle it accordingly
        # One simple method could be truncating the longer array, but other methods might be more suitable depending on the context
        min_length = min(len(gene_expression), len(target))
        gene_expression = gene_expression[:min_length]
        truncated_target = np.array(target)[:min_length]

        # Compute the cross-correlation function
        cross_correlation = ccf(gene_expression, truncated_target)

        # Extract the cross-correlation at the specified lag
        # Handling the case where the specified lag is out of bounds by appending NaN
        if abs(lag) >= len(cross_correlation):
            results_dict["Name"].append(index)
            results_dict["CCF"].append(np.nan)
        else:
            lag_value = cross_correlation[lag] if lag >= 0 else cross_correlation[lag - 1]
            results_dict["Name"].append(index)
            results_dict["CCF"].append(lag_value)

    return pd.DataFrame(results_dict)  # Convert the results dictionary to a DataFrame and return it


# ***************** 2.6 Spearman *********************
def compute_spearman(df, target):
    # Initialize lists to store results
    index_list = []
    corr_list = []
    p_value_list = []

    # Loop through each row in the dataframe
    for idx, row in df.iterrows():
        # Compute Spearman correlation and p-value
        corr, p_value = spearmanr(row, target)

        # Append results to the lists
        index_list.append(idx)
        corr_list.append(corr)
        p_value_list.append(p_value)

    # Create a new dataframe to store results
    results_df = pd.DataFrame({
        'Name': index_list,
        'Spearman': corr_list
        #'P-value': p_value_list
    })

    return results_df


# ***************** 2.7 Pearson *********************
def compute_pearson(df, target):
    # Initialize lists to store results
    index_list = []
    corr_list = []
    p_value_list = []

    # Loop through each row in the dataframe
    for idx, row in df.iterrows():
        # Compute Pearson correlation and p-value
        corr, p_value = pearsonr(row, target)

        # Append results to the lists
        index_list.append(idx)
        corr_list.append(corr)
        p_value_list.append(p_value)

    # Create a new dataframe to store results
    results_df = pd.DataFrame({
        'Name': index_list,
        'Pearson': corr_list
        #'P-value': p_value_list
    })

    return results_df


# ************** fuzz c mean clustering *********************
def clustering(df, n_clusters):
    import skfuzzy as fuzz  # lazy import (heavy optional dependency)

    # Prepare the data
    data = df.values.T  # The cmeans function expects data to be of shape (n_features, n_samples)
    
    # Perform fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, n_clusters, 2, error=0.005, maxiter=1000)
    
    # Get the cluster assignment for each sample
    # The argmax function is used to assign each sample to the cluster for which it has the highest membership value
    cluster_assignment = u.argmax(axis=0)
    
    # Create a DataFrame to hold the results
    results_df = pd.DataFrame({'Name': df.index, 'Cluster': cluster_assignment})
    
    return results_df


# ts Clustering
def ts_clustering(df, n_clusters):
    """
    Cluster the time series data in the input dataframe using the TS-KMeans algorithm.
    
    Parameters:
        df (pandas.DataFrame): The input dataframe.
        n_clusters (int): The number of clusters to create.
    """
    from tslearn.clustering import TimeSeriesKMeans  # lazy import (heavy optional dependency)
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    # Convert DataFrame to NumPy array for compatibility with tslearn
    data = df.values

    # Rescale the time series data so that their mean is 0 and their standard deviation is 1
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    data_scaled = scaler.fit_transform(data)

    # Create the KMeans model
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", max_iter=5, random_state=0)

    # Fit the model to the data
    km.fit(data_scaled)

    # Get the cluster labels for each time series
    labels = km.labels_

    # Add the labels as a new column in the original DataFrame
    df['Cluster'] = labels
    df = df['Cluster']

    return df
