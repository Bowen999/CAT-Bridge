"""High-level pipelines tying the other modules together."""

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from .data_io import read_upload, get_target
from .preprocess import merge_dataframes
from .metrics import (
    compute_granger,
    compute_ccm,
    compute_cca,
    compute_dtw,
    compute_ccf,
    compute_spearman,
    compute_pearson,
    ts_clustering,
)
from .foldchange import find_noontide, df_for_fc, no_repeat_fc, fc_comp
from .scoring import add_annotation_score

__all__ = [
    'pipeline',
    'compute_corr',
]


def pipeline(gene_file, metabo_file, design_file, annotation_file, target, cluster_count, max_lag=1, aggregation_func=None):
    """
    This function processes gene expression data, performs computations, and returns results.

    Parameters:
    - gene_file (str): The filename of the gene count data.
    - metabo_file (str): The filename of the metabolome data.
    - design_file (str): The filename of the experimental design data (can be None).
    - annotation_file (str): The filename of the gene annotation data (can be None).
    - target (str): The target metabolite for the analysis.
    - cluster_count (int): The number of clusters for time series clustering.
    - max_lag (int): The maximum number of lags for Granger causality test (default is 1).
    - aggregation_func (function): The function to be used for data aggregation.

    Returns:
    - result (pd.DataFrame): A pandas DataFrame containing the processed results, with annotations and clustering information if provided.
    """
    # Read data
    gene = read_upload(gene_file)
    metabo = read_upload(metabo_file)

    if design_file is not None:
        # If there is a design file
        design = read_upload(design_file)
        # Process data
        processed_gene = aggregation_func(gene, design)
        processed_metabo = aggregation_func(metabo, design)
        # Get target data
        t = get_target(target, processed_metabo)
        # Compute Granger causality
        granger = compute_granger(processed_gene, t, max_lag)
        pearson = compute_pearson(processed_gene, t)
        # Prepare dataframe for fold change calculation
        df_for_fc(processed_metabo, target, gene, design)
        # Compute fold change
        fc = fc_comp()
        # Merge data
        data = merge_dataframes([granger, fc, pearson])

    else:
        # If there is no design file
        t = get_target(target, metabo)
        noontide = find_noontide(metabo, target)
        granger = compute_granger(gene, t, max_lag)
        # Compute fold change
        fc = no_repeat_fc(gene, noontide)
        data = merge_dataframes([granger, fc])

    if annotation_file is not None:
        annotation = read_upload(annotation_file)
        data = merge_dataframes([data, annotation])
        data = add_annotation_score(data)  # Add a new column for annotation score

    data['log2FoldChange'].fillna(0, inplace=True)
    data['Granger'].fillna(0, inplace=True)
    # add a new column ['Score'] to the dataframe
    # use minmax scaling Granger and log2FoldChange before Euclidean distance
    model = MinMaxScaler()
    data['Score'] = model.fit_transform(data['Granger'].values.reshape(-1,1)) + model.fit_transform(data['log2FoldChange'].values.reshape(-1,1))

    
    
    #min-max scaling
    data['Score'] = model.fit_transform(data['Score'].values.reshape(-1,1))
    result = data.sort_values(by='Score', ascending=False)
    # add a new column ['Rank'] to the dataframe
    result['Rank'] = np.arange(1, len(result)+1)
    result.set_index('Rank', inplace=True)

    return result


def compute_corr(
    gene_file, metabo_file, design_file, annotation_file,
    target, cluster_count, aggregation_func=None, 
    lag=1, E=3, tau=1, n_components=1
):
    """
    Computes correlation metrics between gene expression and metabolite data.
    
    This function reads in gene expression data, metabolite data, optional experimental design, 
    and annotation data from specified files. It then processes the data, computes various correlation 
    and similarity metrics, performs clustering, and optionally annotates the results, 
    returning a DataFrame containing the aggregated results.
    
    Parameters:
    - gene_file (str): Path to the file containing gene count data.
    - metabo_file (str): Path to the file containing metabolome data.
    - design_file (str): Optional; Path to the file containing experimental design data. Default is None.
    - annotation_file (str): Optional; Path to the file containing gene annotation data. Default is None.
    - target (str): The target metabolite for analysis.
    - cluster_count (int): Number of clusters for time series clustering.
    - aggregation_func (callable, optional): Function for data aggregation. Default is None.
    - lag (int, optional): Maximum number of lags for Granger causality test. Default is 1.
    - E (int, optional): The embedding dimension for Convergent Cross Mapping (CCM). Default is 3. Imply the complexity of the system.
    - tau (int, optional): The delay for Convergent Cross Mapping (CCM). Default is 1.
    - n_components (int, optional): Number of components for Canonical Correlation Analysis (CCA). Default is 1.
    
    Returns:
    - result (pd.DataFrame): A DataFrame containing the computed metrics, clustering results, 
                             and optionally annotations.
    
    Raises:
    - Various exceptions may be raised due to file reading, data processing, or computation errors.
    
    Notes:
    - The aggregation_func parameter should be a function that takes two arguments: a DataFrame 
      containing data and a DataFrame containing design information, and returns a processed DataFrame.
    - If the design_file is provided, the data will be aggregated based on the provided aggregation_func.
    - The function employs a variety of metrics such as Granger causality, Convergent Cross Mapping (CCM),
      Canonical Correlation Analysis (CCA), Dynamic Time Warping (DTW), Cross-Correlation Function (CCF),
      Spearman correlation, and Pearson correlation to analyze the relationships between gene expression 
      and metabolite data.
    - It performs time series clustering on the gene expression data.
    - If the annotation_file is provided, the function will merge the annotation data with the results, 
      and compute an annotation score.
    """
    # Read data
    gene = read_upload(gene_file)
    metabo = read_upload(metabo_file)
    
    if design_file is not None:
        design = read_upload(design_file)
        # Process data
        processed_gene = aggregation_func(gene, design)
        processed_metabo = aggregation_func(metabo, design)
        t = get_target(target, processed_metabo)
        fc_data = df_for_fc(processed_metabo, target, gene, design)
        fc = fc_comp(fc_data[0],fc_data[1])
        
    else:
        t = get_target(target, metabo)
        noontide = find_noontide(metabo, target)
        processed_gene = gene
        processed_metabo = metabo
        fc = no_repeat_fc(gene, noontide)
    
    # Compute corr
    granger_score = compute_granger(processed_gene, t, maxlag=lag)
    ccm_score = compute_ccm(processed_gene, t, E=E, tau=tau)
    cca_score = compute_cca(processed_gene, t, n_components=1)
    dtw_score = compute_dtw(processed_gene, t)
    ccf_score = compute_ccf(processed_gene, t, lag=1)
    spearman_score = compute_spearman(processed_gene, t)
    pearson_score = compute_pearson(processed_gene, t)
    clustering_result = ts_clustering(gene, cluster_count)


    data = merge_dataframes([granger_score, ccm_score, cca_score, dtw_score, ccf_score, spearman_score, pearson_score, fc, clustering_result])


    if annotation_file is not None:
        annotation = read_upload(annotation_file)
        data = merge_dataframes([data, annotation])
        data = add_annotation_score(data)  # Add a new column for annotation score

    return data
