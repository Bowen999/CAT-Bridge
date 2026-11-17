"""Scoring and ranking of candidate features."""

import pandas as pd
import numpy as np

from scipy.stats import linregress  # linear regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

__all__ = [
    'score',
    'keywords_scores',
    'add_annotation_score',
    'trend_strength',
    'top_important_features',
    'compute_modulus',
    'compute_score',
]


# ***************  compute score ******************
def score(df):
    """
    Compute a score based on specified columns of a dataframe, and rank the rows based on the score.
    """
    # Exclude the 'Name' column and compute row sums
    df['Score'] = df.drop(columns=['Name']).sum(axis=1)
    
    # Rank rows from high to low based on the sum, '1' being the highest rank
    df['Rank'] = df['Score'].rank(method='min', ascending=False)
    
    # Sort dataframe by rank
    df_sorted = df.sort_values('Rank')
    # df_sorted.set_index('Rank', inplace=True)
    # df_sorted = df_sorted[['Name', 'Score', 'Rank']]
    
    return df_sorted


# Annotation -> Score
keywords_scores = {'ase': 0.2, 'enzyme': 0.2, 'synthase': 0.2}
def add_annotation_score(df, keywords=keywords_scores):
    """
    This function adds a new column 'Description_Score' to the input dataframe. The values in this column are computed
    based on the 'Description' column and the keywords dictionary.
    """
    def compute_description_score(description):
        try:
            if pd.isna(description):  # Check if the value is NaN
                return 0.1
            words = description.split()  # Split the description into words
            max_score = None  # Initialize max score
            for word in words: 
                for keyword, score in keywords.items(): 
                    if word.endswith(keyword):  # If the word ends with a keyword
                        if max_score is None or score > max_score:  # If score is higher than current max score
                            max_score = score  # Update max score
            if max_score is not None:  # If a keyword was found
                return max_score  # Return max score
            return 0  # Return 0 if no keywords are found
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0  # Return 0 in case of an error

    df['Description_Score'] = df['Description'].apply(compute_description_score)
    return df


# ************* Upward trend strength *************
def trend_strength(df):
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


# ****************** Importance *******************
def top_important_features(df, n_top):
    """
    This function calculates the importance of features based on a PCA model. The importance is calculated 
    as the sum of the squares of the PCA loadings for each feature. The function then returns the top n 
    features based on their importance.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame where each row corresponds to a sample and each column corresponds 
                       to a feature (e.g., gene expression values for different genes). The index of the DataFrame
                       is assumed to represent sample identifiers and the columns of the DataFrame are assumed to 
                       represent feature names.
    n_top (int): The number of top features to return based on their calculated importance.

    Returns:
    pd.DataFrame: A DataFrame containing the top n features and their corresponding importances, sorted by importance.
    """

    try:
        # Transpose your data so that rows are time points and columns are genes
        df_transposed = df.transpose()

        # Initialize PCA object
        pca = PCA()

        # Fit the model
        pca.fit(df_transposed)

        # Get the loadings for each gene
        loadings = pca.components_

        # Calculate the importance of each gene as the sum of squares of its loadings
        importance = np.sum(loadings**2, axis=0)

        # Create a DataFrame of importances and gene names
        importance_df = pd.DataFrame({'Gene': df_transposed.columns, 'Importance': importance})

        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)

        # Return the top n genes
        return importance_df.head(n_top)

    except Exception as e:
        print("An error occurred while computing the top important features.")
        print(str(e))
        return None


def compute_modulus(a, b, c=None):
    if c is None:
        return (a ** 2 + b ** 2) ** 0.5
    else:
        return (a ** 2 + b ** 2 + c ** 2) ** 0.5


def compute_score(df, col):
    col = str(col)
    if col == 'Granger':
        df[col] = 1 - df[col]
    # Initializing MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fitting and transforming the specified columns
    df[['norm_col', 'norm_fc']] = scaler.fit_transform(df[[col, 'log2FoldChange']])
    
    # If 'Description_Score' column exists in the DataFrame
    if 'Description_Score' in df.columns:
        df['Score'] = df.apply(lambda row: compute_modulus(row['norm_col'], row['norm_fc'], row['Description_Score']), axis=1)
    else:
        df['Score'] = df.apply(lambda row: compute_modulus(row['norm_col'], row['norm_fc']), axis=1)
    
    df.sort_values(by='Score', ascending=False, inplace=True)
    
    
    df['Score'] = MinMaxScaler().fit_transform(df[['Score']])
    df = df[['Name', col, 'log2FoldChange', 'Cluster', 'Score']]
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1
    
    return df
