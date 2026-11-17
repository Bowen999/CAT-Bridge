"""Plotting helpers (matplotlib / seaborn / networkx)."""

import textwrap

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import merge_dataframes
from .scoring import METRIC_DIRECTION

__all__ = [
    'plot_line',
    'plot_heatmap',
    'plot_pca',
    'plot_top_features',
    'plot_volcano',
    'plot_ts_clusters',
    'plot_ts_clusters_average',
    'plot_network',
    'plot_hexbin',
    'plot_line_heatmap',
    'save_table_as_svg',
    'plot_result',
]


# Line Plot
def plot_line(df, target, save_path=None):
    """
    Plots a line graph of the target row from the dataframe.
    Args:
    df (pd.DataFrame): DataFrame to plot.
    target (str): Target row to plot.
    """
    try:
        # Access the target row
        target_row = df.loc[target]

        # Create the line plot
        plt.figure(figsize=(12,3))
        plt.plot(target_row.index, target_row.values)
        
        # Add labels and title
        plt.title(f'{target}')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


def plot_heatmap(dataframe, palette='vlag', figsize=(6, 8), row_threshold=50, n_clusters=None, save_path=None):
    """
    Plot the heatmap of the dataframe. If the number of rows is greater than row_threshold, row labels are not shown.
    Args:
    dataframe (pd.DataFrame): DataFrame to plot.
    palette (str, optional): Palette to use for the plot. Defaults to 'vlag'.
    figsize (tuple, optional): Size of the plot. Defaults to (6, 8).
    row_threshold (int, optional): Threshold for number of rows to determine whether to show row labels. Defaults to 50.
    n_clusters (int, optional): Number of clusters to use if rows are over row_threshold. Defaults to None.
    save_path (str, optional): If provided, save the plot to this path. Otherwise, the plot is shown using plt.show(). Defaults to None.
    """
    try:
        # Round the dataframe to the nearest thousandth
        dataframe = dataframe.round(2)

        # If the number of rows is greater than row_threshold and n_clusters is not None, perform clustering
        if n_clusters is not None and dataframe.shape[0] > row_threshold:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dataframe)
            dataframe['cluster'] = kmeans.labels_
            dataframe = dataframe.groupby('cluster').mean()

        # Remove the rows that have all 0 values
        dataframe = dataframe.loc[~(dataframe==0).all(axis=1)]

        # Generate the heatmap
        row_labels = dataframe.shape[0] <= row_threshold
        if row_labels:
            plt.figure(figsize=figsize)
            heatmap = sns.heatmap(dataframe, cmap=palette, yticklabels=row_labels, xticklabels=True)
            plt.ylabel('')  # Remove y-axis label
        else:
            g = sns.clustermap(dataframe, cmap=palette, yticklabels=row_labels, xticklabels=True, figsize=figsize, col_cluster=False)
            g.ax_heatmap.set_ylabel('')  # Remove y-axis label
            plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), text='')  # Remove y-axis labels

        # Set x axis label
        plt.xlabel(dataframe.columns.name)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


# ********************* PCA ****************************
def plot_pca(gene, design, n_clusters, save_path=None):
    """
    Perform PCA on the gene expression data, and plot the result.
    
    Parameters:
        gene: the gene expression dataframe
        design: the design dataframe
        n_clusters: the number of clusters to use in K-means clustering
    """
    from adjustText import adjust_text  # lazy import (heavy optional dependency)

    try:
        # Transpose your DataFrame, as PCA works on the features (columns), not on the samples (rows)
        gene_transposed = gene.T

        # Perform PCA on your data
        pca = PCA(n_components=2)  # here we ask for the first two principal components
        pca_result = pca.fit_transform(gene_transposed)

        # Convert the PCA result to a DataFrame
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

        # Add a 'sample' column
        pca_df['sample'] = gene_transposed.index

        # Get group information from the design dataframe
        pca_df = pca_df.merge(design[['group']], left_on='sample', right_index=True)

        fig, ax = plt.subplots(figsize=(6, 8))

        if n_clusters > 0:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_df[['PC1', 'PC2']])
            pca_df['Cluster'] = kmeans.labels_

            # Plot the points on the scatterplot
            scatter = sns.scatterplot(x="PC1", y="PC2", hue="group", data=pca_df, palette="Paired", s=100, alpha=0.7, ax=ax)

            # For each cluster, add a circle at the mean coordinates with radius proportional to the standard deviation
            for cluster in set(kmeans.labels_):
                cluster_points = pca_df[pca_df['Cluster'] == cluster][['PC1', 'PC2']]
                # Calculate mean and standard deviation for the cluster
                cluster_mean = cluster_points.mean().values
                cluster_std = cluster_points.std().values
                # Add a circle at the mean coordinates with radius=stddev
                circle = Circle(cluster_mean, np.linalg.norm(cluster_std), alpha=0.1)
                ax.add_artist(circle)
        else:
            # Plot the points on the scatterplot without clustering
            scatter = sns.scatterplot(x="PC1", y="PC2", hue="group", data=pca_df, palette="Paired", s=100, alpha=0.7, ax=ax)

        # Check if the legend exists before trying to remove it
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        # Annotate points on the graph with the sample names
        texts = []
        for i, sample in enumerate(pca_df['sample']):
            texts.append(plt.text(pca_df.iloc[i].PC1, pca_df.iloc[i].PC2, sample, color='gray'))
            
        # Adjust the labels to avoid overlap
        adjust_text(texts)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        # plt.title('PCA', fontweight='bold')

        # Hide X and Y values
        plt.xticks([])
        plt.yticks([])
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")


# ****************** Top Important Features ******************
def plot_top_features(df, color='crest', save_path=None):
    """
    This function creates a scatter plot of the importances of the top n features, as returned by the 
    top_important_features function. The color palette is applied based on bins of 'Importance'.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame as returned by the top_important_features function.
    color (str): The color map to use for the scatter plot.

    Returns:
    None
    """

    try:
        # Bin the 'Importance' into categories
        df['Importance_Category'] = pd.qcut(df['Importance'], q=4)

        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = sns.scatterplot(data=df, y='Gene', x='Importance', hue='Importance_Category', size='Importance', palette=color, sizes=(50, 200), ax=ax)
        
        # Set title and labels
        ax.set_xlabel('Importance')
        ax.set_ylabel(' ')

        # Split long y labels into two lines
        y_labels = [textwrap.fill(label.get_text(), width=20) for label in scatter.get_yticklabels()]
        ax.set_yticklabels(y_labels)

        # Remove legend
        scatter.legend_.remove()

        # Adjust the space for y label
        plt.subplots_adjust(left=0.3)
          
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        return plt

    except Exception as e:
        print("An error occurred while creating the scatter plot.")
        print(str(e))


# V0lcano
def plot_volcano(df, log2FoldChange_threshold=(2, -2), threshold=2, padj_threshold=0.05, save_path=None):
    """
    Plot a volcano plot based on the provided dataframe.
    
    Parameters:
    - df (DataFrame): The input dataframe containing 'log2FoldChange' and 'padj' columns.
    - log2FoldChange_threshold (tuple): A tuple containing the positive and negative thresholds 
                                        for log2FoldChange. Default is (2, -2).
    - threshold (float): The threshold for -logFDR. Default is 2.
    - padj_threshold (float): The p-adjusted threshold value for significance. Default is 0.05.
    - save_path (str, optional): The path where the plot should be saved. If None, the plot won't be saved. Default is None.
    
    Returns:
    - None: Displays the volcano plot or saves it to the provided path.
    """
    # Set figure size
    plt.figure(figsize=(10, 6))

    # For all other points
    plt.scatter(x=df['log2FoldChange'], y=df['padj'].apply(lambda x: -np.log10(x)), s=10, label="Not significant", color="lightgray", alpha=0.7)

    # Highlight down- or up- regulated genes
    down = df[(df['log2FoldChange'] <= log2FoldChange_threshold[1]) & (df['padj'] <= padj_threshold)]
    up = df[(df['log2FoldChange'] >= log2FoldChange_threshold[0]) & (df['padj'] <= padj_threshold)]

    plt.scatter(x=down['log2FoldChange'], y=down['padj'].apply(lambda x: -np.log10(x)), s=25, label="Down-regulated", color="steelblue", alpha=0.7)
    plt.scatter(x=up['log2FoldChange'], y=up['padj'].apply(lambda x: -np.log10(x)), s=25, label="Up-regulated", color="darkorange", alpha=0.7)

    plt.xlabel("log2FoldChange")
    plt.ylabel("-logFDR")
    plt.axvline(log2FoldChange_threshold[1], color="grey", linestyle="--")
    plt.axvline(log2FoldChange_threshold[0], color="grey", linestyle="--")
    plt.axhline(threshold, color="grey", linestyle="--")
    plt.legend()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)

    # Display the plot
    plt.show()


# ********************** Time Series Clusters **********************
# Plot each line, and put them in the same figure (for each Cluster)
def plot_ts_clusters(result, processed_gene, palette_name='crest', save_fig=False):
    result = result[['Name', 'Cluster']]
    data = merge_dataframes([result, processed_gene])

    # Drop the 'Name' column
    data = data.drop(columns=['Name'])

    # Reset index, group by 'Cluster', and set the index back
    data = data.reset_index()
    grouped = data.groupby('Cluster')

    # Set color palette
    palette = sns.color_palette(palette_name, 12)  # The palette_name palette has maximum 12 distinct colors

    # Iterate over groups (clusters) and plot each one
    for name, group in grouped:
        group = group.set_index('index')  # set the index back to 'index'
        group = group.drop(columns='Cluster')  # drop the 'Cluster' column

        plt.figure(figsize=(10, 3))
        for i, feature in enumerate(group.index):
            plt.plot(group.columns, group.loc[feature], color=palette[i % len(palette)])

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Cluster: {name}', weight='bold')
        # plt.savefig('result/plot/ts_cluster.png')
        # plt.show()

        if save_fig:
            # Save the figure with a unique filename for each cluster
            plt.savefig(f"result/plot/ts_cluster_{name}.png")
        else:
            plt.show()


def plot_ts_clusters_average(result, processed_gene, palette_name='Paired', save_fig=None):
    """
    Plot each line, and put them in the same figure (for each Cluster)
    
    Parameters:
        result: the result dataframe including the 'Name' and 'Cluster' columns
        processed_gene: the dataframe (with the 'Name' column) to plot
    
    result: the result dataframe from the clustering algorithm
    """
    data = merge_dataframes([result, processed_gene])

    # Drop the 'Name' column
    data = data.drop(columns=['Name'])

    # Reset index, group by 'Cluster', and set the index back
    data = data.reset_index()
    grouped = data.groupby('Cluster')

    # Set color palette
    palette = sns.color_palette(palette_name, 12)  # The palette_name palette has maximum 12 distinct colors

    # Iterate over groups (clusters) and plot each one
    for name, group in grouped:
        group = group.set_index('index')  # set the index back to 'index'
        group = group.drop(columns='Cluster')  # drop the 'Cluster' column

        # Convert the group dataframe to long form
        long_form = pd.melt(group.reset_index(), id_vars='index')

        plt.figure(figsize=(20, 8))
        
        # Use seaborn lineplot to plot each line (gene) and the average line with 95% CI
        sns.lineplot(x='variable', y='value', data=long_form, color=palette[int(name) % len(palette)], ci=95)

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Cluster: {name}', weight='bold')
        
        if save_fig:
            # Save the figure with a unique filename for each cluster
            plt.savefig(f"result/plot/ts_cluster_{name}.png")
        else:
            plt.show()


def plot_network(data, target_index, num_nodes=20, save_path=None):
    """
    Plot the network graph of the top num_nodes similar nodes for the target.
    
    Parameters:
        data: the dataframe containing the similarity data
        target_index: the index of the target node
        num_nodes: the number of nodes to plot
    """
    # Compute the similarity
    similarity = cosine_similarity(data)
    similarity_df = pd.DataFrame(similarity, index=data.index, columns=data.index)
    
    # Get the top num_nodes similar nodes for the target
    target_similarities = similarity_df.loc[target_index].sort_values(ascending=False)[1:num_nodes+1].index

    # Create a network graph
    G = nx.Graph()

    # Add nodes and edges
    for node1 in target_similarities:
        for node2 in target_similarities:
            if node1 != node2 and similarity_df.loc[node1, node2] > 0.7:
                G.add_edge(node1, node2, weight=similarity_df.loc[node1, node2])
                
        # Add edge from the target node to all similar nodes
        G.add_edge(target_index, node1, weight=similarity_df.loc[target_index, node1])

    # Add the target node
    G.add_node(target_index)
    
    # Define color maps for nodes and edges
    node_color = [i for i in range(G.number_of_nodes())] 
    edge_color = [G[u][v]['weight'] for u,v in G.edges()]

    # Draw the network
    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.Blues)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='darkorange', font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, edge_cmap=plt.cm.viridis, width=2, alpha=0.2)  # 20% transparency

    plt.axis('off')  # to turn off the frame
    
    if save_path:
        plt.savefig(save_path)
        
    else:
        plt.show()


# Hexbin
def plot_hexbin(data, x_axis, y_axis, gridsize=20, save_path=None):
    """
    Plot a hexbin plot of the specified columns in the input dataframe.
    
    Parameters:
        data (pandas.DataFrame): The input dataframe.
        x_axis (str): The name of the column to plot on the x-axis.
        y_axis (str): The name of the column to plot on the y-axis.
    """
    sns.jointplot(x=x_axis, y=y_axis, data=data, kind='hex', gridsize=20)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return plt


# Line Heatmap
def plot_line_heatmap(df, name_value, cmap='vlag'):  
    """
    Plot a line plot and a heatmap of the specified row in the input dataframe.
    Heatmap includes all columns except 'Name'.
    
    Parameters:
        df (pandas.DataFrame): The input dataframe.
        name_value (str): The value in the 'Name' column of the row to plot.
        cmap (str): The name of the colormap to use for the heatmap. Defaults to 'vlag'.
    """
    # Find the row where 'Name' is equal to name_value
    row = df.loc[df['Name'] == name_value]

    # Extract the desired columns and convert them to a list
    desired_columns = ['Granger', 'log2FoldChange', 'Pearson', 'Spearman']
    list1 = row[desired_columns].values.tolist()[0]

    # Get the other columns by dropping the ones already in list1
    remaining_columns = df.drop(columns=['Name'] + desired_columns)
    list2 = remaining_columns.loc[remaining_columns.index == row.index[0]].values.tolist()[0]

    # Reshape list1 into a 2D array for the heatmap
    list1_2d = np.array(list1).reshape(-1, 1)

    # Create subplots with adjusted sizes
    fig = plt.figure(figsize=(10, 2))
    grid = plt.GridSpec(1, 10, hspace=0.2, wspace=0.2)  # We'll use 8 columns in total

    # Plot the line graph on the left (using 7 out of 8 columns)
    ax1 = plt.subplot(grid[:9])  # equivalent to grid[0, :7]
    ax1.plot(list2, color='navy')
    ax1.set_title(name_value)
    ax1.set_xticks(range(len(remaining_columns.columns)))
    ax1.set_xticklabels(remaining_columns.columns, rotation=90)

    # Plot the heatmap on the right (using 1 out of 8 columns) without color bar
    ax2 = plt.subplot(grid[9:])  # equivalent to grid[0, 7:]
    sns.heatmap(list1_2d, ax=ax2, cmap=cmap, cbar=False, yticklabels=desired_columns)
    ax2.yaxis.tick_right()
    ax2.xaxis.set_visible(False)  # hide the x-axis
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)  # set rotation to 0

    # Show the plotc
    plt.tight_layout()
    plt.show()


def save_table_as_svg(df, save_path=None):
    """
    Save a pandas dataframe as an svg file.
    """ 
    # Set global font properties
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    fig, ax = plt.subplots(1,1)
    #set figure size
    fig.set_size_inches(12, 5.5)

    cell_text = []
    for row in range(len(df)):
        cell_text.append([f"{val:.5}" if isinstance(val, float) else (str(val)[:66] + '....' if len(str(val)) > 70 else str(val)) for val in df.iloc[row]])

    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=cell_text, colLabels=df.columns, cellLoc='center', loc='center')

    # Autoscale
    table.auto_set_column_width(list(range(len(df.columns))))
    table.scale(1, 1.5)  # may need to adjust based on specific table size

    # Change cell color for header cells and make them bold
    for (row, col), cell in table.get_celld().items():
        if (row == 0):
            cell.set_facecolor("steelblue")  # change color as desired
            cell.get_text().set_color('white')  # change font color as desired
            cell.get_text().set_fontsize(12)  # make headers slightly larger
            cell.get_text().set_weight('bold')  # make headers bold
        elif row % 2 == 0:  # alternate color for non-header cells
            cell.set_facecolor("seashell")
        else:
            cell.set_facecolor("aliceblue")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_result(result, score_column, log2FoldChange_column, Granger_column, save_path=None, color_palette="crest", size_scale=150):

    # Ensure 'Granger_column' and 'log2FoldChange_column' are in the DataFrame
    if Granger_column not in result.columns or log2FoldChange_column not in result.columns or score_column not in result.columns:
        raise ValueError("One of the specified columns is missing from the DataFrame.")

    # work on a copy: do not add helper columns to / modify the caller's DataFrame
    result = result.copy()

    # Handle NaN or infinite values in 'log2FoldChange_column'
    result[log2FoldChange_column] = result[log2FoldChange_column].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale 'log2FoldChange_column' using MinMaxScaler
    model = MinMaxScaler()
    result['scaled_log2FoldChange'] = model.fit_transform(result[log2FoldChange_column].values.reshape(-1, 1))
    
    # Calculate size based on 'log2FoldChange', adjusting with the 'size_scale'
    result['size'] = np.abs(result['scaled_log2FoldChange']) * size_scale

    # Colour value = "evidence strength" (darker/higher colour = stronger evidence), following
    # scoring.METRIC_DIRECTION:
    #   'pvalue' (raw p, e.g. Granger): -log10(p)
    #   'lower'  (a distance, e.g. DTW): -x  (smaller distance = stronger)
    #   otherwise: the value itself
    color_values = pd.to_numeric(result[Granger_column], errors='coerce').astype(float)
    color_label = Granger_column
    direction = METRIC_DIRECTION.get(str(Granger_column))
    if direction == 'pvalue':
        color_values = -np.log10(color_values.clip(lower=1e-300, upper=1.0))
        color_label = f"-log10({Granger_column} p-value)"
    elif direction == 'lower':
        color_values = -color_values
        color_label = f"-{Granger_column} (distance; higher = smaller distance = stronger)"

    # Normalize the colour values between 0 and 1 for color mapping
    granger_norm = MinMaxScaler()
    result['Granger_norm'] = granger_norm.fit_transform(color_values.fillna(color_values.min()).values.reshape(-1, 1))

    # Get color palette
    palette = sns.color_palette(color_palette, as_cmap=True)

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 7)
    plt.subplots_adjust(left=0.2)

    # Scatter plot with normalized Granger_column values for color mapping
    points = ax.scatter(x=result[score_column], y=result['Name'], 
                        c=result['Granger_norm'], s=result['size'], 
                        cmap=palette, edgecolor='black')
    
    # Invert y axis
    ax.invert_yaxis()

    # Create colorbar based on the normalized Granger_column values
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label(color_label)

    plt.xlabel(score_column)
    plt.ylabel("Index") # Assuming 'Index' is the desired y-axis label

    # Save or show plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
