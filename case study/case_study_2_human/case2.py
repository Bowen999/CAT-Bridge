from catbridge import catbridge as cat
import matplotlib.pyplot as plt
import pandas as pd


# Read input
gene_file = 'data/gene_exp.csv'
metabo_file = 'data/met_con.csv'
design_file = None
annotation_file = None
gene = cat.read_upload(gene_file)
metabo = cat.read_upload(metabo_file)

# Min-max normalization
def min_max_normalize(lst):
    minimum = min(lst)
    maximum = max(lst)
    normalized = [(x - minimum) / (maximum - minimum) for x in lst]
    return normalized

def plot_gene_vs_metabolite(gene, metabo, target, func_gene, gene_color, metabo_color, cluster_count=8, E=3, tau=1):
    # Extract values for the target metabolite and gene
    target_value = metabo.loc[target].tolist()
    gene_value = gene.loc[func_gene].tolist()

    # Compute correlation score
    score = cat.compute_corr(gene_file, metabo_file, design_file, annotation_file, target, cluster_count=cluster_count, 
                             aggregation_func=cat.repeat_aggregation_mean, E=E, tau=tau)
    print(score[score['Name'] == func_gene])

    # Normalize the target values
    normalized_target = min_max_normalize(target_value)
    normalized_gene = min_max_normalize(gene_value)

    # Create x-axis values
    col = list(range(1, len(normalized_gene) + 1))

    # Plot
    plt.figure(figsize=(7, 3))
    plt.plot(col, normalized_gene, label=func_gene, marker='o', color=gene_color)
    plt.plot(col, normalized_target, label=target, marker='o', color=metabo_color)

    # Setting labels and titles
    plt.title('Gene vs Metabolite Plot')
    plt.xlabel('Age')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(False)

    # Remove right and top spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.show()
    
    
# Compute and plot
plot_gene_vs_metabolite(gene, metabo, 'Glucose', 'HK1', 'lightsteelblue', 'darkorange')
plot_gene_vs_metabolite(gene, metabo, 'Glucose', 'HK2', 'lightsteelblue', 'darkorange')
plot_gene_vs_metabolite(gene, metabo, 'Fructose-1,6BP', 'PFKP', 'lightsteelblue', 'cornflowerblue')
plot_gene_vs_metabolite(gene, metabo, 'Fructose-1,6BP', 'FBP1', 'mistyrose', 'cornflowerblue')
plot_gene_vs_metabolite(gene, metabo, 'Fructose-1,6BP', 'ALDOA', 'gainsboro', 'cornflowerblue')