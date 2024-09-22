import catbridge as cat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from itertools import cycle
import seaborn as sns



gene_file = 'data/merged_tpm.csv'
metabo_file = 'data/merged_metabo.csv'
design_file = None
annotation_file = 'data/description.csv'

score = cat.compute_corr(gene_file, metabo_file, design_file, annotation_file, 'Acetyl-CoA', cluster_count=8, aggregation_func=cat.repeat_aggregation_mean, E=3, tau=3, lag=3)
score.to_csv('result/ecoli_score.csv', index=False)
result = cat.compute_score(score, "CCM")
result.to_csv('result/ecoli_result.csv', index=False)



def plot_metabos_tpm(metabo_names, tpm_name, metabo_df, tpm_df, file_name):
    plt.figure(figsize=(14, 6))
    
    tpm_data = tpm_df[tpm_df['Name'] == tpm_name].iloc[:, 1:].astype(float)
    if tpm_data.empty:
        print(f"TPM name '{tpm_name}' not found in the dataframe.")
        return

    tpm_min = tpm_data.min(axis=1).values[0]
    tpm_max = tpm_data.max(axis=1).values[0]
    tpm_normalized = (tpm_data - tpm_min) / (tpm_max - tpm_min)
    time_points = tpm_data.columns
    x_original = np.arange(len(time_points))


    x_smooth = np.linspace(x_original.min(), x_original.max(), 300)
    tpm_smooth = make_interp_spline(x_original, tpm_normalized.values[0])(x_smooth)
    
    
    colors = cycle(["#FE632A", "#CD5C5C", "#E8A317", "#50EBEC", "#A2AD9C", "#EE82EE"])
    gene_data_found = False
    for metabo_name in metabo_names:
        metabo_data = metabo_df[metabo_df['Name'] == metabo_name].iloc[:, 1:].astype(float)
        if not metabo_data.empty:
            gene_data_found = True
            metabo_min = metabo_data.min(axis=1).values[0]
            metabo_max = metabo_data.max(axis=1).values[0]
            metabo_normalized = (metabo_data - metabo_min) / (metabo_max - metabo_min)
            metabo_smooth = make_interp_spline(x_original, metabo_normalized.values[0])(x_smooth)
            color = next(colors)
            plt.plot(x_smooth, metabo_smooth, linestyle='--', color=color, label=metabo_name, linewidth=1.5)
        else:
            print(f"Metabolite name '{metabo_name}' not found in the dataframe.")
    
    if gene_data_found:
        plt.plot(x_smooth, tpm_smooth, label='Acetyl-CoA', color='#F88017', linewidth=4)
    
    
    plt.xlabel('Time (h)')
    plt.ylabel('Normalized Value')
    plt.title('')
    plt.legend()
    plt.grid(False)
    plt.xticks(np.linspace(0, len(time_points) - 1, len(time_points)), time_points, rotation=30)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
    
    
def map_time_points(df):
    return df.rename(columns=lambda x: time_mapping[x] if x in time_mapping else x)

plot_metabos_tpm(['pta', 'acs'], 'Acetyl-CoA', gene_df, metabo_df, 'result/figuare_a.svg')
plot_metabos_tpm(['aceE', 'aceF', 'lpd'], 'Acetyl-CoA', gene_df, metabo_df, 'result/figure_b.svg')