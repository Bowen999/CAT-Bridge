import matplotlib.pyplot as plt
import catbridge as cat
import os
import psutil
import time
import numpy as np
import sys
import shutil


#gene_file = 'pepper/count.tsv'
#metabo_file = 'pepper/metabo.tsv'
#design_file = 'pepper/design.tsv'
#annotation_file = 'pepper/gene_annotation.tsv'
#target = 'Capsaicin'
#cluster_count = 8
#f = cat.repeat_aggregation_max
gene_file = sys.argv[1]
metabo_file = sys.argv[2]
design_file = sys.argv[3]
a_if = sys.argv[4]
if a_if == "no":
    annotation_file = None
else:
    annotation_file = a_if 
target = sys.argv[5]
cluster_count = sys.argv[6]
cluster_count = int(cluster_count)
f_if = sys.argv[7]
if f_if == "f1":
    f = cat.repeat_aggregation_max
elif f_if == "f2":
    f = cat.repeat_aggregation_mean
position = sys.argv[8]

print("Transcriptomics file:", gene_file)
print("Metabolomics file:", metabo_file)
print("Study design:", design_file)
print("Annotaion file:", a_if)
print("Target is:", target)
print("Cluster number:", cluster_count)
print("Aggreation function:", f_if)
print("Position:", position)


def print_memory_and_cpu_usage():
    # Collect current process details
    process = psutil.Process(os.getpid())
    
    # Print CPU usage
    print("CPU percent: ", psutil.cpu_percent())
    
    # Print memory usage
    mem_info = process.memory_info()
    print("Memory RSS: ", mem_info.rss)
    print("Memory VMS: ", mem_info.vms)
    print("\n")

start_time = time.time()
if not os.path.exists('myapp/result/' + position):
    os.makedirs('myapp/result/' + position)

if not os.path.exists('myapp/result/' + position + '/plot'):
    os.makedirs('myapp/result/' + position + '/plot')




#remove "" of the target
target = target.replace('"', '')

metabo = cat.read_upload(metabo_file)
gene = cat.read_upload(gene_file)
design = cat.read_upload(design_file)

print()
print()
print()

print('Gene:')
print(gene.head(10))
print()
print('Metabo:')
print(metabo.head(10))
print()
print('Design:')
print(design.head(10))


# Compute
result1 = cat.pipeline(gene_file, metabo_file, design_file, annotation_file, target, cluster_count, max_lag=1, aggregation_func=f)
result = result1.head(20)
result1.to_csv("myapp/result/" + position + "/plot/result.csv", index=False)

# AI Yunafang
text_to_save = """Please provide a gene annotation file to use this feature. 
For how to obtain it, please refer to: http://www.catbridge.work/myapp/tutorial/. 

Due to network reasons, the server sometimes cannot connect to the OpenAI API. 
You can try to use the Python package version to use this feature."""

# Specify the filename
filename = "myapp/result/" + position + "/plot/ai.txt"

# Write the text to the file
with open(filename, 'w') as file:
    file.write(text_to_save)



try:
    result = result.drop('Pearson', axis=1)
    result = result.drop('Description_Score', axis=1)
except:
    pass

print(result.head(20))



print()
print()
print('******************* result has been computed ****************')

# Plot result
cat.save_table_as_svg(result, "myapp/result/" + position + "/plot/table.svg")
cat.plot_result(result, 'Score', 'log2FoldChange', 'Granger', "myapp/result/" + position + "/plot/result.svg")
print()
print()
print('********************* result has been plotted *****************')



processed_gene = f(gene, design)
processed_metabo = f(metabo, design)
print(processed_gene)
print(processed_metabo)
t = cat.get_target(target, processed_metabo)
merged = cat.merge_and_reduce(gene, metabo, n_components=3)
g_imp = cat.top_important_features(processed_gene, 15)
m_imp = cat.top_important_features(processed_metabo, 15)
scaled_gene = cat.scale_df(processed_gene)
scaled_metabo = cat.scale_df(processed_metabo)



# Plot
cat.plot_top_features(g_imp, color='flare', save_path='myapp/result/' + position + '/plot/g_imp.svg')
cat.plot_top_features(m_imp, color='crest', save_path='myapp/result/' + position + '/plot/m_imp.svg')
print()
print()
print('***************** top features have been plotted ************')

cat.plot_line(processed_metabo, target, 'myapp/result/' + position + '/plot/line.svg')
print()
print()
print('******************* line plot has been plotted **************')


# Plot PCA
cat.plot_pca(gene, design, 7, save_path='myapp/result/' + position + '/plot/g_pca.svg')
cat.plot_pca(metabo, design, 7, save_path='myapp/result/' + position + '/plot/m_pca.svg')
cat.plot_pca(merged, design, 7, save_path='myapp/result/' + position + '/plot/merged_pca.svg')
print()
print()
print('******************** pca plot has been plotted ****************')



# Plot Network
cat.plot_network(metabo, target, 20, 'myapp/result/' + position + '/plot/network.svg')
print()
print()
print('************************ network has been plotted ***********************')

# # cat.plot_ts_clusters(result1, processed_gene, palette_name='mako', save_fig=True)
# # print()
# # print()
# # print('************************ time series clusters have been plotted ********************')


cat.plot_heatmap(scaled_metabo, palette='crest', save_path='myapp/result/' + position + '/plot/m_heatmap.svg')
cat.plot_heatmap(scaled_gene, palette='flare', n_clusters=1000, save_path='myapp/result/' + position + '/plot/g_heatmap.svg')
print()
print()
print('************************** heatmaps have been plotted ***********************')


cat.plot_volcano('result/fc_for_volcano.csv', 2, 0.05)
shutil.move("/home/ubuntu/catbridge/volcano.png", "myapp/result/" + position + "/plot/volcano.png")
print()
print()
print('************************ volcano plot has been plotted *****************')

# report
print_memory_and_cpu_usage()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
