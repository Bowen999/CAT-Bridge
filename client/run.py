import matplotlib.pyplot as plt
from catbridge import catbridge as cat
import os
import psutil
import time
import numpy as np
import sys
import shutil
# Ignore the specific RuntimeWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import openai








print("""
  _____                       _   
  \_   \ _ __   _ __   _   _ | |_ 
   / /\/| '_ \ | '_ \ | | | || __|
/\/ /_  | | | || |_) || |_| || |_ 
\____/  |_| |_|| .__/  \__,_| \__|
               |_|                         
""")
gene_file = sys.argv[1]
metabo_file = sys.argv[2]


# design_file = sys.argv[3]
design_file = None if len(sys.argv) <= 3 else sys.argv[3]
if design_file == "no":
    design_file = None
else:
    design_file = design_file

a_if = sys.argv[4]
if a_if == "no":
    annotation_file = None
else:
    annotation_file = a_if 

target = sys.argv[5]
target = target.replace('"', '')

cluster_count = sys.argv[6]
cluster_count = int(cluster_count)

f_if = sys.argv[7]
if f_if == "CCM":
    f = 'CCM'
elif f_if == "Granger":
    f = "Granger"
elif f_if == "CCA":
    f = "CCA"
elif f_if == "DTW":
    f = "DTW"
elif f_if == "CCF":
    f = "CCF"
elif f_if == "Spearman":
    f = "Spearman"
elif f_if == "Pearson":
    f = "Pearson"


# position = sys.argv[8]
# repeat_f = cat.repeat_aggregation_mean
if design_file == "no":
    repeat_f = None
else:
    repeat_f = cat.repeat_aggregation_mean


ai_if = sys.argv[8]
if ai_if == "no":
    ai_token = None
else:
    ai_token = ai_if 



print("Transcriptomics file:", gene_file)
print("Metabolomics file:", metabo_file)
print("Study design:", design_file)
print("Annotaion file:", annotation_file)
print("Target is:", target)
print("Cluster number:", cluster_count)
print("Method:", f_if)
# print("Position:", position)
print('Repeat function:', repeat_f)


# record the time
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
if not os.path.exists('result'):
    os.makedirs('result')

if not os.path.exists('result'):
    os.makedirs('result')

print('*******************************************************')
print('*******************************************************')
print()
print()
print()








print("""
   ___       _         _    _                      _         _         
  / _ \ ___ | |_      | |_ | |__    ___         __| |  __ _ | |_  __ _ 
 / /_\// _ \| __|     | __|| '_ \  / _ \       / _` | / _` || __|/ _` |
/ /_  |  __/| |_      | |_ | | | ||  __/      | (_| || (_| || |_| (_| |
\____/ \___| \__|      \__||_| |_| \___|       \__,_| \__,_| \__|\__,_|
                                                                       
""")
metabo = cat.read_upload(metabo_file)
gene = cat.read_upload(gene_file)


print()
print()
print()

print('Gene:')
print(gene)
print()
print('Metabo:')
print(metabo)

if design_file != None:
    design = cat.read_upload(design_file)
    print('Design:')
    print(design.head(10))
else:
    design = None

if annotation_file != None:
    annotation = cat.read_upload(annotation_file)
    print('Annotation:')
    print(annotation.head(10))
else:
    annotation = None







print("""
   ______                                 __      
  / ____/____   ____ ___   ____   __  __ / /_ ___ 
 / /    / __ \ / __ `__ \ / __ \ / / / // __// _ \
/ /___ / /_/ // / / / / // /_/ // /_/ // /_ /  __/
\____/ \____//_/ /_/ /_// .___/ \__,_/ \__/ \___/ 
                       /_/                        
""")

data = cat.compute_corr(gene_file, metabo_file, design_file, annotation_file, target, cluster_count, aggregation_func=repeat_f, lag=1, E=3, tau=1, n_components=1)
print(data.head(10))
result1 = cat.compute_score(data, f)
result = result1.head(20)
print(result)
# Save result
result1.to_csv("result/result.csv", index=False)


# def Yuanfang(df, target, annotation_file, output_path=None):
#     """
#     Use OpenAI's API to generate a question for the user to answer.
#     Question: Which one may be involved in the synthesis of target?
    
#     Parameters:
#         df: the dataframe containing the similarity data
#         target: the target node
#         output_path: (optional) path to save the output as a .txt file
#     """
    
#     annotaion = read_upload(annotation_file)
#     df = pd.merge(df, annotaion, left_on='Name', right_index=True, how='left')
#     df = df.head(100)
#     if 'Description' not in df.columns:
#         error_message = "Please provide a gene annotation file to use this feature. For how to obtain it, please refer to: http://www.catbridge.work/myapp/tutorial/"
#         if output_path:
#             with open(output_path, 'w') as file:
#                 file.write(error_message)
#         else:
#             print(error_message)
#         return

#     # Combining 'Name' and 'Description' columns
#     hits = [f"{name}({desc})" for name, desc in zip(df['Name'], df['Description'])]
#     hits = ', '.join(hits)
    
#     q = hits + '\n\n\nWhich one may be involved in the synthesis of ' + target + '?'
    
#     openai_api_key = getpass.getpass("Please enter your OpenAI API Key: ")
#     openai.api_key = openai_api_key

#     messages = [
#         {"role": "system", "content": "You are a biological chemist and can explain biological mechanisms"},
#         {"role": "user", "content": q}
#     ]

#     completion = openai.ChatCompletion.create(
#         model = "gpt-3.5-turbo",
#         temperature = 0.8,
#         max_tokens = 2000,
#         messages = messages
#     )
    
#     output_content = '\n' + completion.choices[0].message.content + '\n\n\nNOTICE: The output was produced by the large language model GPT 3.5 turbo, so it should only be regarded as a source of inspiration.'
    
#     if output_path:
#         # Save the output to the specified path
#         with open(output_path, 'w') as file:
#             file.write(output_content)
#     else:
#         # Print the output
#         print(output_content)


# text_to_save = cat.Yuanfang(result1, target, annotation_file, output_path=None)

# # Specify the filename
# filename = "/plot/ai.txt"

# # Write the text to the file
# with open(filename, 'w') as file:
#     file.write(text_to_save)

print()
print()
print('******************* result has been computed ****************')







print("""
   ___  _         _   
  / _ \| |  ___  | |_ 
 / /_)/| | / _ \ | __|
/ ___/ | || (_) || |_ 
\/     |_| \___/  \__|
                      
""")
# processing:
if design_file == None:
    design = None
    processed_gene = gene
    processed_metabo = metabo

else: 
    design = cat.read_upload(design_file)
    processed_gene = repeat_f(gene, design)
    processed_metabo = repeat_f(metabo, design)

processed_gene.dropna(inplace=True)
processed_metabo.dropna(inplace=True)

t = cat.get_target(target, processed_metabo)
merged = cat.merge_and_reduce(gene, metabo, n_components=3)
g_imp = cat.top_important_features(processed_gene, 15)
m_imp = cat.top_important_features(processed_metabo, 15)
scaled_gene = cat.scale_df(processed_gene)
scaled_metabo = cat.scale_df(processed_metabo)



print('********************* start plotting *****************')


# Plot result
cat.save_table_as_svg(result, "result/table.svg")
cat.plot_result(result, 'Score', 'log2FoldChange', f, "result/result.svg")
print()
print()
print('********************* result has been plotted *****************')







# Plot VIP
cat.plot_top_features(g_imp, color='flare', save_path='result/g_imp.svg')
cat.plot_top_features(m_imp, color='crest', save_path='result/m_imp.svg')
print()
print()
print('***************** top features have been plotted ************')

# Plot line
cat.plot_line(processed_metabo, target, 'result/line.svg')
print()
print()
print('******************* line plot has been plotted **************')


# Plot PCA
cat.plot_pca(gene, design, 7, save_path='result/g_pca.svg')
cat.plot_pca(metabo, design, 7, save_path='result/m_pca.svg')
cat.plot_pca(merged, design, 7, save_path='result/merged_pca.svg')
print()
print()
print('******************** pca plot has been plotted ****************')



# Plot Network
cat.plot_network(metabo, target, 20, 'result/network.svg')
print()
print()
print('************************ network has been plotted ***********************')

# cat.plot_ts_clusters(result1, processed_gene, palette_name='mako', save_fig=True)
# print()
# print()
# print('************************ time series clusters have been plotted ********************')

# Plot heatmap
cat.plot_heatmap(scaled_metabo, palette='crest', save_path='result/m_heatmap.svg')
cat.plot_heatmap(scaled_gene, palette='flare', n_clusters=1000, save_path='result/g_heatmap.svg')
print()
print()
print('************************** heatmaps have been plotted ***********************')





# Plot Volcano
print()
print()
fc = cat.compute_fc(gene_file, metabo_file, design_file=design_file, aggregation_func=repeat_f, target=target)
cat.plot_volcano(fc,log2FoldChange_threshold=(2, -2), padj_threshold=0.05, save_path='result/volcano.svg')
print('************************ volcano plot has been plotted *****************')



# report
print_memory_and_cpu_usage()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")






print("""
 ██████╗ █████╗ ████████╗    ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗
██╔════╝██╔══██╗╚══██╔══╝    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
██║     ███████║   ██║       ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗  
██║     ██╔══██║   ██║       ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝  
╚██████╗██║  ██║   ██║       ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗
 ╚═════╝╚═╝  ╚═╝   ╚═╝       ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
""")