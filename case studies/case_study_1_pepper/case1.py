from catbridge import catbridge as cat



# Score
## With gene annotation
gene_file = 'data/gene.tsv'
metabo_file = 'data/metabolite.tsv'
design_file = 'data/design.tsv'
annotation_file = 'data/annotation.tsv'
target = 'Capsaicin'

score = cat.compute_corr(gene_file, metabo_file, design_file, annotation_file, target, cluster_count=8, aggregation_func=cat.repeat_aggregation_mean)
result = cat.compute_score(score, "CCM")
result.to_csv('result.csv', index=False)
print(result.head(10))



## Without gene annotation
gene_file = 'data/gene.tsv'
metabo_file = 'data/metabolite.tsv'
design_file = 'data/design.tsv'
annotation_file = None
target = 'Capsaicin'

score = cat.compute_corr(gene_file, metabo_file, design_file, annotation_file, target, cluster_count=8, aggregation_func=cat.repeat_aggregation_mean)
result = cat.compute_score(score, "CCM")
result.to_csv('result_wo_gene_annotation.csv', index=False)
print(result.head(10))






# Plot
gene = cat.read_upload(gene_file)
metabo = cat.read_upload(metabo_file)
design = cat.read_upload(design_file)

processed_gene = cat.repeat_aggregation_mean(gene, design)
processed_metabo = cat.repeat_aggregation_mean(metabo, design)
processed_gene.dropna(inplace=True)
processed_metabo.dropna(inplace=True)

t = cat.get_target(target, processed_metabo)
merged = cat.merge_and_reduce(gene, metabo, n_components=3)

scaled_gene = cat.scale_df(processed_gene)
scaled_metabo = cat.scale_df(processed_metabo)



cat.plot_result(result.head(10), 'Score', 'log2FoldChange', 'CCM')
cat.plot_line(processed_metabo, target)
cat.plot_pca(gene, design, 7)
cat.plot_pca(metabo, design, 7)
cat.plot_pca(merged, design, 7)
g_imp = cat.top_important_features(processed_gene, 15)
cat.plot_top_features(g_imp, color='flare')
m_imp = cat.top_important_features(processed_metabo, 15)
cat.plot_top_features(m_imp, color='crest')
cat.plot_network(metabo, target, 20)
cat.plot_heatmap(scaled_metabo, palette='crest')
cat.plot_heatmap(scaled_gene, palette='flare')
fc = cat.compute_fc(gene_file, metabo_file, design_file=design_file, aggregation_func=cat.repeat_aggregation_mean, target=target)
cat.plot_volcano(fc, log2FoldChange_threshold=(2, -2), padj_threshold=0.05)