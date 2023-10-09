library(DESeq2)
library(readr)

#read file
express_file <- "result/matrix_fc.csv"
express <- read.csv(express_file, row.names = 1, header = TRUE)
# If column names start with "X", remove "X"
names(express) <- gsub("^X", "", names(express))

metadata <- read_csv('result/design_fc.csv')



express <- round(express)
# Create a DESeqDataSet object
dds <- DESeqDataSetFromMatrix(countData = express,
                              colData = metadata,
                              design = ~ group)
# Compute
dds <- DESeq(dds)
res <- results(dds)

# filtered_DEGs <- res[which(res$padj < 0.05 & abs(res$log2FoldChange) > 1), ]

# Save result
res_sub <- res[, c("log2FoldChange")]
# rownames(res_sub) <- rownames(res)
res_sub <- data.frame("index" = rownames(res), "log2FoldChange" = res$log2FoldChange)


#res_sub <- data.frame(log2FoldChange = res$log2FoldChange)
#rownames(res_sub) <- rownames(res)
write.csv(res, "result/fc_for_volcano.csv")
write.csv(res_sub, "result/fc.csv", row.names=FALSE)

print('The fold change has been calculated, please check result/fc.csv')


