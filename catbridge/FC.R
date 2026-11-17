# Required Libraries
library(tidyverse)
library(DESeq2)
library(readr)

# Function to Generate Design File and Calculate Fold Changes
calculate_fc <- function(express_file = "rna_seq_tpm.csv"){
  
  # Load Data
  express <- read.csv(express_file, row.names = 1, header = TRUE)
  names(express) <- gsub("^X", "", names(express))
  
  # Remove NAs from count data
  express <- express[complete.cases(express), ]
  
  # Generate design file
  design <- tibble(sample = colnames(express),
                   group = str_extract(sample, "^[^_]*"))
  
  write.csv(design, "design.csv", row.names = FALSE)
  
  express <- round(express)
  
  # List all groups
  groups <- unique(design$group)
  
  # Loop over groups to calculate fold change for each
  for (group in groups){
    if (group == "WT"){
      next
    }
    # Create a DESeqDataSet object
    dds <- DESeqDataSetFromMatrix(countData = express,
                                  colData = design,
                                  design = ~ group)
    # Compute DESeq
    dds <- DESeq(dds)
    
    # Calculate results
    res <- results(dds, contrast = c("group", group, "WT"))
    
    # Save result
    res_sub <- data.frame("index" = rownames(res), "log2FoldChange" = res$log2FoldChange)
    write.csv(res_sub, paste0("result/fc_", group, ".csv"), row.names = FALSE)
  }
  print('The fold change has been calculated, please check the result files in the result folder')
}

# Call function
calculate_fc()

