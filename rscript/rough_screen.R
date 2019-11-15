#读入count矩阵
cancer_count<-read.table(header = T, file = "C:/Users/73978/Desktop/数据挖掘/结直肠癌/TCGA-COAD.htseq_counts.tsv")
head(cancer_count)

library("biomaRt")
listEnsembl()
cancer_ensembl = useMart("ensembl")
listDatasets(cancer_ensembl)
cancer_ensembl = useDataset("hsapiens_gene_ensembl", mart = cancer_ensembl)

#利用Ensembl的API转换Gene ID
filters = listFilters(cancer_ensembl)
attributes = listAttributes(cancer_ensembl)
View(attributes)

bmd<-getBM(attributes = c("external_gene_name","en8sembl_gene_id_version"), 
           filters = "ensembl_gene_id_version", values = cancer_count$Ensembl_ID, mart = cancer_ensembl)

rownames(cancer_count) <- cancer_count$Ensembl_ID
cancer_count_match <- cancer_count[bmd$ensembl_gene_id_version,]

#合并重复数据、未匹配数据
library('dplyr')
Convert_ID_unique <- distinct(bmd,external_gene_name,.keep_all = TRUE)
cancer_count_unique <- cancer_count_match[Convert_ID_unique$ensembl_gene_id_version,]
rownames(cancer_count_unique) <-Convert_ID_unique$external_gene_name
Sample_name <- colnames(cancer_count_unique)
Gene_ID <- rownames(cancer_count_unique)

Sample_name <-gsub(".","-",Sample_name, fixed = T)
colnames(cancer_count_unique) <- Sample_name
Sample_name <-Sample_name[-1]
View(Sample_name)

#读入erzide表型数据
#pheno<-read.table(header = T, file = "C:/users/73978/Desktop/数据挖掘/胃癌/TCGA-STAD.GDC_phenotype.tsv",fill = T, sep = "\t")
pheno <- read.csv(file = "C:\\Users\\73978\\Desktop\\数据挖掘\\结直肠癌\\TCGA-COAD.GDC_phenotype.csv")
tran<-as.data.frame(pheno$pathologic_M)
rownames(tran)<-rownames(pheno)
tran$`pheno$pathologic_M`<-gsub("M0", "0", tran$`pheno$pathologic_M`)
#View(tran)
tran$`pheno$pathologic_M`<-gsub("M1b", "1", tran$`pheno$pathologic_M`)
tran$`pheno$pathologic_M`<-gsub("M1a", "1", tran$`pheno$pathologic_M`)
tran$`pheno$pathologic_M`<-gsub("M1", "1", tran$`pheno$pathologic_M`)
tran$`pheno$pathologic_M`<-gsub("MX","NA", tran$`pheno$pathologic_M`)
#View(tran)
tran$`pheno$pathologic_M`<-gsub("MX","NA", tran$`pheno$pathologic_M`)
tran$`pheno$pathologic_M`<-gsub("","NA", tran$`pheno$pathologic_M`)
View(tran)
tran$`pheno$pathologic_M`<-gsub("NA0NA","0", tran$`pheno$pathologic_M`)
tran$`pheno$pathologic_M`<-gsub("NA1NA","1", tran$`pheno$pathologic_M`)
View(tran)
tran$`pheno$pathologic_M`<-gsub("NANANA",NA, tran$`pheno$pathologic_M`)
tran$`pheno$pathologic_M`<-gsub("NA",NA, tran$`pheno$pathologic_M`)
View(tran)
tran_raw<-as.data.frame(pheno$pathologic_M)
rownames(tran)<-pheno$锘縮ubmitter_id.samples
View(tran)
tran_filter <- subset(tran, !is.na(tran$`pheno$pathologic_M`))

#匹配表型数据、生存数据与计数矩阵
pheno_filter <- tran_filter
pheno_filter <- cbind(pheno_filter, c(1:length(rownames(pheno_filter))))
pheno_filter <- pheno_filter[-grep(pattern = '11A',x = rownames(pheno_filter)),]
pheno_filter <- pheno_filter[-grep(pattern = '11B',x = rownames(pheno_filter)),]
View(pheno_filter)
pheno_match <- pheno_filter[Sample_name,]
pheno_match <- pheno_match[grep(pattern="TCGA", x=rownames(pheno_match)),]
pheno_match[2] <- pheno_match[1]
pheno_match[1] <- rownames(pheno_match)
pheno_match <- pheno_match[2]
View(pheno_match)

Sample_name <- rownames(pheno_match)
cancer_count_unique <- cancer_count_unique[Sample_name]
cancer_count_unique <- cancer_count_unique[grep(pattern = "TCGA", x = colnames(cancer_count_unique))]
View(cancer_count_unique)

write.csv(cancer_count_unique, file="colon_count.csv")
# write.csv(Survival_con_match, file="Survival_con_modified.csv")
colnames(pheno_match) <- "x"
write.csv(pheno_match, file="colon_tran.csv")

# ---------------------------------------------------------------------
# 使用Seurat对注释好的计数矩阵进行第一次预筛选
# ---------------------------------------------------------------------

library(Seurat)
library(ggplot2)

# 
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install("Seurat", version = "3.8")
# install.packages("colorspace")

pbmc.data <- cancer_count_unique
dense.size <- object.size(x = as.matrix(x = pbmc.data))
dense.size
sparse.size <- object.size(x = pbmc.data)
sparse.size
dense.size/sparse.size
# Initialize the Seurat object with the raw (non-normalized data).  Keep all
# genes expressed in >= 3 cells (~0.1% of the data). Keep all cells with at
# least 200 detected genes

pbmc <- CreateSeuratObject(raw.data = pbmc.data, min.cells = 3, min.genes = 200)

# The number of genes an0d/jeq1 UMIs (nGene and nUMI) are automatically calculated
# for every object by Seurat.  For non-UMI data, nUMI represents the sum of
# the non-normalized values within a cell We calculate the percentage of
# mitochondrial genes here and store it in percent.mito using AddMetaData.
# We use object@raw.data since this represents non-transformed and
# non-log-normalized counts The % of UMI mapping to MT-genes is a common
# scRNA-seq QC metric.
mito.genes <- grep(pattern = "^MT-", x = rownames(x = pbmc@data), value = TRUE)
percent.mito <- Matrix::colSums(pbmc@raw.data[mito.genes, ])/Matrix::colSums(pbmc@raw.data)

# AddMetaData adds columns to object@meta.data, and is a great place to
# stash QC stats n n n
pbmc <- AddMetaData(object = pbmc, metadata = percent.mito, col.name = "percent.mito")
VlnPlot(object = pbmc, features.plot = c("nGene", "nUMI", "percent.mito"), nCol = 3)
pbmc <- NormalizeData(object = pbmc, normalization.method = "LogNormalize", 
                      scale.factor = 10000)
pbmc <- FindVariableGenes(object = pbmc, mean.function = ExpMean, dispersion.function = LogVMR, 
                          x.low.cutoff = 0.0125, x.high.cutoff = 1.5, y.cutoff = -1.5)

pbmc <- ScaleData(object = pbmc, vars.to.regress = c("nUMI", "percent.mito"))
pbmc <- RunPCA(object = pbmc, pc.genes = pbmc@var.genes, do.print = TRUE, pcs.print = 1:5, 
               genes.print = 5)
PCAPlot(object = pbmc, dim.1 = 1, dim.2 = 2)

# save.SNN = T saves the SNN so that the clustering algorithm can be rerun
# using the same graph but with a different resolution value (see docs for
# full details)

pbmc <- FindNeighbors(object = pbmc, dims = 1:10)

pbmc <- FindClusters(object = pbmc, reduction.type = "pca", dims.use = 1:10, 
                     resolution = 0.6, print.output = 0, save.SNN = TRUE)
PrintFindClustersParams(object = pbmc)
pbmc <- RunTSNE(object = pbmc, dims.use = 1:10, do.fast = TRUE)
# note that you can set do.label=T to help label individual clusters
TSNEPlot(object = pbmc)
pbmc.markers <- FindAllMarkers(object = pbmc, only.pos = TRUE)
pbmc.markers %>% group_by(cluster) %>% top_n(n = 2, wt = avg_logFC)


export <- as.matrix(pbmc@data)
export <- as.data.frame(export)
export_rowname <- rownames(pbmc.markers)
export <- export[export_rowname,]
export <- export[-grep(pattern = 'NA', x = rownames(export)),]
write.csv(export, file = "colon_filtered.csv")
