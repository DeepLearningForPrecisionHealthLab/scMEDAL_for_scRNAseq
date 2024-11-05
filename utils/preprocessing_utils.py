import scanpy as sc

def scRNAseq_pipeline(adata, min_genes_per_cell=10, min_cells_per_gene=3, total_counts_per_cell=10000, n_top_genes=1000, n_components=50):
    # 1. Filter low quality cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    # 2. Normalize with total UMI count per cell
    sc.pp.normalize_total(adata, target_sum=total_counts_per_cell)

    # 3. Logarithmize the data
    sc.pp.log1p(adata)

    # 4. Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # Filter the data for highly variable genes
    adata = adata[:, adata.var.highly_variable]

    # 5. Apply PCA: This solver uses the ARPACK implementation of the truncated singular value decomposition (SVD). It's more suitable for smaller datasets or when you need a small number of components (principal components). It can be slower but is typically more accurate, especially when the number of requested components is much smaller than the number of features.
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_components)

    return adata

def plot_pca(adata, sample_id_column='sampleID'):

    sc.pl.pca(adata, color=sample_id_column)

# import scanpy as sc


def scRNAseq_pipeline_log(adata, min_genes_per_cell=10, min_cells_per_gene=3, total_counts_per_cell=10000):
    # Create a copy of the data to keep the raw data unchanged
    adata_copy = adata.copy()
    
    # 1. Filter low quality cells and genes
    sc.pp.filter_cells(adata_copy, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata_copy, min_cells=min_cells_per_gene)

    # 2. Normalize with total UMI count per cell
    sc.pp.normalize_total(adata_copy, target_sum=total_counts_per_cell)

    # 3. Logarithmize the data
    sc.pp.log1p(adata_copy)
    return adata_copy

def scRNAseq_pipeline_loghvg(adata, min_genes_per_cell=10, min_cells_per_gene=3, total_counts_per_cell=10000,n_top_genes=3000):
    # Create a copy of the data to keep the raw data unchanged
    adata_copy = adata.copy()
    
    # 1. Filter low quality cells and genes
    sc.pp.filter_cells(adata_copy, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata_copy, min_cells=min_cells_per_gene)

    # 2. Normalize with total UMI count per cell
    sc.pp.normalize_total(adata_copy, target_sum=total_counts_per_cell)

    # 3. Logarithmize the data
    sc.pp.log1p(adata_copy)
    # 4. Select highly variable genes and subset
    sc.pp.highly_variable_genes(adata_copy, n_top_genes=n_top_genes, subset=True)
    return adata_copy

def scRNAseq_pipeline_v2(adata, min_genes_per_cell=10, min_cells_per_gene=3, total_counts_per_cell=10000, n_top_genes=1000, n_components=50):
    # Create a copy of the data to keep the raw data unchanged
    adata_copy = adata.copy()
    
    # 1. Filter low quality cells and genes
    sc.pp.filter_cells(adata_copy, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata_copy, min_cells=min_cells_per_gene)

    # 2. Normalize with total UMI count per cell
    sc.pp.normalize_total(adata_copy, target_sum=total_counts_per_cell)

    # 3. Logarithmize the data
    sc.pp.log1p(adata_copy)

    # 4. Select highly variable genes and subset
    sc.pp.highly_variable_genes(adata_copy, n_top_genes=n_top_genes, subset=True)

    # 5. Scale the data
    sc.pp.scale(adata_copy)

    # 6. Apply PCA
    sc.tl.pca(adata_copy, svd_solver='arpack', n_comps=n_components)

    # 7. Compute the neighborhood graph
    sc.pp.neighbors(adata_copy)

    # 8. Run UMAP
    sc.tl.umap(adata_copy)

    # Return the processed data
    return adata_copy

def plot_results(adata_copy, batch_column="BATCH", celltype_column="celltype",file_name ='heart_latent'):
    # Plot UMAP with color annotations for batch and cell type
    sc.pl.pca(adata_copy,color=[batch_column, celltype_column],save=file_name+'.png')
    sc.pl.umap(adata_copy,color=[batch_column, celltype_column],save=file_name+'.png')
    