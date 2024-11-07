import os              
        
import pandas as pd                                                    
import numpy as np                                                     
import scanpy as sc                                                                                                                                      


from matplotlib import pyplot as plt
import seaborn as sns

from anndata import read_mtx
from anndata.utils import make_index_unique
import anndata

from scipy.stats import gamma

import glob
import copy
import sys


sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")

from utils import save_adata
from preprocessing_utils import scRNAseq_pipeline_loghvg


parent_path = "/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics/Genomic_data/ASD/norm"
out_path = os.path.join(parent_path,"reverse_norm")



# 1. Read the dataset and metadata
# Since the dataset is in genes*cells format, we transpose it while reading
adata_log2 = sc.read_text(parent_path + "/exprMatrix.tsv", first_column_names=True).T

# Read meta data and gene IDs
meta = pd.read_csv(parent_path + "/meta.tsv", sep="\t")
gene_ids = pd.read_csv(parent_path + "/geneids.txt", sep="\t")

# Assign metadata and gene IDs to the AnnData object
adata_log2.obs = meta
adata_log2.var['gene_ids'] = gene_ids.values

# Ensure uniqueness of gene and cell names
adata_log2.var_names_make_unique(join="-")
adata_log2.obs_names_make_unique(join="-")

# Print dimensions of the AnnData object to confirm cells*genes format
print("AnnData object dimensions (cells*genes):", adata_log2.X.shape)

# 2.1. Preprocessing step: Reverse log2 transformation of the data
adata = copy.deepcopy(adata_log2)
# Rename cluster column (otherwise it is confusing)
adata.obs['celltype'] = adata.obs['cluster']
adata.obs['celltype'] = adata.obs['celltype'].astype('category')
#adata.obs['donor'] = adata.obs['individual']
adata.obs['donor'] = ['donor_'+str(d) for d in adata.obs['individual']]
adata.obs['donor'] = adata.obs['donor'].astype('category')
adata.obs['batch'] = adata.obs['donor'] 
print("Normalized data range before reversing log2:", np.min(adata.X), np.max(adata.X))

# Reverse log2 transformation: Add 1 before applying exp2 to revert log2(x+1)
adata.X = np.exp2(adata.X) - 1
print("Data range after reversing log2 transformation:", np.min(adata.X), np.max(adata.X))
print("adata",adata)


save_data = True
expt = "log_transformed_2916hvggenes"
if save_data:
    out_path = os.path.join(out_path,expt)
    print(out_path)
    # Check if the directory exists, if not, create it
    if not os.path.exists(out_path):
        os.makedirs(out_path)
if expt == "log_transformed_2916hvggenes":
    adata_log_hvg = scRNAseq_pipeline_loghvg(adata,
                                             min_genes_per_cell=10,
                                             min_cells_per_gene=3,
                                             total_counts_per_cell=10000,
                                             n_top_genes=2916)
    if save_data:
        save_adata(adata_log_hvg,out_path)
    print("adata after preprocessing",adata_log_hvg)