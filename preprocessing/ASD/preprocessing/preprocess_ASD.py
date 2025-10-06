"""
ASD Data Processing Script

This script processes the ASD dataset obtained from https://cells.ucsc.edu/?ds=autism.
The dataset should be saved in:
    scMEDAL_for_scRNAseq/Experiments/data/ASD_data/norm

The script preprocesses the data and stores the processed output in:
    scMEDAL_for_scRNAseq/Experiments/data/ASD_data/scenario_id

By default, `scenario_id` is set to "log_transformed_2916hvggenes".

Environment:
    This script should be run in the `Aixa_scDML` environment.
"""

                    
import pandas as pd                                                    
import numpy as np                                                      
import scanpy as sc                                                                                                                                      
import copy


import os
import sys
from pathlib import Path

ROOT_PATH  = Path.cwd().resolve().parents[2]  # two levels up from AML dir (scMEDAL_for_scRNAseq)
sys.path.insert(0, str(ROOT_PATH ))
print(ROOT_PATH )

# Now you can import from the parent directory
from utils.defaults import ASD_DATA_DIR,ASD_EXPERIMENT_NAME 
from utils.preprocessing import scRNAseq_pipeline_loghvg
from utils.utils import save_adata


# ---- Define Paths ----
# Path to the directory containing normalized data
parent_path = os.path.join(ASD_DATA_DIR, "norm")

# File name for the dataset
healthy_heart_file = "Healthy_human_heart_adata.h5ad"
file_path = os.path.join(parent_path, healthy_heart_file)

# ---- Step 1: Read the Dataset and Metadata ----
# Load gene expression data (transpose since dataset is in genes*cells format)
adata_log2 = sc.read_text(parent_path + "/exprMatrix.tsv", first_column_names=True).T

# Load metadata and gene IDs
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

# ---- Step 2: Preprocessing ----
# Reverse log2 transformation of the data
adata = copy.deepcopy(adata_log2)

# Rename cluster column to celltype for clarity
adata.obs['celltype'] = adata.obs['cluster'].astype('category')

# Format donor information
adata.obs['donor'] = ['donor_' + str(d) for d in adata.obs['individual']]
adata.obs['donor'] = adata.obs['donor'].astype('category')

# Assign donor information as batch
adata.obs['batch'] = adata.obs['donor'] 

# Print range of normalized data before transformation
print("Normalized data range before reversing log2:", np.min(adata.X), np.max(adata.X))

# Reverse log2 transformation: Add 1 before applying exp2 to revert log2(x+1)
adata.X = np.exp2(adata.X) - 1

# Print range of data after transformation
print("Data range after reversing log2 transformation:", np.min(adata.X), np.max(adata.X))
print("adata:", adata)

# ---- Step 3: Save Processed Data (Optional) ----
save_data = False
expt = ASD_EXPERIMENT_NAME   # Scenario ID determines predefined preprocessing steps

if save_data:
    out_path = os.path.join(ASD_DATA_DIR, expt)
    print(out_path)
    # Create directory if it does not exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)

# ---- Step 4: Additional Preprocessing for Specific Experiments ----
if expt == "log_transformed_2916hvggenes":
    adata_log_hvg = scRNAseq_pipeline_loghvg(
        adata,
        min_genes_per_cell=10,
        min_cells_per_gene=3,
        total_counts_per_cell=10000,
        n_top_genes=2916
    )
    
    if save_data:
        save_adata(adata_log_hvg, out_path)
    
    print("adata after preprocessing:", adata_log_hvg)