"""AML Data Processing Script

This script processes the unified count matrix of the AML dataset.  
**Prerequisite:** Run `AML_data_reader.ipynb` first and save the unified count matrix in:  
`/scMEDAL_for_scRNAseq/Experiments/data/AML_data/adata_merged`

The script then preprocesses the data and saves the processed output in:  
`/scMEDAL_for_scRNAseq/Experiments/data/AML_data/scenario_id`

By default, `scenario_id` is set to `"log_transformed_2916hvggenes"`.

Environment: preprocess_and_plot_umaps_env

"""
import os
import sys
import anndata

# ---- Add Parent Directory to Path ----
sys.path.append("../")  # To import data_base_path from paths_config.py

from paths_config import data_base_path, scenario_id
from scMEDAL.utils.utils import read_adata, save_adata
from scMEDAL.utils.preprocessing_utils import scRNAseq_pipeline_loghvg

# ---- Define Data Paths ----
parent_path = os.path.join(data_base_path, "adata_merged")

# ---- Read AnnData ----
X, var, obs = read_adata(folder_path=parent_path)
adata = anndata.AnnData(X=X, obs=obs, var=var)

# ---- Convert 'nan' to String ----
adata.obs["CellType"] = adata.obs["CellType"].astype("str")

# ---- Filter Cells ----
# van Galen et al. 2019 randomly filtered 783 cells of 1,590 BM5 CD34+CD38- cells to reduce representation of this population
# Condition 1: Exclude 'nan' cells
keep_cells_condition1 = adata.obs["CellType"] != "nan"

# Dai et al. 2021 excludes AML314, AML371, AML722B, and AML997 samples due to unconfident annotations
# https://www.frontiersin.org/articles/10.3389/fcell.2021.762260/full
# Condition 2: Exclude specific IDs
exclude_ids = ["AML314", "AML371", "AML722B", "AML997"]
keep_cells_condition2 = ~adata.obs["id"].isin(exclude_ids)

# Combine both conditions using logical AND (&)
keep_cells_combined_condition = keep_cells_condition1 & keep_cells_condition2

# Apply the combined condition to filter `adata` directly
filtered_adata = adata[keep_cells_combined_condition, :].copy()

# ---- Rename Columns for Clarity ----
filtered_adata.obs["celltype"] = filtered_adata.obs["CellType"]
filtered_adata.obs["batch"] = filtered_adata.obs["id"]

print(f"Original number of cells: {adata.X.shape[0]}")
print(f"Number of cells after filtering: {filtered_adata.X.shape[0]}")
print("adata before preprocessing:", filtered_adata)

# ---- Save Processed Data ----
save_data = True
expt = scenario_id

if save_data:
    out_path = os.path.join(data_base_path, expt)
    print(out_path)
    os.makedirs(out_path, exist_ok=True)  # Create directory if it doesn't exist

# ---- Preprocessing: Log Transformation and HVG Selection ----
if expt == "log_transformed_2916hvggenes":
    adata_log_hvg = scRNAseq_pipeline_loghvg(
        filtered_adata,
        min_genes_per_cell=10,
        min_cells_per_gene=3,
        total_counts_per_cell=10_000,
        n_top_genes=2916,
    )
    
    if save_data:
        save_adata(adata_log_hvg, out_path)
    
    print("adata after preprocessing:", adata_log_hvg)