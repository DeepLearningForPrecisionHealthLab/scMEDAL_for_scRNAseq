#using Aixa_scDML env
import anndata
import os
import sys
# To import data_base_path from paths_config
# Add the parent directory to the Python path
sys.path.append("../")
# Now you can import from the parent directory
from paths_config import data_base_path,scenario_id
from scMEDAL.utils.utils import read_adata,save_adata
from scMEDAL.utils.preprocessing_utils import scRNAseq_pipeline_loghvg




# Path to the directory containing norm data
parent_path = os.path.join(data_base_path, "adata_merged")
# 1.Read adata
X,var,obs = read_adata(folder_path= parent_path)

# Assuming `X`, `obs`, and `var` are already defined and you've created an AnnData object:
adata = anndata.AnnData(X=X, obs=obs, var=var)

# Convert 'nan' values to string 'nan' for easy filtering
adata.obs["CellType"] = adata.obs["CellType"].astype("str")


# 2.1. Filter cells
# van Galen et al 2019 randomly filtered 783 cells of 1,590 BM5 CD34+CD38- cells to reduce representation of this population
# Condition 1: Exclude 'nan' cells
keep_cells_condition1 = adata.obs["CellType"] != 'nan'

# Dai et al 2021 excludes AML314, AML371, AML722B, and AML997 samples due to unconfident annotations
# https://www.frontiersin.org/articles/10.3389/fcell.2021.762260/full
# Condition 2: Exclude specific IDs
exclude_ids = ['AML314', 'AML371', 'AML722B', 'AML997']
keep_cells_condition2 = ~adata.obs['id'].isin(exclude_ids)

# Combine both conditions using logical AND (&)
keep_cells_combined_condition = keep_cells_condition1 & keep_cells_condition2

# Apply the combined condition to filter `adata` directly
filtered_adata = adata[keep_cells_combined_condition, :].copy()

filtered_adata.obs["celltype"] = filtered_adata.obs["CellType"]
filtered_adata.obs["batch"] = filtered_adata.obs["id"]

print(f"Original number of cells: {adata.X.shape[0]}")
print(f"Number of cells after filtering: {filtered_adata.X.shape[0]}")
print("adata before preprocessing",filtered_adata)

save_data = True
expt = scenario_id
if save_data:
    out_path = os.path.join(data_base_path,expt)
    print(out_path)
    # Check if the directory exists, if not, create it
    if not os.path.exists(out_path):
        os.makedirs(out_path)
if expt == "log_transformed_2916hvggenes":
    adata_log_hvg = scRNAseq_pipeline_loghvg(filtered_adata,
                                             min_genes_per_cell=10,
                                             min_cells_per_gene=3,
                                             total_counts_per_cell=10000,
                                             n_top_genes=2916)
    if save_data:
        save_adata(adata_log_hvg,out_path)
    print("adata after preprocessing",adata_log_hvg)