# ----------------------------------------------------------------------------
# Import Standard Library
# ----------------------------------------------------------------------------
import os
import sys

# ----------------------------------------------------------------------------
# Import Third-Party Libraries
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
from anndata import AnnData
from scipy.stats import mannwhitneyu

# Set the Matplotlib backend to 'Agg'
matplotlib.use('Agg')

from utils.compare_results_utils import (
    get_recon_paths_df,
    get_input_paths_df
)

from utils.utils import read_adata

# Add the path to the configuration file
# Path to paths_config: /MyscMEDALExpt/Experiments/HealthyHeart/paths_config.py
sys.path.append("../../../")

# Make sure you update the expt
from paths_config import (
    run_names_dict,
    results_path_dict,
    compare_models_path,
    data_base_path,
    scenario_id,
    input_base_path
)

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 1. Load Input and Reconstruction Paths
# ----------------------------------------------------------------------------
df_recon = get_recon_paths_df(results_path_dict, get_batch_recon_paths=True)
df_inputs = get_input_paths_df(input_base_path)

# Merge dataframes to consolidate paths
df = pd.merge(df_recon, df_inputs, on=["Split", "Type"], how="left")

# Extract a prefix identifier for reconstruction files
df["recon_prefix"] = [
    path.split("/")[-1].split(".npy")[0] for path in df["ReconPath"]
]
print("Reading paths,\ndf paths:", df.head(5))

# ----------------------------------------------------------------------------
# 2. Define Model Parameters and Paths
# ----------------------------------------------------------------------------
# Define lists of models, types, and splits. This script will only run one genomap.
models = ['scMEDAL-RE']  # Add all your models to this list
types = ['train']        # Add all types you need to iterate through
splits = [2]             # Get data from fold 2

# Select the first model, type, and split
Type = types[0]
Split = splits[0]
model_name = models[0]

# RE recon path: The counterfactual batches are stored in this directory
re_recon_path = os.path.join(results_path_dict["scMEDAL-RE"], f"splits_{Split}")

# Define output directory to save results
out_name = os.path.join(
    compare_models_path, run_names_dict["run_name_all"])
if not os.path.exists(out_name):
    os.makedirs(out_name)
print("Saving results to", out_name)

# Filter relevant reconstruction paths
recon_paths = df.loc[
    (df["Key"] == model_name)
    & (df["Split"] == Split)
    & (df["Type"] == Type)
    & (df["recon_prefix"] != "recon_train"),
    "ReconPath"
].values
recon_prefix = df.loc[
    (df["Key"] == model_name)
    & (df["Split"] == Split)
    & (df["Type"] == Type)
    & (df["recon_prefix"] != "recon_train"),
    "recon_prefix"
].values
print("n recon paths:", len(recon_paths))

# Path to the original inputs (counts or scaled data)
inputs_path = df.loc[
    (df["Key"] == model_name)
    & (df["Split"] == Split)
    & (df["Type"] == Type),
    "InputsPath"
].values[0]

# ----------------------------------------------------------------------------
# 3. Additional Paths for Fixed Effects
# ----------------------------------------------------------------------------
add_inputs_fe = True
extra_recon = "fe"


if extra_recon == "fe":
    # n_inputs_fe = 2: One for inputs and another one for fixed effects (fe).
    # You could also add a fe classifier recon and a base autoencoder recon.
    n_inputs_fe = 2
    # Get fixed effects path: unique for "scMEDAL-FE"
    fe_ae_path = df.loc[
        (df["Key"] == "scMEDAL-FE") & (df["Split"] == Split) & (df["Type"] == Type),
        "ReconPath"
    ].values[0]
    # Define extra paths and prefix
    extra_paths = [inputs_path, fe_ae_path]
    extra_prefix = [f"input_{Type}", f"fe_ae_recon_{Type}"]

# ----------------------------------------------------------------------------
# 4. Load Gene and Cell Metadata
# ----------------------------------------------------------------------------
# Real gene IDs were not stored in the splits, so we must map them carefully
gene_index_col = "Gene"
gene_ids_path = os.path.join(data_base_path, scenario_id, "geneids.csv")
var = pd.read_csv(gene_ids_path, index_col=gene_index_col)

# Get cell metadata (obs) from inputs_path. It is the same for all models with
# same Type and same split.
_, _, obs = read_adata(inputs_path, issparse=True)

# ----------------------------------------------------------------------------
# 5. Define Variables
# ----------------------------------------------------------------------------
celltype = ["Mono", "Mono-like"]
n_cells_per_batch = 300
n_batches = 19

# The total number of genes used to build the genomap
n_genes = 2916

# Dimensions of the genomap
colNum = 54
rowNum = 54

if add_inputs_fe:
    recon_prefix = extra_prefix + recon_prefix.tolist()
    recon_paths = extra_paths + recon_paths.tolist()
    n_cells = n_cells_per_batch * (n_batches + n_inputs_fe)
else:
    n_cells = n_cells_per_batch * n_batches

# ----------------------------------------------------------------------------
# 6. Load Multibatch Count Matrix
# ----------------------------------------------------------------------------
if isinstance(celltype, str):
    celltype_name = celltype.replace("/", "")
elif isinstance(celltype, list):
    celltype_name = "_".join([ct.replace("/", "") for ct in celltype])
else:
    celltype_name = None

cm_multibatch_name = f"CMmultibatch_{n_cells_per_batch}_cells_per_batch_{n_batches}batches"
if celltype_name:
    cm_multibatch_name += f"_{celltype_name}"
if add_inputs_fe:
    cm_multibatch_name += f"_with_{n_inputs_fe}fe_input"

cm_multibatch_path = os.path.join(out_name, cm_multibatch_name)
print("cm_multibatch_path:", cm_multibatch_path)


X_multibatch, var_multibatch, obs_multibatch = read_adata(
    cm_multibatch_path, issparse=False
)

adata_multibatch_n_cells = AnnData(
    X=X_multibatch, var=var_multibatch, obs=obs_multibatch
)
adata_multibatch_n_cells.obs.index = adata_multibatch_n_cells.obs.index.astype(int)

print("adata_multibatch_n_cells.X", adata_multibatch_n_cells.X.shape)

# ----------------------------------------------------------------------------
# 7. Generate Genomaps paths
# ----------------------------------------------------------------------------
genomap_name = (
    f"{n_cells_per_batch}cells_per_batch_{n_batches}batches_{Type}_{Split}"
)
if celltype_name:
    genomap_name += f"_{celltype_name}"
if add_inputs_fe:
    genomap_name += f"_with_{n_inputs_fe}fe_input"

path_2_genomap = os.path.join(out_name, genomap_name)
print("Genomap stored in", path_2_genomap)

# ----------------------------------------------------------------------------
# 8. Load Genomap Data
# ----------------------------------------------------------------------------
order = 'C'
statistic = 'std'

genomap_path = os.path.join(
    path_2_genomap, f'genomap_{genomap_name}.npy'
)
genomap_coordinates_path = os.path.join(
    path_2_genomap, f'gene_coordinates_{genomap_name}.csv'
)
print("Genomap path:", genomap_path)

genomap = np.load(genomap_path)
genomap_coordinates = pd.read_csv(genomap_coordinates_path)
genomap_coordinates.rename(columns={"Unnamed: 0": "gene_names"}, inplace=True)

# Determine patient group by batch
batches_provided = True
unique_combinations = obs[["Patient_group", "batch"]].drop_duplicates().reset_index(
    drop=True
)
unique_dict = dict(zip(unique_combinations["batch"],
                       unique_combinations["Patient_group"]))
print(unique_dict)

dict_batches = unique_dict
print("Batch dictionary:", dict_batches)
batches_to_select_from = list(dict_batches.keys())

# Obtain AML and control keys (and similarly for cell lines if needed)
aml_keys = [key for key, value in dict_batches.items() if value == "AML"]
aml_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in aml_keys]
aml = obs_multibatch.loc[
    (obs_multibatch["recon_prefix"].str.contains('batch'))
    & (obs_multibatch["recon_prefix"].isin(aml_recon_batch_list))
]

control_keys = [key for key, value in dict_batches.items() if value == "control"]
control_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in control_keys]
control = obs_multibatch.loc[
    (obs_multibatch["recon_prefix"].str.contains('batch'))
    & (obs_multibatch["recon_prefix"].isin(control_recon_batch_list))
]

cl_keys = [key for key, value in dict_batches.items() if value == "celline"]
cl_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in cl_keys]
cl = obs_multibatch.loc[
    (obs_multibatch["recon_prefix"].str.contains('batch'))
    & (obs_multibatch["recon_prefix"].isin(cl_recon_batch_list))
]

# Matching indexes for AML and Control
idx_aml = aml.index
idx_control = control.index

# ----------------------------------------------------------------------------
# 11. Aggregate Genomaps per Batch
# ----------------------------------------------------------------------------
print("Aggregating Genomaps per Batch")
aml_avg_maps = []
for b in aml_recon_batch_list:
    row_inds = obs_multibatch.index[obs_multibatch['recon_prefix'] == b]
    batch_genomaps = genomap[row_inds, :, :, :]
    avg_map = batch_genomaps.mean(axis=0)
    aml_avg_maps.append(avg_map)

ctrl_avg_maps = []
for b in control_recon_batch_list:
    row_inds = obs_multibatch.index[obs_multibatch['recon_prefix'] == b]
    batch_genomaps = genomap[row_inds, :, :, :]
    avg_map = batch_genomaps.mean(axis=0)
    ctrl_avg_maps.append(avg_map)

aml_avg_maps = np.stack(aml_avg_maps, axis=0)    # shape (n_asd, 54, 54, 1)
ctrl_avg_maps = np.stack(ctrl_avg_maps, axis=0)  # shape (n_ctrl, 54, 54, 1)

print("ASD average maps shape:", aml_avg_maps.shape)
print("Control average maps shape:", ctrl_avg_maps.shape)

# ----------------------------------------------------------------------------
# 12. Perform Pixel-wise Mann-Whitney U Test
# ----------------------------------------------------------------------------
print("Perform Pixel-wise Mann-Whitney U Test")

_, height, width, _ = aml_avg_maps.shape
uvals = np.zeros((height, width))
pvals = np.zeros((height, width))

for i in range(height):
    for j in range(width):
        u_stat, p_value = mannwhitneyu(
            aml_avg_maps[:, i, j, 0],
            ctrl_avg_maps[:, i, j, 0],
            alternative="two-sided"
        )
        uvals[i, j] = u_stat
        pvals[i, j] = p_value

# Process gene coordinates for significance
i_coords = genomap_coordinates["pixel_i"].values
j_coords = genomap_coordinates["pixel_j"].values
genomap_coordinates["pval"] = pvals[i_coords, j_coords]
genomap_coordinates = genomap_coordinates.sort_values(by="pval")

p_threshold = 0.05
genomap_coordinates["significant"] = (
    genomap_coordinates["pval"] < p_threshold
)
print("\n# significant genes:",
      len(genomap_coordinates[genomap_coordinates["significant"]]))
print(genomap_coordinates[genomap_coordinates["significant"]])


# Save final dataframe with p-values and gene annotations
genomap_coordinates.to_csv(
    os.path.join(out_name, "pvals_300cellsavg_mwutest.csv")
)