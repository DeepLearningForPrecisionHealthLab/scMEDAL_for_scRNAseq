
import os
import sys
import gc
import shutil
import random

import pandas as pd
import numpy as np
import matplotlib

# Set the Matplotlib backend to 'Agg'
matplotlib.use('Agg')

from scMEDAL.utils.compare_results_utils import (
    get_recon_paths_df,
    get_input_paths_df
)
from scMEDAL.utils.genomaps_utils import (
    select_cells_from_batches,
    find_intersection_batches,
    process_and_plot_genomaps_singlepath,
    create_count_matrix_multibatch,
    compute_cell_stats_acrossbatchrecon,
    plot_cell_recon_genomap
)
from scMEDAL.utils.utils import read_adata

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
    input_base_path,
    path2results_file,
    scaling
)

# I run this script with Aixa_genomap env
# --------------------------------------------------------------------------------------
# 1. Get input paths and recon paths
# --------------------------------------------------------------------------------------

# Merge paths
df_recon = get_recon_paths_df(results_path_dict, get_batch_recon_paths=True)
df_inputs = get_input_paths_df(input_base_path)
df = pd.merge(df_recon, df_inputs, on=["Split", "Type"], how="left")

# Add 'recon_prefix' column
df["recon_prefix"] = [
    recon_path.split("/")[-1].split(".npy")[0] for recon_path in df["ReconPath"]
]

print("Reading paths,\ndf paths:", df.head(5))


# --------------------------------------------------------------------------------------
# 1.2. Base: I will run the genomap for random effects reconstructions: (scMEDAL-RE) outputs
# --------------------------------------------------------------------------------------
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

# Define experiment output directory
out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])
if not os.path.exists(out_name):
    os.makedirs(out_name)
print("Saving results to", out_name)


# --------------------------------------------------------------------------------------
# Copy path2results.py and 'pipeline_CMmultibatch_genomap_and_plot_file.py' file
# --------------------------------------------------------------------------------------
path2results_destination_path = os.path.join(out_name, 'paths_config.py')

print("\nCopying paths_config.py and pipeline_CMmultibatch_genomap_and_plot_file.py file to:", out_name)
pipeline_CMmultibatch_genomap_and_plot_file = os.path.abspath(__file__)
pipeline_CMmultibatch_destination_path = os.path.join(
    out_name, 'pipeline_CMmultibatch_genomap_and_plot_file.py'
)
# Copy the files
shutil.copy(path2results_file, path2results_destination_path)
shutil.copy(pipeline_CMmultibatch_genomap_and_plot_file, pipeline_CMmultibatch_destination_path)

# --------------------------------------------------------------------------------------
# Get batch reconstruction paths and prefix (to indicate batch)
# --------------------------------------------------------------------------------------

recon_paths = df.loc[
    (df["Key"] == model_name) & (df["Split"] == Split) & 
    (df["Type"] == Type) & (df["recon_prefix"] != "recon_train"),
    "ReconPath"
].values

recon_prefix = df.loc[
    (df["Key"] == model_name) & (df["Split"] == Split) & 
    (df["Type"] == Type) & (df["recon_prefix"] != "recon_train"),
    "recon_prefix"
].values

print("n recon paths:", len(recon_paths))

# Get inputs path: Same for all models, split and type.
inputs_path = df.loc[
    (df["Key"] == model_name) & (df["Split"] == Split) & (df["Type"] == Type),
    "InputsPath"
].values[0]


# --------------------------------------------------------------------------------------
# 1.2. Extra (optional): If extra paths are added, set add_inputs_fe = True
# Add cells from input + model reconstructions for the genomap
# --------------------------------------------------------------------------------------
add_inputs_fe = True
# extra_recon = "fe" or "all"
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

elif extra_recon == "all":
    n_inputs_fe = 5
    # Get model recon effects paths: unique for each model, split and type
    fe_ae_path = df.loc[
        (df["Key"] == "scMEDAL-FE") & (df["Split"] == Split) & (df["Type"] == Type),
        "ReconPath"
    ].values[0]
    fe_aec_path = df.loc[
        (df["Key"] == "scMEDAL-FEC") & (df["Split"] == Split) & (df["Type"] == Type),
        "ReconPath"
    ].values[0]
    aec_path = df.loc[
        (df["Key"] == "AEC") & (df["Split"] == Split) & (df["Type"] == Type),
        "ReconPath"
    ].values[0]
    ae_path = df.loc[
        (df["Key"] == "AE") & (df["Split"] == Split) & (df["Type"] == Type),
        "ReconPath"
    ].values[0]
    # Define extra paths and prefix
    extra_paths = [inputs_path, ae_path, aec_path, fe_ae_path, fe_aec_path]
    extra_prefix = [
        f"input_{Type}", f"ae_recon_{Type}", f"aec_recon_{Type}",
        f"fe_ae_recon_{Type}", f"fe_aec_recon_{Type}"
    ]


# --------------------------------------------------------------------------------------
# 2. Get genes and cells metadata
# --------------------------------------------------------------------------------------
# Get genes metadata (var)
# The splits did not store the real gene_ids. This changes on every experiment.
gene_ids_path = os.path.join(data_base_path, scenario_id, "geneids.csv")
var = pd.read_csv(gene_ids_path, index_col="_index")

# Get cell metadata (obs) from inputs_path. This is the same for all models for same Type, split
_, _, obs = read_adata(inputs_path, issparse=True)


# --------------------------------------------------------------------------------------
# 3. Define variables
# --------------------------------------------------------------------------------------
# celltype can be a list of cell types
celltype = ["Ventricular_Cardiomyocyte", "Endothelial", "Fibroblast", "Pericytes"]
n_cells_per_batch = 300
n_batches = 147

# Define number of genes in the genomap = colNum*rowNum
n_genes = 2916
colNum = 54  # Column number of genomap
rowNum = 54  # Row number of genomap

if add_inputs_fe:
    recon_prefix = extra_prefix + recon_prefix.tolist()
    recon_paths = extra_paths + recon_paths.tolist()
    n_cells = n_cells_per_batch * (n_batches + n_inputs_fe)
else:
    n_cells = n_cells_per_batch * (n_batches)


# --------------------------------------------------------------------------------------
# 4. Create multibatch count matrix for the genomap
# --------------------------------------------------------------------------------------
print("\nCreating count_matrix_multibatch..")

# If batches are provided
# Change this to specific batches if needed
batches_to_select_from = ["H0037_Apex", "HCAHeart7836681", "HCAHeart8102861", "H0015_septum"]

# Set the seed for reproducibility
seed = 42
random.seed(seed)

adata_multibatch_n_cells = create_count_matrix_multibatch(
    recon_prefix=recon_prefix,
    recon_paths=recon_paths,
    obs=obs,
    var=var,
    n_genes=n_genes,
    n_cells=n_cells_per_batch,
    n_batches=n_batches,
    out_path=out_name,
    add_inputs_fe=add_inputs_fe,
    n_inputs_fe=n_inputs_fe if add_inputs_fe else None,
    celltype=celltype,  # Take cells from celltype only if not None
    save_data=True,
    scaling=scaling,
    issparse=True,
    seed=seed,
    force_batches=batches_to_select_from  # Depends on the experiment
)

adata_multibatch_n_cells.obs.index = adata_multibatch_n_cells.obs.index.astype(int)

gc.collect()
print("adata_multibatch_n_cells.X", adata_multibatch_n_cells.X.shape)
print("adata_multibatch_n_cells.obs", adata_multibatch_n_cells.obs)
print("adata_multibatch_n_cells.var", adata_multibatch_n_cells.var)

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


# --------------------------------------------------------------------------------------
# 5. Get genomaps
# --------------------------------------------------------------------------------------
print("\nComputing genomap..")
genomap_name = f"{n_cells_per_batch}cells_per_batch_{n_batches}batches_{Type}_{Split}"

if celltype:
    genomap_name += f"_{celltype_name}"
if add_inputs_fe:
    genomap_name += f"_with_{n_inputs_fe}fe_input"

path_2_genomap = os.path.join(out_name, genomap_name)


gene_names = var.index[0:n_genes]

process_and_plot_genomaps_singlepath(
    cm_multibatch_path,
    ncells=n_cells,
    ngenes=n_genes,
    rowNum=rowNum,
    colNum=colNum,
    epsilon=0.0,
    num_iter=100, # Use 2 iterations for testing. Change the number of iterations. We used 100 for our experiments.
    output_folder=path_2_genomap,
    genomap_name=genomap_name,
    gene_names=gene_names
)
print("genomap stored in", path_2_genomap)

gc.collect()

# --------------------------------------------------------------------------------------
# 6. Plot genomaps
# --------------------------------------------------------------------------------------
order = 'C'
statistic = 'std'
if add_inputs_fe:
    # recon2plot = ['input','fe_recon']
    recon2plot = extra_prefix
else:
    recon2plot = []

# Get genomap paths
genomap_path = os.path.join(path_2_genomap, f'genomap_{genomap_name}.npy')
genomap_coordinates_path = os.path.join(
    path_2_genomap, f'gene_coordinates_{genomap_name}.csv'
)
print("genomap_path", genomap_path)

# Read genomap
genomap = np.load(genomap_path)

# Read genomap coordinates
genomap_coordinates = pd.read_csv(genomap_coordinates_path)
genomap_coordinates.rename(columns={"Unnamed: 0": "gene_names"}, inplace=True)

# Get metadata of the multibatch count matrix (cells that we are going to plot)
obs_multibatch = adata_multibatch_n_cells.obs
print("cm_multibatch_path", cm_multibatch_path)
print("obs multibatch", obs_multibatch)

# Determine which cells to plot
cell_ids_all = np.unique(obs_multibatch["_index"].values)

# Initialize Cells to be plotted
cell_ids_2plot = []
if isinstance(celltype, str) or isinstance(celltype, list):
    if isinstance(celltype, list):
        # Get batches that intersect from the cells in cm_multibatch
        intersection_batches = find_intersection_batches(obs_multibatch, celltype)
        print("Intersection of batches:", intersection_batches)
        if batches_to_select_from is None:
            # Select how many batches you want to plot. It will be one per column.
            batches_to_select_from = list(intersection_batches)[0:4]

    print("selected celltypes and batches:", celltype, batches_to_select_from)
    # select cells from batches
    cell_ids_2plot = select_cells_from_batches(
        obs_multibatch, celltype, batches_to_select_from, seed=seed
    )
    # Here, the n cells to plot are different from the number of original batches.
    # One batch can have one cell from every celltype.
    n_batch_cols2plot = len(batches_to_select_from)
else:
    # Here the number of cells to plot is equal to the number of batches to plot
    n_cells_2_plot = 4
    # Randomly select 4 cells from all available cells
    cell_ids_all = obs_multibatch['_index'].values
    cell_ids_2plot = random.sample(list(cell_ids_all), n_cells_2_plot)
    # We will only plot one cell per original batch
    n_batch_cols2plot = n_cells_2_plot

print("Selected cell IDs to plot:", cell_ids_2plot)

# Get the batches that correspond to the original batch of the cells to plot
original_batch_list = []
for cell_id in cell_ids_2plot:
    original_batch = obs_multibatch.loc[
        (obs_multibatch["_index"] == cell_id) & 
        (obs_multibatch["recon_prefix"] == "input_train"), "batch"
    ].values[0]
    original_batch_list.append(original_batch)

# Plot cell_id with few batches (only for batches that came originally from a cell_id in cell_ids2plot)
recon2plot = recon2plot + original_batch_list

plot_min = -2
plot_max = 8

# Get stats and plot cell_ids
for cell_id in cell_ids_2plot:
    print("cell_id", cell_id)

    # Get indexes of cells of cell_id (CF reconstructions for cell_id)
    # Get original batch for cell_id
    original_batch = obs_multibatch.loc[
        (obs_multibatch["_index"] == cell_id) & 
        (obs_multibatch["recon_prefix"] == "input_train"), "batch"
    ].values[0]
    print("original batch:", original_batch)

    original_celltype = obs_multibatch.loc[
        (obs_multibatch["_index"] == cell_id) & 
        (obs_multibatch["recon_prefix"] == "input_train"), "celltype"
    ].values[0]
    print("original celltype:", original_celltype)

    # Get cell indexes for cell_id
    cell_indexes = obs_multibatch.loc[
        obs_multibatch["_index"] == cell_id
    ].index.values.astype(int)
    print("n cell indexes", cell_indexes)

    # Get cell indexes for cell_id that are recon cf (contains batch in the recon_prefix)
    cell_indexes_batch_cf = obs_multibatch.loc[
        (obs_multibatch["_index"] == cell_id) &
        (obs_multibatch["recon_prefix"].str.contains('batch'))
    ].index.values.astype(int)
    print("n cell indexes for batch CF recon", cell_indexes_batch_cf)
    print("obs_multibatch['recon_prefix']", obs_multibatch["recon_prefix"].values)

    # Get genes that vary the most across batches
    genomap_coordinates = compute_cell_stats_acrossbatchrecon(
        genomap,
        cell_indexes_batch_cf,
        genomap_coordinates,
        statistic=statistic,
        n_top_genes=10,
        order='C',
        path_2_genomap=path_2_genomap,
        file_name=cell_id
    )

    # Plot n top variable genes for all batches
    print(genomap_coordinates[genomap_coordinates["Top_N"]])
    plot_cell_recon_genomap(
        genomap,
        cell_indexes,
        genomap_coordinates,
        obs=obs_multibatch,
        original_batch=original_batch,
        n_top_genes=10,
        min_val=plot_min,
        max_val=plot_max,
        order='C',
        path_2_genomap=path_2_genomap,
        file_name=f"{cell_id}_{statistic}_{original_celltype}"
    )

    cell_indexes_few_batches = obs_multibatch.loc[
        (obs_multibatch["_index"] == cell_id) &
        obs_multibatch["recon_prefix"].apply(lambda x: any(r in x for r in recon2plot))
    ].index.values

    plot_cell_recon_genomap(
        genomap,
        cell_indexes=cell_indexes_few_batches,
        genomap_coordinates=genomap_coordinates,
        obs=obs_multibatch,
        original_batch=original_batch,
        n_top_genes=10,
        min_val=plot_min,
        max_val=plot_max,
        n_cols=n_batch_cols2plot + n_inputs_fe,
        order='C',
        path_2_genomap=path_2_genomap,
        file_name=f"{cell_id}_few_batches_{statistic}_{original_celltype}_labels",
        remove_ticks=True
    )

    plot_cell_recon_genomap(
        genomap,
        cell_indexes=cell_indexes_few_batches,
        genomap_coordinates=None,
        obs=obs_multibatch,
        original_batch=original_batch,
        n_top_genes=10,
        min_val=plot_min,
        max_val=plot_max,
        n_cols=n_batch_cols2plot + n_inputs_fe,
        order='C',
        path_2_genomap=path_2_genomap,
        file_name=f"{cell_id}_few_batches_{statistic}_{original_celltype}",
        remove_ticks=True
    )