
import pandas as pd
import glob
import os
import re
import numpy as np
# import scipy.stats as stats

from anndata import AnnData
import scanpy as sc

import pandas as pd # Please install pandas and matplotlib before you run this example
import matplotlib.pyplot as plt
import matplotlib
# Set the Matplotlib backend to 'Agg'
matplotlib.use('Agg')
import numpy as np
import scipy
# import genomap as gp
import numpy as np
import scipy.stats as stats
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sklearn.metrics as mpd
# from genomap.genomapOPT import create_space_distributions, gromov_wasserstein_adjusted_norm
# from genomap.genomap import createMeshDistance

import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
from compare_results_utils import get_recon_paths_df,get_input_paths_df
from genomaps_utils import process_and_plot_genomaps_singlepath,create_count_matrix_multibatch,compute_cell_stats_acrossbatchrecon,plot_cell_recon_genomap
from utils import read_adata
# import the path
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/compare_results")
# Make sure you update the expt. For example expt=="expt3_batch_cf" has an example of getting counterfactuals of batch effects
from path2results import run_names_dict,results_path_dict,run_names_dict,compare_models_path 
import gc
# I run this script with Aixa_genomap env

###############################################################################
# 1. Get input paths and recon paths

# 1.1. Define base paths
data_base_path = "/archive/bioinformatics/DLLab/AixaAndrade/data/Genomic_data/heart_data/"
scenario_id = "Healthy_human_heart_data/log_transformed_3000hvggenes"
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')

# Merge paths
df_recon = get_recon_paths_df(results_path_dict,get_batch_recon_paths = True)
df_inputs = get_input_paths_df(input_base_path)
df = pd.merge(df_recon, df_inputs, on=["Split", "Type"], how="left")
df["recon_prefix"] = [recon_path.split("/")[-1].split(".npy")[0] for recon_path in df["ReconPath"]]
print("Reading paths,\ndf paths:", df.head(5))

#########################################################################################
##################### 1. 2. Base: I will run the genomap for random effects reconstructions: (AE_RE) outputs
# Define Lists of models, types, and splits. This script will only run one genomap.
models = ['AE_RE']  # Add all your models to this list
types = ['train']  # Add all types you need to iterate through
splits = [1] 

# Select the first model, type, and split
Type = types[0]
Split = splits[0]
model_name = models[0]

# RE recon path: The counterfactual batches are stored in this directory
re_recon_path = os.path.join(results_path_dict["AE_RE"], f"splits_{Split}")

# Define experiment output directory
out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])
if not os.path.exists(out_name):
    os.makedirs(out_name)


# Get batch reconstruction paths and prefix (to indicate batch)
recon_paths = df.loc[(df["Key"]==model_name)&(df["Split"]==Split)&(df["Type"]==Type)&(df["recon_prefix"]!="recon_train"),"ReconPath"].values
recon_prefix = df.loc[(df["Key"]==model_name)&(df["Split"]==Split)&(df["Type"]==Type)&(df["recon_prefix"]!="recon_train"),"recon_prefix"].values
print("n recon paths:",len(recon_paths))

# Get inputs path: Same for all models, split and type.
inputs_path = df.loc[(df["Key"]==model_name)&(df["Split"]==Split)&(df["Type"]==Type),"InputsPath"].values[0]

######################### 1. 2. Extra (optional): if extra paths are added, set add_inputs_fe = True
# Add cells from input + model reconstructions for the genomap
add_inputs_fe = True
# extra_recon = "fe"
extra_recon = "all"
# extra_recon = input + fe 
if extra_recon == "fe":
    # n_inputs_fe = 2: One for inputs and another one for fixed effects (fe). You could also add a fe classifier recon and a base autoencoder recon.
    n_inputs_fe = 2 
    # Get fixed effects path: Unique for AE_DA model, split and type.
    fe_ae_path = df.loc[(df["Key"]=="AE_DA")&(df["Split"]==Split)&(df["Type"]==Type),"ReconPath"].values[0]
    # Define extra paths and expre prefix
    extra_paths = [inputs_path, fe_ae_path]
    extra_prefix = [f"input_{Type}",f"fe_ae_recon_{Type}"]
# extra_recon = input + all models
elif extra_recon == "all":
    n_inputs_fe = 5
    # Get model recon effects paths: Unique for each model, split and type.
    fe_ae_path = df.loc[(df["Key"]=="AE_DA")&(df["Split"]==Split)&(df["Type"]==Type),"ReconPath"].values[0]
    fe_aec_path = df.loc[(df["Key"]=="AEC_DA")&(df["Split"]==Split)&(df["Type"]==Type),"ReconPath"].values[0]
    aec_path = df.loc[(df["Key"]=="AEC")&(df["Split"]==Split)&(df["Type"]==Type),"ReconPath"].values[0]
    ae_path = df.loc[(df["Key"]=="AE")&(df["Split"]==Split)&(df["Type"]==Type),"ReconPath"].values[0] 
    # Define extra paths and expre prefix
    extra_paths = [inputs_path, ae_path, aec_path, fe_ae_path, fe_aec_path]
    extra_prefix = [f"input_{Type}",f"ae_recon_{Type}",f"aec_recon_{Type}",f"fe_ae_recon_{Type}",f"fe_aec_recon_{Type}"] 

###########################################################################
# 2. Get genes and cells metadata

# Get genes metadata (var)
# The splits did not store the real gene_ids. This changes on every experiment
gene_ids_path = os.path.join(data_base_path, scenario_id,"geneids.csv")
var = pd.read_csv(gene_ids_path,index_col="_index")

# Get cell metadata (obs) from inputs_path. This is the same for all models for same Type, split
_, _, obs = read_adata(inputs_path, issparse=True)

###########################################################################
# 3. Define variables

celltype = "Endothelial"
n_cells_per_batch = 300
n_batches = 147

# Define number of genes in the genomap = colNum*rowNum
n_genes = 2916 #
colNum=54 # Column number of genomap
rowNum=54 # Row number of genomap


if add_inputs_fe:
    recon_prefix = extra_prefix + recon_prefix.tolist()
    recon_paths = extra_paths + recon_paths.tolist()
    n_cells = n_cells_per_batch * (n_batches + n_inputs_fe)

else:
    n_cells = n_cells_per_batch * (n_batches)
###########################################################################
print("\nCreating count_matrix_multibatch..")
# 4. Create multibatch count matrix for the genomap
adata_multibatch_n_cells = create_count_matrix_multibatch(recon_prefix,
                            recon_paths,
                            obs,
                            var,
                            n_genes = n_genes,
                            n_cells = n_cells_per_batch,
                            n_batches = n_batches,
                            out_path = re_recon_path,
                            add_inputs_fe = add_inputs_fe,
                            n_inputs_fe = n_inputs_fe,
                            celltype = celltype, # Take cells from celltype only if not None
                            save_data = True
                            )

adata_multibatch_n_cells.obs.index = adata_multibatch_n_cells.obs.index.astype(int)

gc.collect()
print("adata_multibatch_n_cells.X",adata_multibatch_n_cells.X.shape)
print("adata_multibatch_n_cells.obs",adata_multibatch_n_cells.obs)
print("adata_multibatch_n_cells.var",adata_multibatch_n_cells.var)

cm_multibatch_name = f"CMmultibatch_{n_cells_per_batch}_cells_per_batch_{n_batches}batches" if celltype is None else f"CMmultibatch_{n_cells_per_batch}cells_per_batch_{n_batches}batches_{celltype}"
if add_inputs_fe:
    cm_multibatch_name += f"_with_{n_inputs_fe}fe_input"
cm_multibatch_path = os.path.join(re_recon_path, cm_multibatch_name)



###########################################################################
# 5. Get genomaps
print("\nComputing genomap..")
genomap_name = f"{n_cells_per_batch}cells_per_batch_{n_batches}batches_{Type}_{Split}"

if celltype:
    genomap_name += f"_{celltype}"
if add_inputs_fe:
    genomap_name += f"_with_{n_inputs_fe}fe_input"

path_2_genomap  = os.path.join(out_name, genomap_name)
print("genomap stored in",path_2_genomap)
gene_names = var.index[0:n_genes]

process_and_plot_genomaps_singlepath(cm_multibatch_path ,
                                        ncells = n_cells,
                                        ngenes = n_genes,
                                        rowNum=rowNum,
                                        colNum=colNum,
                                        epsilon = 0.0,
                                        num_iter=100,
                                        output_folder = path_2_genomap,
                                        genomap_name = genomap_name,
                                        gene_names = gene_names)

gc.collect()

###########################################################################
# 6. Plot genomaps

# Define variables
order = 'C'
statistic = 'std'
if add_inputs_fe:
    # recon2plot = ['input','fe_recon']
    recon2plot = extra_prefix 
else:
    recon2plot = []

# Get genomap paths
genomap_path = os.path.join(path_2_genomap,f'genomap_{genomap_name}.npy')
genomap_coordinates_path = os.path.join(path_2_genomap, f'gene_coordinates_{genomap_name}.csv')
print("genomap_path",genomap_path)
# Read genomap
genomap = np.load(genomap_path)
# Read genomap coordinates
genomap_coordinates = pd.read_csv(genomap_coordinates_path)
genomap_coordinates.rename(columns={"Unnamed: 0": "gene_names"}, inplace=True)
# get metadata of the multibatch count matrix (cells that we are going to plot)
obs_multibatch = adata_multibatch_n_cells.obs
print("cm_multibatch_path",cm_multibatch_path)

# obs_multibatch_path = cm_multibatch_path+"/meta.csv"
# obs_multibatch = pd.read_csv(obs_multibatch_path)
print("obs multibatch",obs_multibatch)

# Determine which cells 2 plot
cell_ids_all = np.unique(obs_multibatch["_index"].values)
# n_cells_2_plot = 4
# cell_ids_2plot = cell_ids_all[:n_cells_2_plot]

# Hand picked cells
cell_ids_2plot = ["b'TCAGCAATCAGTTCGA-1-HCAHeart7829976'",
                    "b'ATGTCCCTCATGCCCT-1-H0037_Apex'", # batch effect looks big
                    "b'CGGGTCACAGTTGGTT-1-HCAHeart8102868'",
                    "b'GGGCATCAGCCAGTTT-1-HCAHeart7850546'"]

n_cells_2_plot = len(cell_ids_2plot)

# get the batches that correspond to the original batch of the cells to plot
original_batch_list = []
for cell_id in cell_ids_2plot:
    original_batch = obs_multibatch.loc[(obs_multibatch["_index"] ==cell_id)&(obs_multibatch["recon_prefix"]=="input_train"),"batch"].values[0]
    original_batch_list.append(original_batch)

# Plot cell_id with few batches (only for batches that came originally from a cell_id in cell_ids2plot)
recon2plot = recon2plot + original_batch_list



# Get stats and plot cell_ids
for cell_id in cell_ids_2plot:
    print("cell_id",cell_id)


    # Get indexes of cells of cell_id (CF reconstructions for cell_id)
    # get original batch for cell_id
    original_batch = obs_multibatch.loc[(obs_multibatch["_index"] ==cell_id)&(obs_multibatch["recon_prefix"]=="input_train"),"batch"].values[0]
    print("original batch:",original_batch)
    # get cell indexes for cell_id
    cell_indexes = obs_multibatch.loc[obs_multibatch["_index"] == cell_id].index.values
    # Convert the array of index values to integers
    cell_indexes = cell_indexes.astype(int)
    print("n cell indexes",cell_indexes)
    # get cell indexes for cell_id: that are recon cf (contains batch in the recon_prefix).
    cell_indexes_batch_cf = obs_multibatch.loc[(obs_multibatch["_index"] == cell_id) & (obs_multibatch["recon_prefix"].str.contains('batch'))].index.values
    # Convert the array of index values to integers
    cell_indexes_batch_cf = cell_indexes_batch_cf.astype(int)

    print("n cell indexes for batch CF recon",cell_indexes_batch_cf)
    print("obs_multibatch['recon_prefix']",obs_multibatch["recon_prefix"].values)



    # Get genes that vary the most across batches
    
    genomap_coordinates=compute_cell_stats_acrossbatchrecon(genomap,
                                                                cell_indexes_batch_cf,
                                                                genomap_coordinates,
                                                                statistic=statistic,
                                                                n_top_genes=10,
                                                                order='C',
                                                                path_2_genomap=path_2_genomap,
                                                                file_name=cell_id)

    # Plot n top variable genes fro all batches
    print(genomap_coordinates[genomap_coordinates["Top_N"]])
    plot_cell_recon_genomap(genomap,
                                cell_indexes,
                                genomap_coordinates, 
                                obs = obs_multibatch, 
                                original_batch = original_batch,
                                n_top_genes=10,
                                min_val=-5,
                                max_val=10,
                                order='C',
                                path_2_genomap=path_2_genomap,
                                file_name=cell_id+"_"+statistic)




    cell_indexes_few_batches = obs_multibatch.loc[
        (obs_multibatch["_index"]==cell_id) & 
        obs_multibatch["recon_prefix"].apply(lambda x: any(recon in x for recon in recon2plot))
    ].index.values

    plot_cell_recon_genomap(genomap,
                                cell_indexes=cell_indexes_few_batches,
                                genomap_coordinates=genomap_coordinates, 
                                obs = obs_multibatch, 
                                original_batch=original_batch,
                                n_top_genes=10,
                                min_val=-5,
                                max_val=10,
                                n_cols = n_cells_2_plot + n_inputs_fe,
                                order='C',
                                path_2_genomap=path_2_genomap,
                                file_name=cell_id+"few_batches_"+statistic)
