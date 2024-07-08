import sys
# Set 
# expt="expt3_batch_cf"
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/compare_results")
from path2results import run_names_dict,results_path_dict,run_names_dict, glob_like,compare_models_path ,get_pca
import pandas as pd
import glob
import os
import numpy as np
import scipy.stats as stats

from anndata import AnnData
import scanpy as sc

#I will run it  scDML (for UMAP)

import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
from compare_results_utils import get_input_paths_df,get_latent_paths_df,get_umap_plot
from utils import read_adata,min_max_scaling
# import the path
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/compare_results")
# Make sure you update the expt. For example expt=="expt3_batch_cf" has an example of getting counterfactuals of batch effects
from path2results import run_names_dict,results_path_dict,run_names_dict,compare_models_path 


# 1. Get input paths and recon paths

# 1.1. Define base paths
data_base_path = "/archive/bioinformatics/DLLab/AixaAndrade/data/Genomic_data/heart_data/"
scenario_id = "Healthy_human_heart_data/log_transformed_3000hvggenes"
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')

# Merge paths
df_latent = get_latent_paths_df(results_path_dict)
df_inputs = get_input_paths_df(input_base_path)

# Define output path
# Define experiment output directory
out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])
if not os.path.exists(out_name):
    os.makedirs(out_name)
# Define umap path
umap_path = os.path.join(out_name,"umap_1")
# create directory if it does not exist
if not os.path.exists(umap_path):
    os.makedirs(umap_path)

df = pd.merge(df_latent, df_inputs, on=["Split", "Type"], how="left")
df["latent_prefix"] = [model+"_latent_"+Type+"_"+str(Split) for model,Split,Type in zip(df["Key"],df["Split"],df["Type"])]
df["input_prefix"] = [Type+"_"+str(Split) for model,Split,Type in zip(df["Key"],df["Split"],df["Type"])]
print("Reading paths,\ndf paths:", df.head(5))

latent_prefixes = df["latent_prefix"].values
latent_paths = df["LatentPath"].values

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

# Get inputs path: Same for all models, split and type.
inputs_path = df.loc[(df["Key"]==model_name)&(df["Split"]==Split)&(df["Type"]==Type),"InputsPath"].values[0]


# Define common plotting parameters. You will update the outpath after creating model_params with ModelManager
plot_params = {"markers":["x","+","<","h","s",".",'o', 's', '^', '*','1','8','p','P','D','|',0,',','d',2],
                "shape_col":"celltype", 
                "color_col":"celltype",
                "use_rep":"X_umap",
                "clustering_scores":None, 
                "save_fig":True, 
                "showplot":False,
                "palette_choice":"tab20"}

# #Load input data
# X, var, obs= read_adata(inputs_path, issparse=True)
# X = X.toarray()
# # input needs to be scaled because reconstructions are already scaled
# X = min_max_scaling(X)

# adata_input = AnnData(X[0:1000], obs=obs.loc[0:1000-1], var=var)

# sc.pp.neighbors(adata_input,n_neighbors=15)
# sc.tl.umap(adata_input,use_rep="X")
# # sc.pl.umap(adata_input,color=["celltype"])


# outpath = "/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/compare_results/umap_plots"
# plot_rep(adata_input, shape_col="celltype", color_col="celltype", use_rep="X_umap", clustering_scores=None, save_fig=True, outpath=outpath, showplot=False, palette_choice="tab20",file_name="umap_test",**plot_params)

print("Computing umaps")
get_umap_plot(df, umap_path, plot_params,sample_size=50000)