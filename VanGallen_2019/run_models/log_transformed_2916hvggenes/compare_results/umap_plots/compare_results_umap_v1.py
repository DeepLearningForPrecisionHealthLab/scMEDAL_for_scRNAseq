import sys
# Set 

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
from compare_results_utils import get_input_paths_df,get_latent_paths_df,get_umap_plot,filter_models_by_type_and_split,DimensionalityReductionProcessor
# import the path
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/VanGallen_2019/run_models/log_transformed_2916hvggenes/compare_results")
# Make sure you update the expt. For example expt=="expt3_batch_cf" has an example of getting counterfactuals of batch effects
from path2results import run_names_dict,results_path_dict,run_names_dict,compare_models_path,input_base_path,scaling


# 1. Get input paths and recon paths

# Merge paths
df_latent = get_latent_paths_df(results_path_dict)
df_inputs = get_input_paths_df(input_base_path)

print("\ndf_latent",df_latent.columns,df_latent)
print("\ndf_inputs",df_inputs.columns,df_inputs)

# Define output path
# Define experiment output directory
out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])
if not os.path.exists(out_name):
    os.makedirs(out_name)
# Define umap path
umap_path = os.path.join(out_name,"umap_20batches_seed_5_AEC_cce")
# create directory if it does not exist
if not os.path.exists(umap_path):
    os.makedirs(umap_path)

df = pd.merge(df_latent, df_inputs, on=["Split", "Type"], how="left")
df["latent_prefix"] = [model+"_latent_"+Type+"_"+str(Split) for model,Split,Type in zip(df["Key"],df["Split"],df["Type"])]
df["input_prefix"] = [Type+"_"+str(Split) for Split,Type in zip(df["Split"],df["Type"])]
print("Reading paths,\ndf paths:", df.head(5))

latent_prefixes = df["latent_prefix"].values
latent_paths = df["LatentPath"].values

#########################################################################################
##################### 1. 2. 
# Define Lists of models, types, and splits. This script will only run one genomap.
# models = ['AE_RE']  # Add all your models to this list
# types = ['train']  # Add all types you need to iterate through
# splits = [1] 

# # Select the first model, type, and split
# Type = types[0]
# Split = splits[0]
# model_name = models[0]


filter_folds = {
    "AE": 2,
    "AEC": 2,
    "AEC_DA": 2,
    "AE_DA": 2, 
    "AE_RE": 2
}



# filtering entries
filtered_df = filter_models_by_type_and_split(df, filter_folds, Type='train')

# filtered_df = df.loc[df["Type"]=="train",:]
# print(filtered_df)

# Define common plotting parameters. You will update the outpath after creating model_params with ModelManager
plot_params = {"markers":["x","+","<","h","s",".",'o', 's', '^', '*','1','8','p','P','D','|',0,',','d',2],
                "shape_col":"celltype", 
                "color_col":"celltype",
                "use_rep":"X_umap",
                "clustering_scores":None, 
                "save_fig":True, 
                "showplot":False,
                "palette_choice": [
                    '#e6194b',  # Red
                    '#3cb44b',  # Green
                    '#ffe119',  # Yellow
                    '#4363d8',  # Blue
                    '#f58231',  # Orange
                    '#911eb4',  # Purple
                    '#46f0f0',  # Cyan
                    '#f032e6',  # Magenta
                    '#000000',  # Black (replacing the lightest green)
                    '#fabebe',  # Light pink
                    '#008080',  # Teal
                    '#e6beff',  # Lavender
                    '#9a6324',   # Brown,
                    '#d2f53c',  # Lime
                    '#ff69b4',  # Hot pink
                    '#000080',  # Navy
                    '#800000',  # Maroon
                    '#808000',  # Olive
                    '#800080',  # Dark purple
                    '#808080',   # Gray
                    '#ffd700'  # Gold
                        ]}

                #"palette_choice":"tab20"}


print("Computing umaps")
# get_umap_plot(filtered_df, umap_path, plot_params,sample_size=None,scaling=scaling, n_batches_sample=20, batch_col="batch")


processor = DimensionalityReductionProcessor(filtered_df, umap_path , plot_params,
                                                sample_size=None,
                                                n_neighbors=15,
                                                scaling="min_max",
                                                n_batches_sample=19, 
                                                batch_col="batch", 
                                                plot_tsne=False,
                                                n_pca_components=2)
processor.get_dimensionality_reduction_plots(process_allbatches=False,seed = 5, issparse=False)