# import pandas
import numpy as np
import pandas as pd
import tensorflow as tf

import sys
# sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/utilities")
# from tensorflow_utilities import use_specific_gpu

print(tf.__version__)

# change path to utils
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
from utils import *
from model_train_utils import run_all_folds,get_metric2optimizemodel#,run_model_pipeline,ModelManager,get_train_val_data,load_data,train_and_save_model,PlotLoss,get_pca_scoresandplots,get_encoder_latentandscores
# path to model_config
sys.path.append("../")
from model_config import *
# path to models
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/models")
import glob
import shutil
# import models
from AE_v4 import DomainEnhancingAutoencoderClassifier

# 0. Define Fold and GPU
folds_list = list(range(1,6,1)) #there are 10 folds. Running 2 for testing
intGPU = 1


# Set GPU card
# use_specific_gpu(intGPU, fraction=1)

# #######################################################################################################################
# 1. Define batch and bio cols and order of the donors and celltypes
# If you need to run a quick test: Update the 'epochs' key in the dictionary. Note: This will not update the name of the folder.
# model_params_dict["epochs"] = 20
# Define the batch and bio column
# batch_col = 'batch'
# bio_col = "celltype"
# donor_col = "DonorID"
# tissue_col = "TissueDetail"

batch_col = model_params_dict['batch_col']
bio_col = model_params_dict['bio_col']
donor_col = model_params_dict['donor_col'] #this column is optional

tissue_col = model_params_dict['tissue_col']

# We will use this dictionary to plot latent spaces. The basic combination is  {"shape_col": bio_col, "color_col": batch_col}, but when we have lots of cells, we cannot distinguish shapes.
# We only plot bio_col
shape_color_dict={f"{bio_col}-{bio_col}": {"shape_col": bio_col, "color_col": bio_col},f"{donor_col}-{donor_col}": {"shape_col": donor_col, "color_col": donor_col},f"{tissue_col}-{tissue_col}": {"shape_col": tissue_col, "color_col": tissue_col}}
#f"{bio_col}-{batch_col}": {"shape_col": bio_col, "color_col": batch_col},
#f"{batch_col}-{batch_col}": {"shape_col": batch_col, "color_col": batch_col},
                                    

# Define the One Hot encoded (OHE) order for donor and celltype categories
# get metadata before splits
metadata_all = pd.read_csv(glob.glob(data_path+"/*meta.csv")[0])

metadata_all['celltype'] = metadata_all['celltype'].astype('category')
metadata_all['batch'] = metadata_all["sampleID"].astype('category')

print("n batches",len(np.unique(metadata_all[batch_col]).tolist()))



# Define the One Hot encoded (OHE) order for donor and celltype categories
# Seen donors are only pairs
seen_donor_ids = np.unique(metadata_all[batch_col]).tolist()
print("check ordered batches: ",seen_donor_ids)

# celltypes 
celltype_ids = np.unique(metadata_all[bio_col]).tolist()



######################################################################################################################
#########################################################################################################################
# Notes: If you have compile_dict defined separately from model_params_dict, use it to compile model. Othewise extract compile_dict from model_params_dict using compile_keys.
# compile_keys = ["loss_gen_weight","loss_recon_weight","loss_class_weight"]
# compile_dict = filter_keys(model_params_dict, compile_keys)

# If you have build_model_dict defined separately, use it to build model. 
# Othewise extract compile_dict from model_params_dict using buil_model_keys.
# build_model_keys =  ["n_latent_dims","layer_units","n_clusters","layer_units_latent_classifier","n_pred","get_pred","last_activation","name"]
# build_model_dict= filter_keys(model_params_dict, build_model_keys)

# Run all folds. It returns a dataframe with 1/db, ch and silhouette score for celltypes and donors.
# higher scores, better clustering. We need low clustering for batch and high clustering for celltype.
mean_scores = run_all_folds(Model=DomainEnhancingAutoencoderClassifier,
                input_base_path=data_seen,
                out_base_paths_dict=base_paths_dict,
                folds_list=folds_list, #there are 10 folds for just running 2 for testing
                run_name=run_name,
                model_params_dict=model_params_dict,
                build_model_dict=build_model_dict,
                compile_dict=compile_dict,
                save_model=save_model,
                batch_col=batch_col,
                bio_col=bio_col,
                batch_col_categories=seen_donor_ids,
                bio_col_categories=celltype_ids,
                model_type="ae_re",
                issparse=True,
                load_dense=True,                
                shape_color_dict=shape_color_dict,
                sample_size=model_params_dict["sample_size"])

# maximizing biological clustering and minimizing batch cluster
# metric2optimize = bio_mean - batch_mean
# metric2optimize = get_metric2optimizemodel(mean_scores, subset='val', metric='silhouette', batch_col='donor', bio_col='celltype')

    
# print("mean scores\n",mean_scores)
# # For HPO: We want to maximize celltype scores, and minimize donor scores
# print("val donor Silhouette:",mean_scores["val"].loc[('donor', 'silhouette'), 'mean'])
# print("val celltype Silhouette:",mean_scores["val"].loc[('celltype', 'silhouette'), 'mean'])
# print("metric to optimize",-1*metric2optimize)

##############################################################################################
############## save config.py file
destination_path = os.path.join(saved_models_base, run_name, 'model_config.py')

# Ensure the destination directory exists
# os.makedirs(os.path.dirname(destination_path), exist_ok=True)

print("\nCopying config.py file to:", destination_path)

# Copy the file
shutil.copy(source_file, destination_path)




