import sys
import pandas as pd


sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
from compare_results_utils import get_input_paths_df,get_latent_paths_df,create_latent_dict_from_df

#from genomaps_utils import select_cells_from_batches,find_intersection_batches,process_and_plot_genomaps_singlepath,create_count_matrix_multibatch,compute_cell_stats_acrossbatchrecon,plot_cell_recon_genomap
#from utils import read_adata
# import the path
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/VanGallen_2019/run_models/log_transformed_2916hvggenes/compare_results")
# Make sure you update the expt. For example expt=="expt3_batch_cf" has an example of getting counterfactuals of batch effects
from path2results import results_path_dict,input_base_path
import gc
import shutil



from model_train_utils import ModelManager,run_model_pipeline_LatentClassifier_v2_PCA,calculate_metrics_with_ci

from model_config import data_path, model_params_dict,base_paths_dict,run_name,LatentClassifier_config,load_latent_spaces_dict,saved_models_base,source_file


sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/models")
from AE_v4 import MixedEffectsModel
import glob
import numpy as np
import gc  # Import garbage collector module
import os


###############################################################################
# 1. Get input paths and latent paths

# Merge paths
# df_recon = get_recon_paths_df(results_path_dict,get_batch_recon_paths = True)
df_latent = get_latent_paths_df(results_path_dict, k_folds=5)
#df_inputs = get_input_paths_df(input_base_path)
df_inputs = get_input_paths_df(input_base_path,k_folds = 5,eval_test=True)
df = pd.merge(df_latent, df_inputs, on=["Split", "Type"], how="left")
# df["recon_prefix"] = [recon_path.split("/")[-1].split(".npy")[0] for recon_path in df["ReconPath"]]
print("Reading paths,\ndf paths:", df.head(5))

# 2. Define variables necessary to load data and  train model
batch_col = model_params_dict['batch_col']
bio_col = model_params_dict['bio_col']

# Define the One Hot encoded (OHE) order for donor and celltype categories
# get metadata before splits
metadata_all = pd.read_csv(glob.glob(data_path+"/*meta.csv")[0])
metadata_all[bio_col] = metadata_all[bio_col].astype('category')
metadata_all[batch_col] = metadata_all[batch_col].astype('category')
print("n batches",len(np.unique(metadata_all[batch_col]).tolist()))

# Define the One Hot encoded (OHE) order for donor and celltype categories
# Batch categories
batch_col_categories = np.unique(metadata_all[batch_col]).tolist()
print("check ordered batches: ",batch_col_categories)

# Bio col categories
bio_col_categories = np.unique(metadata_all[bio_col]).tolist()
print("bio categories: ",bio_col_categories)


# 3. Update the config for the model
load_latent_spaces_dict['batch_col_categories'] = batch_col_categories
load_latent_spaces_dict['bio_col_categories'] = bio_col_categories
# Update model
LatentClassifier_config['Model'] =  MixedEffectsModel
# Update latent path dict
latent_path_dict = create_latent_dict_from_df(df_latent)
load_latent_spaces_dict["latent_path_dict"] = latent_path_dict


# 4. Run the classifier for all folds latent space
folds_list = list(range(1, 6))  # List of folds from 1 to 5
all_folds_metrics_df = pd.DataFrame()  # Initialize an empty DataFrame

for fold in folds_list:
    print("fold",fold)
    # Update fold
    load_latent_spaces_dict["fold"] = fold
    # Get model manager
    model_manager = ModelManager(model_params_dict,
                                base_paths_dict,
                                run_name, 
                                save_model=LatentClassifier_config["save_model"],
                                use_kfolds=True,
                                kfold=load_latent_spaces_dict["fold"])
    # Update LatentClassifier config
    load_latent_spaces_dict["model_params"] = model_manager.params

    pipeline_LatentClassifier_config = {**load_latent_spaces_dict ,**LatentClassifier_config}
    results = run_model_pipeline_LatentClassifier_v2_PCA(**pipeline_LatentClassifier_config)
    results["metrics"]["fold"]=fold

    # Append the results["metrics"] DataFrame to all_folds_metrics_df
    all_folds_metrics_df = pd.concat([all_folds_metrics_df, results["metrics"]], ignore_index=True)
    
    # Clear any unused memory
    gc.collect()

# Save all folds metrics results
# Use os.path.join to construct the file path
output_path = os.path.join(load_latent_spaces_dict["model_params"].latent_path_main, "metrics_allfolds.csv")
all_folds_metrics_df.to_csv(output_path)
print("\nall_folds_metrics_df:",all_folds_metrics_df)


# Calculate 95%CI
results_df = calculate_metrics_with_ci(all_folds_metrics_df)
# Save 95%CI
output_path = os.path.join(load_latent_spaces_dict["model_params"].latent_path_main, "metrics_allfolds_95CI.csv")
results_df.to_csv(output_path)
print("\nresults_df:",results_df)


############## save config.py file
destination_path = os.path.join(saved_models_base, run_name, 'model_config.py')
print("\nCopying config.py file to:", destination_path)
# Copy the file
shutil.copy(source_file, destination_path)
