import sys
import os
import glob
import gc
import shutil
import numpy as np
import pandas as pd

from scMEDAL.utils.compare_results_utils import (
    get_input_paths_df,
    get_latent_paths_df,
    create_latent_dict_from_df
)

# Add the path to the configuration file
sys.path.append("../../../../")
from paths_config import results_path_dict, input_base_path

from scMEDAL.utils.model_train_utils import (
    ModelManager,
    run_model_pipeline_LatentClassifier_v2_PCA,
    calculate_metrics_with_ci
)
from model_config import (
    data_path, model_params_dict, base_paths_dict, run_name,
    LatentClassifier_config, load_latent_spaces_dict,
    saved_models_base, source_file
)
from scMEDAL.models.scMEDAL import MixedEffectsModel

# --------------------------------------------------------------------------------------
# 1. Get Input Paths and Latent Paths
# --------------------------------------------------------------------------------------
df_latent = get_latent_paths_df(results_path_dict, k_folds=5)
df_inputs = get_input_paths_df(input_base_path, k_folds=5, eval_test=True)

# Merge latent and input paths
df = pd.merge(df_latent, df_inputs, on=["Split", "Type"], how="left")
print("Reading paths,\ndf paths:\n", df.head(5))

# --------------------------------------------------------------------------------------
# 2. Define Variables Necessary to Load Data and Train Model
# --------------------------------------------------------------------------------------
batch_col = model_params_dict['batch_col']
bio_col = model_params_dict['bio_col']

# Load metadata
metadata_all = pd.read_csv(glob.glob(os.path.join(data_path, "*meta.csv"))[0])
metadata_all[bio_col] = metadata_all[bio_col].astype('category')
metadata_all[batch_col] = metadata_all[batch_col].astype('category')

print("n batches:", len(np.unique(metadata_all[batch_col]).tolist()))

# Define categories for batch and bio columns
batch_col_categories = np.unique(metadata_all[batch_col]).tolist()
print("check ordered batches:", batch_col_categories)

bio_col_categories = np.unique(metadata_all[bio_col]).tolist()
print("bio categories:", bio_col_categories)

# --------------------------------------------------------------------------------------
# 3. Update the Config for the Model
# --------------------------------------------------------------------------------------
load_latent_spaces_dict['batch_col_categories'] = batch_col_categories
load_latent_spaces_dict['bio_col_categories'] = bio_col_categories

# Update model
LatentClassifier_config['Model'] = MixedEffectsModel

# Create latent path dictionary
latent_path_dict = create_latent_dict_from_df(df_latent)
load_latent_spaces_dict["latent_path_dict"] = latent_path_dict

# --------------------------------------------------------------------------------------
# 4. Run the Classifier for All Folds Latent Space
# --------------------------------------------------------------------------------------
folds_list = list(range(1, 6))  # Folds 1 to 5
all_folds_metrics_df = pd.DataFrame()

for fold in folds_list:
    print("fold", fold)
    load_latent_spaces_dict["fold"] = fold

    # Initialize model manager
    model_manager = ModelManager(
        model_params_dict,
        base_paths_dict,
        run_name,
        save_model=LatentClassifier_config["save_model"],
        use_kfolds=True,
        kfold=load_latent_spaces_dict["fold"]
    )

    # Update LatentClassifier config
    load_latent_spaces_dict["model_params"] = model_manager.params
    pipeline_LatentClassifier_config = {**load_latent_spaces_dict, **LatentClassifier_config}

    # Run pipeline
    results = run_model_pipeline_LatentClassifier_v2_PCA(**pipeline_LatentClassifier_config)
    results["metrics"]["fold"] = fold

    # Append metrics
    all_folds_metrics_df = pd.concat([all_folds_metrics_df, results["metrics"]], ignore_index=True)

    # Clear memory
    gc.collect()

# --------------------------------------------------------------------------------------
# 5. Save All Folds Metrics Results
# --------------------------------------------------------------------------------------
output_path = os.path.join(
    load_latent_spaces_dict["model_params"].latent_path_main,
    "metrics_allfolds.csv"
)
all_folds_metrics_df.to_csv(output_path)
print("\nall_folds_metrics_df:", all_folds_metrics_df)

# --------------------------------------------------------------------------------------
# 6. Calculate and Save 95% CI
# --------------------------------------------------------------------------------------
results_df = calculate_metrics_with_ci(all_folds_metrics_df)
output_path = os.path.join(
    load_latent_spaces_dict["model_params"].latent_path_main,
    "metrics_allfolds_95CI.csv"
)
results_df.to_csv(output_path)
print("\nresults_df:", results_df)

# --------------------------------------------------------------------------------------
# 7. Save Configuration File
# --------------------------------------------------------------------------------------
destination_path = os.path.join(saved_models_base, run_name, 'model_config.py')
print("\nCopying config.py file to:", destination_path)
shutil.copy(source_file, destination_path)
