

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import glob

PROJECT_ROOT = Path.cwd().resolve().parents[2]  # two levels up from AML dir (scMEDAL_for_scRNAseq)
sys.path.insert(0, str(PROJECT_ROOT))

from scMEDAL_for_scRNAseq.utils.defaults import AML_PATHS_CONFIG


data_base_path = AML_PATHS_CONFIG.get("data_base_path")
scenario_id = AML_PATHS_CONFIG.get("scenario_id")
outputs_path = AML_PATHS_CONFIG.get("outputs_path")

from scMEDAL_for_scRNAseq.utils.model_train_utils import generate_run_name

# Detect TensorFlow version
import tensorflow as tf
tf_version = tf.__version__
print("tf_version",tf_version)


from scMEDAL_for_scRNAseq.comparables.scanorama import run_scanorama_across_folds


compile_dict={}

build_model_dict = {
    "n_latent_dims": 50,   # Number of latent dimensions in the model

    #"n_hidden": 132,       # for scNORAMA is going to be the number of pca components

}



#--------------------------------------------------------------------------------------
# Data loading parameters
#--------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,      # Set to True if test data should be loaded for evaluation
    "use_z": False,         # If the model requires a design matrix Z, set True. AE_conv does not need it.
    "get_pred": False,      # If predictions are needed. Useful when using classifiers, not for a simple AE.
    "scaling": "min_max"    # Input scaling: "min_max" or "z_scores"
}


#--------------------------------------------------------------------------------------
# Training parameters
#--------------------------------------------------------------------------------------
train_model_dict = {
#    "batch_size": 512,      # Training batch size
    # "epochs": 500,            # For testing; for full experiments use larger epochs (e.g., 500)
    #"epochs": 500,
    #"epochs": 500,
    # "monitor_metric": 'val_loss',  
    # "patience": 30,         # Early stopping patience
    # "stop_criteria": "early_stopping",
    # "compute_latents_callback": False,
    "sample_size": 10000,   # Used in clustering score callbacks
    "model_type": "ae"      # Type of model: 'ae' for autoencoder
}


#--------------------------------------------------------------------------------------
# Latent space and score computation parameters
#--------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "scanorama_latent_50", # Modify depending on the model used
    # "get_pca": True,
    # "n_components": 2,
    "get_baseline": False   # If True, compute baseline, but it's time-consuming
}


#--------------------------------------------------------------------------------------
# Experimental design parameters
#--------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
expt_design_dict = {
    'batch_col': 'batch',         # Name of the batch column
    'bio_col': 'celltype'         # Biological condition (e.g., cell type)
}


#--------------------------------------------------------------------------------------
# Combine all dictionaries into a single dictionary
#--------------------------------------------------------------------------------------
model_params_dict = {
#    **compile_dict,
    **build_model_dict,
    **load_data_dict,
    **train_model_dict,
    **get_scores_dict,
    **expt_design_dict
}


#--------------------------------------------------------------------------------------
# Define common plotting parameters
#--------------------------------------------------------------------------------------
plot_params = {
    "shape_col": "celltype",
    "color_col": "batch",
    "markers": [
        "x", "+", "<", "h", "s", ".", 'o', 's', '^', '*', '1', '8', 'p', 'P', 'D', '|',
        0, ',', 'd', 2
    ],
    "showplot": False,
    "save_fig": True,
    "outpath": None  # Will be set by the ModelManager
}



#--------------------------------------------------------------------------------------
# Define base paths for input data
#--------------------------------------------------------------------------------------
data_path = os.path.join(data_base_path, scenario_id)
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')
print(f"Parent folder: {input_base_path}")


#--------------------------------------------------------------------------------------
# Define paths for experiment outputs
#--------------------------------------------------------------------------------------
print("Outputs saved to:", outputs_path)

folder_name = scenario_id
model_name = "scanorama_50dims"

# Define base paths for saving models, figures, and latent spaces
saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)


#--------------------------------------------------------------------------------------
# Generate run name for the experiment
#--------------------------------------------------------------------------------------
constant_keys = [
    'batch_col', 'bio_col', 'donor_col', "layer_units_latent_classifier", "name",
    "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z',
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred',
    "eval_test", "optimizer", "loss", "loss_weights", "metrics"
]

# Generate a name for this run, excluding constant_keys from the naming scheme
run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
print("run_name:", run_name)


#--------------------------------------------------------------------------------------
# Define dictionary of base paths
#--------------------------------------------------------------------------------------
base_paths_dict = {
    "models": saved_models_base,
    "figures": figures_base,
    "latent": latent_space_base
}


#--------------------------------------------------------------------------------------
# Set whether to save the model
#--------------------------------------------------------------------------------------
save_model = True
print("save_model set to:", save_model)


#--------------------------------------------------------------------------------------
# Get the source path of this config file
#--------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)




# ======================================================
# 4. Main execution block
# ======================================================

# -------------------------------------
# 4.1 CV & metadata preparation
# -------------------------------------
folds_list = list(range(1, 6))  # 5‑fold CV
# folds_list = [1,2]

batch_col = model_params_dict["batch_col"]
bio_col   = model_params_dict["bio_col"]
#donor_col = model_params_dict["donor_col"]  # optional in plotting only

shape_color_dict = {
    f"{bio_col}-{bio_col}":   {"shape_col": bio_col,   "color_col": bio_col},
    f"{batch_col}-{batch_col}": {"shape_col": batch_col, "color_col": batch_col},
    }

metadata_all = pd.read_csv(glob.glob(data_path + "/*meta.csv")[0])
# Convert cell type and batch columns to categorical
metadata_all['celltype'] = metadata_all['celltype'].astype('category')
metadata_all['batch'] = metadata_all["batch"].astype('category')
seen_donor_ids = np.unique(metadata_all[batch_col]).tolist()
celltype_ids   = np.unique(metadata_all[bio_col]).tolist()

print("Number of batches:", len(seen_donor_ids))

# -------------------------------------
# 4.2 Run CV
# -------------------------------------
cv_results = run_scanorama_across_folds(
    input_base_path        = input_base_path,
    out_base_paths_dict    = base_paths_dict,
    folds_list             = folds_list,
    run_name               = run_name,
    model_params_dict      = model_params_dict,
    build_model_dict       = build_model_dict,
    compile_dict           = compile_dict,
    save_model             = save_model,
    batch_col              = batch_col,
    bio_col                = bio_col,
    batch_col_categories   = seen_donor_ids,
    bio_col_categories     = celltype_ids,
    model_type             = "scanorama",
    issparse               = True,
    load_dense             = True,
    shape_color_dict       = shape_color_dict,
    return_scores_temp     = True,
    sample_size = 10000,
    n_batch2plot = None,
    seed = 5,
    plot_params = plot_params
    )




# --------------------------------------------------------------------------------------
# 3. Save the configuration file
# --------------------------------------------------------------------------------------

# Define destination path for the configuration file
destination_path = os.path.join(saved_models_base, run_name, 'model_config.py')

print("\nCopying config.py file to:", destination_path)

# Copy the configuration file to the destination
shutil.copy(source_file, destination_path)





