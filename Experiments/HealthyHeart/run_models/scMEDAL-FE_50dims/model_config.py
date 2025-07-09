import sys
# import paths
sys.path.append("../../")
from paths_config import data_base_path,scenario_id,outputs_path 
import os
import tensorflow as tf

from scMEDAL.utils.model_train_utils import generate_run_name

# --------------------------------------------------------------------------------------
# Compilation parameters
# --------------------------------------------------------------------------------------
compile_dict = {
    "loss_recon": tf.keras.losses.MeanSquaredError(),    # Reconstruction loss
    "loss_multiclass": tf.keras.losses.CategoricalCrossentropy(),  # Classification loss
    "metric_multiclass": tf.keras.metrics.CategoricalAccuracy(name='acc'),
    "opt_autoencoder": tf.keras.optimizers.Adam(lr=0.0001),  # Optimizer for the autoencoder component
    "opt_adversary": tf.keras.optimizers.Adam(lr=0.0001),    # Optimizer for the adversary component
    "loss_gen_weight": 1,  
    # "loss_recon_weight": 1800,
    "loss_recon_weight": 5400,  # 1800 * 3; Adjusted reconstruction loss weight
    # "loss_recon_weight": 0,   # For testing if we get max (4.99) - commented out
    "loss_class_weight": 1
}


# --------------------------------------------------------------------------------------
# Model building parameters
# --------------------------------------------------------------------------------------
build_model_dict = {
    "n_latent_dims": 50,          # Number of latent dimensions
    # "layer_units": [10],
    "layer_units": [512, 132],   # Encoder/Decoder layer units
    "n_clusters": 147,           # Number of clusters (batches)
    # "layer_units_latent_classifier": [2], # Not needed for ae_da
    # "n_pred": 13,               # Number of cell types (not needed for ae_da)
    "get_pred": False,           # Set True if using a celltype classification loss
    # "last_activation": "sigmoid",
    "last_activation": "linear", # Last activation of the decoder
    "use_batch_norm": True,      # Use batch normalization in encoder
    "name": "ae_da"              # Model type/name
}


# --------------------------------------------------------------------------------------
# Data loading parameters
# --------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,   # Set True to load test data
    "use_z": True,       # If the model requires a design matrix Z
    "scaling": "min_max" # Input data scaling: "min_max" or "z_scores"
}


# --------------------------------------------------------------------------------------
# Training parameters
# --------------------------------------------------------------------------------------
train_model_dict = {
    "batch_size": 512,
    "epochs": 500,            # For testing; for full experiments use larger epochs (e.g., 500)
    # "epochs": 500,
    "monitor_metric": 'val_total_loss',
    "patience": 30,
    "stop_criteria": "early_stopping",
    "compute_latents_callback": False,
    "sample_size": 10000,  # Used in clustering scores callback
    "model_type": "ae_da"
}


# --------------------------------------------------------------------------------------
# Score computation parameters (latent space analysis)
# --------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "scMEDAL-FE_latent_50", # Depends on the model used
    "get_pca": False,
    "n_components": 50,
    "get_baseline": False
}


# --------------------------------------------------------------------------------------
# Experimental design parameters
# --------------------------------------------------------------------------------------
expt_design_dict = {
    'batch_col': 'batch',       # Name of the batch column
    'bio_col': 'celltype',      # Biological condition column
    'donor_col': 'DonorID'      # Donor ID column (optional, useful for plotting)
}


# --------------------------------------------------------------------------------------
# Combine all dictionaries into a single dictionary
# --------------------------------------------------------------------------------------
model_params_dict = {
    **compile_dict,
    **build_model_dict,
    **load_data_dict,
    **train_model_dict,
    **get_scores_dict,
    **expt_design_dict
}


# --------------------------------------------------------------------------------------
# Define plotting parameters
# --------------------------------------------------------------------------------------
plot_params = {
    "shape_col": "celltype",
    "color_col": "donor",
    "markers": [
        "x", "+", "<", "h", "s", ".", 'o', 's', '^', '*', '1', '8', 'p', 'P', 'D', '|',
        0, ',', 'd', 2
    ],
    "showplot": False,
    "save_fig": True,
    "outpath": None  # Will be updated by ModelManager
}


# --------------------------------------------------------------------------------------
# Define base paths for input data
# --------------------------------------------------------------------------------------
data_path = os.path.join(data_base_path, scenario_id)
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')
print(f"Parent folder: {input_base_path}")


# --------------------------------------------------------------------------------------
# Define paths for experiment outputs
# --------------------------------------------------------------------------------------
folder_name = scenario_id
model_name = "scMEDAL-FE_50dims"

# Base directories for saved models, figures, and latent spaces
saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)


# --------------------------------------------------------------------------------------
# Generate a run name for the experiment
# --------------------------------------------------------------------------------------
constant_keys = [
    "compute_latents_callback", 'n_components', 'batch_col', 'bio_col', 'donor_col',
    "loss_recon", "loss_multiclass", "metric_multiclass", "opt_autoencoder",
    "opt_adversary", "layer_units_latent_classifier", "n_pred", "n_clusters", "name",
    "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z',
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred', "eval_test"
]

# run_name = generate_run_name(model_params_dict, constant_keys, name='run_HPO')
run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
print("run_name:", run_name)


# --------------------------------------------------------------------------------------
# Define dictionary of base paths
# --------------------------------------------------------------------------------------
base_paths_dict = {
    "models": saved_models_base,
    "figures": figures_base,
    "latent": latent_space_base
}


# --------------------------------------------------------------------------------------
# Set whether to save the model
# --------------------------------------------------------------------------------------
save_model = True
print("save_model set to:", save_model)

# You may initialize ModelManager here if needed:
# model_manager = ModelManager(model_params_dict=model_params_dict, base_paths_dict=base_paths_dict, run_name=run_name, save_model=save_model)
# model_manager.update_params({'new_param': 'new_value'})  # Update parameters if needed


# --------------------------------------------------------------------------------------
# Get the source path of the current configuration file
# --------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)




