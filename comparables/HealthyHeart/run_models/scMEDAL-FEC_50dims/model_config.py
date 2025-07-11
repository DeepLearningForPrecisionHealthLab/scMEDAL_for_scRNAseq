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
    "loss_recon": tf.keras.losses.MeanSquaredError(),          # Reconstruction loss
    "loss_multiclass": tf.keras.losses.CategoricalCrossentropy(), # Classification loss
    "metric_multiclass": tf.keras.metrics.CategoricalAccuracy(name='acc'),
    "opt_autoencoder": tf.keras.optimizers.Adam(lr=0.0001),    # Optimizer for autoencoder
    "opt_adversary": tf.keras.optimizers.Adam(lr=0.0001),      # Optimizer for adversary
    "loss_gen_weight": 1,    # Weight for generator loss
    #"loss_recon_weight": 9450,  # Weight for reconstruction loss
    #"loss_recon_weight":4000,
    "loss_recon_weight":2000,

    "loss_class_weight": 1   # Weight for classification loss
}

# --------------------------------------------------------------------------------------
# Model building parameters
# --------------------------------------------------------------------------------------
build_model_dict = {
    "n_latent_dims": 50,
    "layer_units": [512, 132],
    "n_clusters": 147,                   # Number of clusters (batches)
    "layer_units_latent_classifier": [2],
    "n_pred": 13,                        # Number of cell types
    "get_pred": True,                    # If True, include celltype classification loss (scMEDAL-FEC)
    # "last_activation": "sigmoid",
    "last_activation": "linear",         # Determines reconstructed output activation
    "use_batch_norm": True,              # Use batch normalization in encoder
    "name": "ae_da"                      # Model name/type
}

# --------------------------------------------------------------------------------------
# Data loading parameters
# --------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,   # Load test data
    "use_z": True,       # If model requires a design matrix Z
    "scaling": "min_max" # Input scaling: "min_max" or "z_scores"
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
    "sample_size": 10000,  # Used in clustering scores callbacks
    "model_type": "ae_da"
}

# --------------------------------------------------------------------------------------
# Score computation parameters
# --------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "scMEDAL-FEC_latent_50", # Modify depending on the model
    "get_pca": False,
    "n_components": 50,
    "get_baseline": False
}

# --------------------------------------------------------------------------------------
# Experimental design parameters
# --------------------------------------------------------------------------------------
expt_design_dict = {
    'batch_col': 'batch',        # Batch column name
    'bio_col': 'celltype',       # Biological condition column name
    'donor_col': 'DonorID'       # Donor ID column (optional, useful for plotting)
}

# --------------------------------------------------------------------------------------
# Combine all dictionaries into a single dictionary (model_params_dict)
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
# Plotting parameters (will be updated by ModelManager as needed)
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
    "outpath": None
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
print("Outputs saved to:", outputs_path)

folder_name = scenario_id
model_name = "scMEDAL-FEC_50dims"

# Base directories for saved models, figures, and latent spaces
saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

# --------------------------------------------------------------------------------------
# Generate a run name
# --------------------------------------------------------------------------------------
constant_keys = [
    "compute_latents_callback", "n_components", 'batch_col', 'bio_col', 'donor_col',
    "loss_recon", "loss_multiclass", "metric_multiclass", "opt_autoencoder", "opt_adversary",
    "layer_units_latent_classifier", "n_pred", "n_clusters", "name", "monitor_metric", 
    "stop_criteria", "get_pca", "get_baseline", 'use_z', 'encoder_latent_name', 
    'sigmoid_eval_test', 'last_activation', 'get_pred', "eval_test"
]

# run_name = generate_run_name(model_params_dict, constant_keys, name='run_HPO')
run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
print("run_name:", run_name)

# --------------------------------------------------------------------------------------
# Define base paths dictionary
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


# --------------------------------------------------------------------------------------
# Get the source path of this configuration file
# --------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)
