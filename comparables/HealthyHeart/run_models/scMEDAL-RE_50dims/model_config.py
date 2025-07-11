import os
import sys
# import paths
sys.path.append("../../")
from paths_config import data_base_path,scenario_id,outputs_path 
import tensorflow as tf
from scMEDAL.utils.model_train_utils import generate_run_name

# --------------------------------------------------------------------------------------
# Compilation parameters
# --------------------------------------------------------------------------------------
compile_dict = {
    "loss_recon": tf.keras.losses.MeanSquaredError(),       # Reconstruction loss
    "loss_multiclass": tf.keras.losses.CategoricalCrossentropy(),  # Multiclass classification loss
    "metric_multiclass": tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
    "optimizer": tf.keras.optimizers.Adam(lr=0.0001),
    "loss_recon_weight": 110.0,
    # "loss_class_weight": 0.01,  # Not relevant if "get_pred" is False
    "loss_latent_cluster_weight": 0.1
    # "loss_recon_cluster_weight": 0.001 # Not relevant if "get_recon_cluster" is False
}


# --------------------------------------------------------------------------------------
# Model building parameters
# --------------------------------------------------------------------------------------
build_model_dict = {
    "n_latent_dims": 50,            # Initial latent dimensions
    "n_clusters": 147,             # Number of clusters (batches)
    "layer_units": [512, 132],
    "layer_units_classifier": [5],
    "n_pred": 13,                  # Number of cell types
    # "last_activation": "sigmoid",
    "last_activation": "linear",   # Determines reconstructed output values
    "post_loc_init_scale": 0.1,
    "prior_scale": 0.25,
    "kl_weight": 1e-5,
    "get_pred": False,             # Set True if including classification loss
    "get_recon_cluster": False,    # Set True if including reconstruction cluster loss
    "name": "ae_re"                # Model name/type
}


# --------------------------------------------------------------------------------------
# Data loading parameters
# --------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,    # Load test data for evaluation
    "use_z": True,        # If model requires a design matrix Z
    "scaling": "min_max"  # Input scaling: "min_max" or "z_scores"
}


# --------------------------------------------------------------------------------------
# Training parameters
# --------------------------------------------------------------------------------------
train_model_dict = {
    "batch_size": 512,
    "epochs": 500,          # For testing; use higher epochs for full experiments
    # "epochs": 500,
    "monitor_metric": 'val_total_loss',
    "patience": 30,
    "stop_criteria": "early_stopping",
    "compute_latents_callback": False,
    "sample_size": 10000, # Used in clustering scores callback
    "model_type": "ae_re"
}


# --------------------------------------------------------------------------------------
# Score computation parameters
# --------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "scMEDAL-RE_latent_50",  # Adjust based on model
    "get_pca": False,
    "n_components": 50,
    "get_baseline": False,
    "get_cf_batch": True  # Additional parameter, e.g., for batch correction factors
}


# --------------------------------------------------------------------------------------
# Experimental design parameters
# --------------------------------------------------------------------------------------
expt_design_dict = {
    'batch_col': 'batch',     # Batch column name
    'bio_col': 'celltype',    # Biological condition column
    'donor_col': 'DonorID',   # Donor ID column (optional)
    'tissue_col': 'TissueDetail'  # Tissue detail column (optional)
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
# Plotting parameters (updated later by ModelManager)
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
model_name = "scMEDAL-RE_50dims"

# Set whether to save the model
save_model = True
print("save_model set to:", save_model)


# --------------------------------------------------------------------------------------
# RAY_RUN control (if needed)
# --------------------------------------------------------------------------------------
RAY_RUN = False
if not RAY_RUN:
    # Folder structure setup
    saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
    figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
    latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

    # Generate the run name
    constant_keys = [
        "get_cf_batch", "compute_latents_callback", "n_components", "loss_recon",
        "loss_multiclass", "metric_multiclass", "optimizer", 'model_type', 'tissue_col',
        'batch_col', 'bio_col', 'donor_col', "layer_units_classifier", "get_recon_cluster",
        "prior_scale", "post_loc_init_scale", "layer_units_latent_classifier", "n_pred",
        "n_clusters", "name", "monitor_metric", "stop_criteria", "get_pca", "get_baseline",
        'use_z', 'encoder_latent_name', 'sigmoid_eval_test', 'last_activation',
        'get_pred', "eval_test"
    ]

    run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
    print("run_name:", run_name)

    # Define dictionary of base paths
    base_paths_dict = {
        "models": saved_models_base,
        "figures": figures_base,
        "latent": latent_space_base
}


# --------------------------------------------------------------------------------------
# Get the source path of the current configuration file
# --------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)
