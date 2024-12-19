
import sys
# import paths
sys.path.append("../../")
from paths_config import data_base_path,scenario_id,outputs_path 
import os
from scMEDAL.utils.model_train_utils import generate_run_name
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as mse_loss
# from tensorflow.keras.losses import BinaryCrossentropy as bce_loss
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
# from tensorflow.keras.metrics import AUC as auc_metric
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import CategoricalAccuracy as cce_metric

# --------------------------------------------------------------------------------------
# Compilation parameters
# --------------------------------------------------------------------------------------
compile_dict = {
    "optimizer": Adam(lr=0.0001),
    "loss": {
        'reconstruction_output': mse_loss(name='mse'),
        'classification_output': cce_loss(name='cce')
    },
    "loss_weights": {
        'reconstruction_output': 100,
        'classification_output': 0.1
    },
    "metrics": {
        'reconstruction_output': [mse_metric(name="mse_metric")],
        'classification_output': [cce_metric(name='cce_metric')]
    }
}

# --------------------------------------------------------------------------------------
# Model building parameters
# --------------------------------------------------------------------------------------
build_model_dict = {
    "n_latent_dims": 2,        # Number of latent dimensions
    # "layer_units": [10],
    "layer_units": [512, 132],
    "layer_units_latent_classifier": [2],
    "n_pred": 13,              # Number of celltypes
    # "last_activation": "sigmoid",
    "last_activation": "linear",  # Determines output reconstruction range
    "use_batch_norm": True,       # Use batch normalization in encoder
    "name": "AEC"                 # Model name/type
}

# --------------------------------------------------------------------------------------
# Data loading parameters
# --------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,   # Load test data for evaluation
    "use_z": False,      # If the model requires a Z design matrix
    "get_pred": True,    # If predictions (classification) are required
    "scaling": "min_max" # Input scaling: "min_max" or "z_scores"
}

# --------------------------------------------------------------------------------------
# Training parameters
# --------------------------------------------------------------------------------------
train_model_dict = {
    # "batch_size": 60,
    "batch_size": 512,
    "epochs": 2,            # For testing; for full experiments use larger epochs (e.g., 500)
    # "epochs": 500,
    "monitor_metric": 'val_loss',
    "patience": 30,                # Early stopping patience
    "stop_criteria": "early_stopping",
    "compute_latents_callback": False,
    "sample_size": 10000,          # Used in clustering score callbacks
    "model_type": "aec"            # Model type for reference
}

# --------------------------------------------------------------------------------------
# Score computation parameters
# --------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "AEC_latent_2", # Latent space name depends on model
    "get_pca": True,
    "n_components": 2,
    "get_baseline": False             # Computing baseline can be time-consuming
}

# --------------------------------------------------------------------------------------
# Experimental design parameters
# --------------------------------------------------------------------------------------
expt_design_dict = {
    'batch_col': 'batch',     # Name of batch column
    'bio_col': 'celltype',    # Name of bio/condition column
    'donor_col': 'DonorID'    # Optional, useful for plotting
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
# Define plotting parameters (updated later by ModelManager)
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
model_name = "AEC"

# Directories for saved models, figures, and latent spaces
saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

# --------------------------------------------------------------------------------------
# Generate the run name
# --------------------------------------------------------------------------------------
constant_keys = [
    "n_components", 'batch_col', 'bio_col', 'donor_col', "layer_units_latent_classifier", 
    "name", "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z', 
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred', "eval_test", 
    "optimizer", "loss", "loss_weights", "metrics"
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

# ---------------------------------------------------------------------------------
# Get the source path of the configuration file
# --------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)
