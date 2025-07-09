import sys
import os

# Add paths for custom imports
sys.path.append("../../")
from paths_config import data_base_path, scenario_id, outputs_path
from scMEDAL.utils.model_train_utils import generate_run_name
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as mse_loss
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
# from tensorflow.keras.metrics import AUC as auc_metric

# --------------------------------------------------------------------------------------
# Compilation parameters
# --------------------------------------------------------------------------------------
compile_dict = {
    "optimizer": Adam(lr=0.0001),  # Learning rate for the optimizer
    "loss": [mse_loss(name='mean_squared_error')],  # Loss function
    "loss_weights": 1,  # Weighting for loss components
    "metrics": [[mse_metric()]]  # Metrics for evaluation
}

# --------------------------------------------------------------------------------------
# Model building parameters
# --------------------------------------------------------------------------------------
build_model_dict = {
    "n_latent_dims": 2,          # Number of latent dimensions
    "layer_units": [512, 132],   # Units for encoder/decoder layers
    "last_activation": "linear",  # Last activation function of the decoder
    "use_batch_norm": True,      # Apply batch normalization in the encoder
    "name": "AE"                 # Model name
}

# --------------------------------------------------------------------------------------
# Data loading parameters
# --------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,          # Load test data for evaluation
    "use_z": False,             # Use design matrix Z if required by the model
    "get_pred": False,          # Enable predictions if using a classifier
    "scaling": "min_max"        # Scaling method: "min_max" or "z_scores"
}

# --------------------------------------------------------------------------------------
# Training parameters
# --------------------------------------------------------------------------------------
train_model_dict = {
    "batch_size": 512,          # Batch size for training
    "epochs": 2,            # For testing; for full experiments use larger epochs (e.g., 500)
    # "epochs": 500,
    "monitor_metric": 'val_loss',  # Metric to monitor during training
    "patience": 30,             # Early stopping patience
    "stop_criteria": "early_stopping",  # Criteria for stopping training
    "compute_latents_callback": False,  # Compute latent space during callbacks
    "sample_size": 10000,       # Sample size for clustering score callbacks
    "model_type": "ae"          # Model type: 'ae' for autoencoder
}

# --------------------------------------------------------------------------------------
# Latent space and score computation parameters
# --------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "AE_latent_2",  # Latent space name for the encoder
    "get_pca": True,                      # Enable PCA computation
    "n_components": 2,                    # Number of PCA components
    "get_baseline": False                 # Compute baseline (time-intensive)
}

# --------------------------------------------------------------------------------------
# Experimental design parameters
# --------------------------------------------------------------------------------------
expt_design_dict = {
    'batch_col': 'batch',         # Name of the batch column
    'bio_col': 'celltype'         # Biological condition (e.g., cell type)
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
# Plotting parameters
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
    "outpath": None  # Will be set by the ModelManager
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
model_name = "AE"

saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

# --------------------------------------------------------------------------------------
# Generate run name for the experiment
# --------------------------------------------------------------------------------------
constant_keys = [
    'batch_col', 'bio_col', 'donor_col', "layer_units_latent_classifier", "name",
    "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z',
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred',
    "eval_test", "optimizer", "loss", "loss_weights", "metrics"
]

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

# --------------------------------------------------------------------------------------
# Get the source path of this configuration file
# --------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)
