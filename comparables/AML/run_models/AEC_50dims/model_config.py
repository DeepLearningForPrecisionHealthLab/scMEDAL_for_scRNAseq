import sys
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as mse_loss
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
from tensorflow.keras.metrics import CategoricalAccuracy as cce_metric
from scMEDAL.utils.model_train_utils import generate_run_name

# Add paths for custom imports
sys.path.append("../../")
from paths_config import data_base_path, scenario_id, outputs_path

# --------------------------------------------------------------------------------------
# 1. Compilation Parameters
# --------------------------------------------------------------------------------------
compile_dict = {
    "optimizer": Adam(lr=0.0001),  # Optimizer
    "loss": {
        'reconstruction_output': mse_loss(name='mse'),  # Reconstruction loss
        'classification_output': cce_loss(name='cce')   # Classification loss
    },
    "loss_weights": {
        'reconstruction_output': 100,   # Weight for reconstruction loss
        'classification_output': 0.1   # Weight for classification loss
    },
    "metrics": {
        'reconstruction_output': [mse_metric(name="mse_metric")],
        'classification_output': [cce_metric(name='cce_metric')]
    }
}

# --------------------------------------------------------------------------------------
# 2. Model Building Parameters
# --------------------------------------------------------------------------------------
build_model_dict = {
    "n_latent_dims": 50,          # Number of latent dimensions
    "layer_units": [512, 132],   # Encoder/decoder layer units
    "layer_units_latent_classifier": [2],  # Latent classifier layer units
    "n_pred": 21,                # Number of predictions (e.g., cell types)
    "last_activation": "linear",  # Decoder's last activation function
    "use_batch_norm": True,      # Use batch normalization in encoder
    "name": "AEC"                # Model name
}

# --------------------------------------------------------------------------------------
# 3. Data Loading Parameters
# --------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,          # Enable test data evaluation
    "use_z": False,             # Use Z design matrix if required
    "get_pred": True,           # Enable predictions
    "scaling": "min_max"        # Scaling method: "min_max" or "z_scores"
}

# --------------------------------------------------------------------------------------
# 4. Training Parameters
# --------------------------------------------------------------------------------------
train_model_dict = {
    "batch_size": 512,          # Batch size for training
    "epochs": 500,            # For testing; for full experiments use larger epochs (e.g., 500)
    # "epochs": 500,             # Number of training epochs
    "monitor_metric": 'val_loss',  # Metric to monitor during training
    "patience": 30,             # Early stopping patience
    "stop_criteria": "early_stopping",  # Early stopping criteria
    "compute_latents_callback": False,  # Compute latents during callbacks
    "sample_size": 10000,       # Sample size for clustering scores
    "model_type": "aec"         # Model type
}

# --------------------------------------------------------------------------------------
# 5. Latent Space and Score Computation Parameters
# --------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "AEC_latent_50",  # Encoder latent space name
    "get_pca": False,                      # Enable PCA computation
    "n_components": 50,                    # Number of PCA components
    "get_baseline": False                 # Disable baseline computation
}

# --------------------------------------------------------------------------------------
# 6. Experimental Design Parameters
# --------------------------------------------------------------------------------------
expt_design_dict = {
    'batch_col': 'batch',        # Name of the batch column
    'bio_col': 'celltype'        # Name of the biological condition column
}

# --------------------------------------------------------------------------------------
# 7. Combine All Parameters into a Single Dictionary
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
# 8. Plotting Parameters
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
    "outpath": None  # Will be updated later
}

# --------------------------------------------------------------------------------------
# 9. Define Data Paths
# --------------------------------------------------------------------------------------
data_path = os.path.join(data_base_path, scenario_id)
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')
print(f"Parent folder: {input_base_path}")

# --------------------------------------------------------------------------------------
# 10. Define Output Paths
# --------------------------------------------------------------------------------------
folder_name = scenario_id
model_name = "AEC_50dims"

saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

# --------------------------------------------------------------------------------------
# 11. Generate Run Name
# --------------------------------------------------------------------------------------
constant_keys = [
    "n_components", 'batch_col', 'bio_col', 'donor_col', "layer_units_latent_classifier",
    "name", "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z',
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred',
    "eval_test", "optimizer", "loss", "loss_weights", "metrics"
]

run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
print("Run name:", run_name)

# --------------------------------------------------------------------------------------
# 12. Define Base Paths Dictionary
# --------------------------------------------------------------------------------------
base_paths_dict = {
    "models": saved_models_base,
    "figures": figures_base,
    "latent": latent_space_base
}

# --------------------------------------------------------------------------------------
# 13. Set Save Model Flag
# --------------------------------------------------------------------------------------
save_model = True
print("Save model set to:", save_model)

# --------------------------------------------------------------------------------------
# 14. Get Source Path
# --------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)
