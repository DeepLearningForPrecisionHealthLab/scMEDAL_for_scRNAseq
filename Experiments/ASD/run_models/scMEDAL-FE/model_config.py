import sys
import os
import tensorflow as tf

# Add paths for custom imports
sys.path.append("../../")
from paths_config import data_base_path, scenario_id, outputs_path
from scMEDAL.utils.model_train_utils import generate_run_name

# --------------------------------------------------------------------------------------
# 1. Compilation Parameters
# --------------------------------------------------------------------------------------
compile_dict = {
    "loss_recon": tf.keras.losses.MeanSquaredError(),  # Reconstruction loss
    "loss_multiclass": tf.keras.losses.CategoricalCrossentropy(),  # Classification loss
    "metric_multiclass": tf.keras.metrics.CategoricalAccuracy(name='acc'),  # Accuracy metric
    "opt_autoencoder": tf.keras.optimizers.Adam(lr=0.0001),  # Optimizer for AEC
    "opt_adversary": tf.keras.optimizers.Adam(lr=0.0001),  # Optimizer for Adversary
    "loss_gen_weight": 1,  # General loss weight
    "loss_recon_weight": 3000,  # Reconstruction loss weight
    "loss_class_weight": 1  # Classification loss weight
}

# --------------------------------------------------------------------------------------
# 2. Model Building Parameters
# --------------------------------------------------------------------------------------
build_model_dict = {
    "n_latent_dims": 2,          # Number of latent dimensions
    "layer_units": [512, 132],   # Encoder/decoder layer units
    "n_clusters": 31,            # Number of clusters (batches)
    "get_pred": False,           # Disable celltype classification loss
    "last_activation": "linear",  # Decoder's last activation function
    "use_batch_norm": True,      # Enable batch normalization in encoder
    "name": "ae_da"              # Model name
}

# --------------------------------------------------------------------------------------
# 3. Data Loading Parameters
# --------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,          # Enable test data evaluation
    "use_z": True,              # Use Z design matrix
    "scaling": "min_max"        # Input scaling method: "min_max" or "z_scores"
}

# --------------------------------------------------------------------------------------
# 4. Training Parameters
# --------------------------------------------------------------------------------------
train_model_dict = {
    "batch_size": 512,          # Batch size for training
    "epochs": 2,            # For testing; for full experiments use larger epochs (e.g., 500)
    # "epochs": 500,
    "monitor_metric": 'val_total_loss',  # Metric to monitor during training
    "patience": 30,             # Early stopping patience
    "stop_criteria": "early_stopping",  # Early stopping criteria
    "compute_latents_callback": False,  # Disable latent computation in callbacks
    "sample_size": 10000,       # Sample size for clustering score callbacks
    "model_type": "ae_da"       # Model type
}

# --------------------------------------------------------------------------------------
# 5. Latent Space and Score Computation Parameters
# --------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "scMEDAL-FE_latent_2",  # Encoder latent space name
    "get_pca": False,                         # Disable PCA computation
    "n_components": 2,                        # Number of PCA components
    "get_baseline": False                     # Disable baseline computation
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
    "outpath": None  # Will be updated by ModelManager
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
model_name = "scMEDAL-FE"

saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

# --------------------------------------------------------------------------------------
# 11. Generate Run Name
# --------------------------------------------------------------------------------------
constant_keys = [
    "compute_latents_callback", 'n_components', 'batch_col', 'bio_col', 'donor_col',
    "loss_recon", "loss_multiclass", "metric_multiclass", "opt_autoencoder", 
    "opt_adversary", "layer_units_latent_classifier", "n_pred", "n_clusters", "name", 
    "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z', 
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred', "eval_test"
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
# 14. Get Source Path of Configuration File
# --------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)
