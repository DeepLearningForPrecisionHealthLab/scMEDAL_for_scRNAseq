import sys
import os
import tensorflow as tf

# Import configurations
sys.path.append("../../")
from paths_config import data_base_path, scenario_id, outputs_path
from scMEDAL.utils.model_train_utils import generate_run_name

# --------------------------------------------------------------------------------------
# 1. Compilation Parameters
# --------------------------------------------------------------------------------------
compile_dict = {
    "loss_recon": tf.keras.losses.MeanSquaredError(),  # Reconstruction loss
    "loss_multiclass": tf.keras.losses.CategoricalCrossentropy(),  # Multi-class loss
    "metric_multiclass": tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),  # Accuracy metric
    "optimizer": tf.keras.optimizers.Adam(lr=0.0001),
    "loss_recon_weight": 110.0,  # Weight for reconstruction loss
    "loss_latent_cluster_weight": 0.1,  # Weight for latent cluster loss
}

# --------------------------------------------------------------------------------------
# 2. Model Building Parameters
# --------------------------------------------------------------------------------------
build_model_dict = {
    "n_latent_dims": 50,  # Number of latent dimensions
    "n_clusters": 19,  # Number of clusters (batches)
    "layer_units": [512, 132],  # Encoder/decoder layer units
    "layer_units_classifier": [5],  # Classifier layer units
    "n_pred": 21,  # Number of predicted cell types
    "last_activation": "linear",  # Last activation function for decoder
    "post_loc_init_scale": 0.1,  # Initialization scale for posterior mean
    "prior_scale": 0.25,  # Scale for prior distribution
    "kl_weight": 1e-5,  # KL divergence weight
    "get_pred": False,  # Include classification output
    "get_recon_cluster": False,  # Include reconstruction cluster loss
    "name": "ae_re",  # Model name
}

# --------------------------------------------------------------------------------------
# 3. Data Loading Parameters
# --------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,  # Load test data
    "use_z": True,  # Use Z design matrix
    "scaling": "min_max",  # Input scaling method
}

# --------------------------------------------------------------------------------------
# 4. Training Parameters
# --------------------------------------------------------------------------------------
train_model_dict = {
    "batch_size": 512,  # Batch size
    #"epochs": 2,            # For testing; for full experiments use larger epochs (e.g., 500)
    "epochs": 500,             # Number of training epochs
    "monitor_metric": 'val_total_loss',  # Metric to monitor
    "patience": 30,  # Early stopping patience
    "stop_criteria": "early_stopping",  # Early stopping criteria
    "compute_latents_callback": False,  # Disable latent computation in callbacks
    "sample_size": 10000,  # Sample size for clustering scores
    "model_type": "ae_re",  # Model type
}

# --------------------------------------------------------------------------------------
# 5. Score Computation Parameters
# --------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "scMEDAL-RE_latent_50",  # Latent space name
    "get_pca": False,  # Disable PCA computation
    "n_components": 50,  # Number of PCA components
    "get_baseline": False,  # Disable baseline computation
    "get_cf_batch": True,  # Include batch-conditioned computation
}

# --------------------------------------------------------------------------------------
# 6. Experimental Design Parameters
# --------------------------------------------------------------------------------------
expt_design_dict = {
    "batch_col": "batch",  # Batch column name
    "bio_col": "celltype",  # Biological condition column name
}

# --------------------------------------------------------------------------------------
# 7. Combine All Parameters
# --------------------------------------------------------------------------------------
model_params_dict = {
    **compile_dict,
    **build_model_dict,
    **load_data_dict,
    **train_model_dict,
    **get_scores_dict,
    **expt_design_dict,
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
    "outpath": None,  # To be updated dynamically
}

# --------------------------------------------------------------------------------------
# 9. Data Paths
# --------------------------------------------------------------------------------------
data_path = os.path.join(data_base_path, scenario_id)
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')
print(f"Parent folder: {input_base_path}")
# --------------------------------------------------------------------------------------
# 10. Output Paths
# --------------------------------------------------------------------------------------
folder_name = scenario_id
model_name = "scMEDAL-RE_50dims"
saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

# --------------------------------------------------------------------------------------
# 11. Generate Run Name
# --------------------------------------------------------------------------------------
constant_keys = [
    "get_cf_batch", "compute_latents_callback", "n_components", "loss_recon", 
    "loss_multiclass", "metric_multiclass", "optimizer", "model_type", "tissue_col", 
    "batch_col", "bio_col", "donor_col", "layer_units_classifier", "get_recon_cluster", 
    "prior_scale", "post_loc_init_scale", "layer_units_latent_classifier", "n_pred", 
    "n_clusters", "name", "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 
    "use_z", "encoder_latent_name", "sigmoid_eval_test", "last_activation", "get_pred", 
    "eval_test",
]

run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
print(f"Run name: {run_name}")

base_paths_dict = {
    "models": saved_models_base,
    "figures": figures_base,
    "latent": latent_space_base,
}

# --------------------------------------------------------------------------------------
# 12. Save Model Flag
# --------------------------------------------------------------------------------------
save_model = True
print(f"Save model set to: {save_model}")

# --------------------------------------------------------------------------------------
# 13. Source Path
# --------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)
