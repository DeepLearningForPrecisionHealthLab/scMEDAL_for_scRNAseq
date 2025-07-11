
import sys
# Import paths
sys.path.append("../../../../")
from paths_config import data_base_path, scenario_id, outputs_path
from scMEDAL.utils.model_train_utils import generate_run_name
import os

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import CategoricalCrossentropy
# from tensorflow.keras.metrics import CategoricalAccuracy
# --------------------------------------------------------------------------------------
# Compilation Parameters
# --------------------------------------------------------------------------------------
compile_dict = {}
#     "optimizer": Adam(learning_rate=0.001),
#     "loss": CategoricalCrossentropy(name='categorical_crossentropy'),
#     "loss_weights": 1,
#     "metrics": [CategoricalAccuracy(name='accuracy')]
# }

# --------------------------------------------------------------------------------------
# Model Building Parameters
# --------------------------------------------------------------------------------------
build_model_dict = {
    # "n_latent_dims": 2,
    # "layer_units": [8, 4],
    "n_pred": 2, #dx
    # "add_re_2_meclass": False,
    "name": "mec"
}

# --------------------------------------------------------------------------------------
# Data Loading Parameters
# --------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,
    "scaling": None
}

# --------------------------------------------------------------------------------------
# Training Parameters
# --------------------------------------------------------------------------------------
train_model_dict = {
    # "batch_size": 512,
    # "epochs": 2,            # For testing; for full experiments use larger epochs (e.g., 200)
    # "epochs": 200,
    # "monitor_metric": 'val_loss',
    # "patience": 30,
    # "stop_criteria": "early_stopping",
    # "compute_latents_callback": False
}

# --------------------------------------------------------------------------------------
# Latent Space and Score Computation Parameters
# --------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "MEC_latent",
    "get_pca": False,
    "get_baseline": False
}

# --------------------------------------------------------------------------------------
# Experimental Design Parameters
# --------------------------------------------------------------------------------------
expt_design_dict = {
    'batch_col': 'batch',
    'bio_col': 'diagnosis' #predict diagnosis
}


# --------------------------------------------------------------------------------------
# Model Configuration Parameters
# --------------------------------------------------------------------------------------
save_model = True
print("save model set to", save_model)

LatentClassifier_config = {
    'Model': None,
    'build_model_dict': build_model_dict,
    'compile_dict': compile_dict,
    'save_model': save_model,
    'latent_keys_config': {
    'fe_latent': 'harmony_50dims_latent',
    're_latent': 'scMEDAL-RE_50dims_latent',
    },
    'return_metrics': True,
    'return_adata_dict': True,
    'return_trained_model': True,
    'model_type': 'mec',
    'seed': 42
}

# --------------------------------------------------------------------------------------
# Load Latent Spaces Configuration
# --------------------------------------------------------------------------------------
load_latent_spaces_dict = {
    'latent_path_dict': None,
    'model_params': None,
    'base_path': None,
    'fold': 2,# placeholder, will be changed in for loop
    'models_list': ["scMEDAL-RE_50dims","harmony_50dims"],
    'batch_col_categories': None,
    'bio_col_categories': None
}

# Update with experimental design parameters
load_latent_spaces_dict.update(expt_design_dict)

# --------------------------------------------------------------------------------------
# Combine All Dictionaries into model_params_dict
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
# Define Base Paths for Input Data
# --------------------------------------------------------------------------------------
data_path = os.path.join(data_base_path, scenario_id)
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')
load_latent_spaces_dict['base_path'] = input_base_path
print(f"Parent folder: {input_base_path}")

# --------------------------------------------------------------------------------------
# Define Paths for Experiment Outputs
# --------------------------------------------------------------------------------------
folder_name = scenario_id
model_name = "MEC_50dims"

saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

# --------------------------------------------------------------------------------------
# Generate Run Name
# --------------------------------------------------------------------------------------
constant_keys = [
    # "batch_size", "batch_col", "bio_col", "compute_latents_callback",
    # "layer_units", "layer_units_latent_classifier", "name", "monitor_metric",
    # "stop_criteria", "get_pca", "get_baseline", 'use_z', 'encoder_latent_name',
    # 'sigmoid_eval_test', 'last_activation', 'get_pred', "eval_test", "optimizer",
    # "loss", "loss_weights", "metrics", "add_re_2_meclass"
]

run_name = generate_run_name(
    LatentClassifier_config['latent_keys_config'],
    constant_keys=None,
    name='run_latent_classifier_harmony_dx'
)
print("run_name", run_name)

# --------------------------------------------------------------------------------------
# Define Dictionary of Base Paths
# --------------------------------------------------------------------------------------
base_paths_dict = {
    "models": saved_models_base,
    "figures": figures_base,
    "latent": latent_space_base
}

# --------------------------------------------------------------------------------------
# Get Source File Path
# --------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)