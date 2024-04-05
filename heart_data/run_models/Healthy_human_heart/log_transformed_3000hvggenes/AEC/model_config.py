import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics/utils")
from model_train_utils import generate_run_name
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as mse_loss
from tensorflow.keras.losses import BinaryCrossentropy as bce_loss
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
from tensorflow.keras.metrics import AUC as auc_metric
import os




# Define individual dictionaries

compile_dict = {# compile settings
    "optimizer":Adam(lr=0.0001),
    "loss":{'reconstruction_output': mse_loss(name='mse'),'classification_output': bce_loss(name='bce')},
    "loss_weights":{'reconstruction_output': 1.0, 'classification_output': 0.1},
    "metrics":{'reconstruction_output':[mse_metric(name="mse_metric")],'classification_output': [auc_metric(name='auroc')]}
}



build_model_dict = {
    "n_latent_dims": 2,  # init settings
    "layer_units": [10],
#    "layer_units": [512,132],
    "layer_units_latent_classifier": [2],
    "n_pred": 13, #n celltypes
    "last_activation": "sigmoid",
    "name": "AEC" # Call the model that you want to use
}

load_data_dict = {
    "eval_test": False,# Set to true if you want to load test data
    "use_z": False, # Depending on the model you may need z design matrix: For AE_conv you do not need it
    "get_pred": True #I put it here because it is not needed in build_model_dict but we still use it to load data
}

train_model_dict = {
#    "batch_size": 60,  # training settings
    "batch_size": 512,
#    "epochs": 20,
    "epochs": 200,
    "monitor_metric": 'val_loss',
    "patience": 30,
    "stop_criteria": "early_stopping"
}

get_scores_dict = {
    "encoder_latent_name":"AEC_latent_2", #Modify depending on the model
    "get_pca": False,
    "get_baseline": False #take forever
}

# Combine all dictionaries into model_params_dict

# model_params_dict now contains all key-value pairs from the individual dictionaries
model_params_dict = {**compile_dict, **build_model_dict, **load_data_dict, **train_model_dict, **get_scores_dict}

# Define common plotting parameters. You will update the outpath after creating model_params with ModelManager
plot_params = {"shape_col": "celltype",
        "color_col": "donor",
        "markers":["x","+","<","h","s",".",'o', 's', '^', '*','1','8','p','P','D','|',0,',','d',2],
        "showplot": False,
        "save_fig": True,
        "outpath": None}

# Base data path

data_base_path = "/archive/bioinformatics/DLLab/AixaAndrade/data/Genomic_data/heart_data/"

scenario_id = "Healthy_human_heart_data/log_transformed_3000hvggenes"

# Construct the data paths
# data_path = os.path.join(data_base_path, simulation_folder)
# data_seen = os.path.join(data_path, scenario_id,'splits')
data_path = os.path.join(data_base_path,scenario_id)
data_seen = os.path.join(data_base_path,scenario_id,'splits')
print(f"Parent folder: {data_seen}")



# Base output path
# outputs_path = "/path/to/outputs"
#outputs_path = "../../../../../../../outputs"
outputs_path ="/archive/bioinformatics/DLLab/AixaAndrade/results/mixedeffectsdl/results/ARMED_genomics/heart_data/outputs"
# dataset_type = "subsets"
folder_name = "Healthy_human_heart_data/log_transformed_3000hvggenes"
model_name = "AEC"

# Folder structure setup
# saved_models_base = 'path_to_saved_models'
saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
# figures_base = 'path_to_figures'
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
# latent_space_base = 'path_to_latent_space'
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)


# Define the run name (ensure model_params_dict is defined before this point)
#"layer_units"
constant_keys = ["layer_units_latent_classifier", "name", "monitor_metric", "stop_criteria","get_pca","get_baseline",'use_z','encoder_latent_name','sigmoid_eval_test','last_activation','get_pred',"eval_test","optimizer","loss","loss_weights","metrics"]
# run_name = generate_run_name(model_params_dict, constant_keys, name='run_HPO')
run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
print("run_name",run_name)

# define dict of base paths: # base_paths = {"models:'/path/to/saved_models', "figures":'/path/to/figures',"latent": '/path/to/latent_spaces'}
base_paths_dict = {"models":saved_models_base,"figures":figures_base,"latent":latent_space_base}
# set save_model to False/True
save_model = True
print("save model set to ",save_model)
# initialize model manager
# model_manager = ModelManager(model_params_dict =model_params_dict, base_paths_dict=base_paths_dict, run_name=run_name, save_model=save_model)
# Update parameters if needed:
# model_manager.update_params({'new_param': 'new_value'})



