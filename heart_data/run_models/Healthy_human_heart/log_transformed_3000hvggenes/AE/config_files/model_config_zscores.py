import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
from model_train_utils import generate_run_name
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError as mse_loss
# from tensorflow.keras.losses import BinaryCrossentropy as bce_loss
from tensorflow.keras.metrics import MeanSquaredError as mse_metric
# from tensorflow.keras.metrics import AUC as auc_metric
import os



compile_dict = {# compile settings
    "optimizer":Adam(lr=0.0001),
    "loss":[mse_loss(name='mean_squared_error')],
    "loss_weights":1,
    "metrics":[[mse_metric()]]}



build_model_dict = {
    "n_latent_dims": 2,  # init settings
#    "layer_units": [10],
    "layer_units": [512,132],
#    "layer_units_latent_classifier": [2], # comment this line for AE
#    "n_pred": 13, #n celltypes
#    "last_activation": "sigmoid",
    "last_activation": "linear", #last activation of the decoder (will determine how the reconstructed outputs look)
    "use_batch_norm":True, #This is batch norm for encoder. Default is False
    "name": "AE" # Call the model that you want to use
}


load_data_dict = {
    "eval_test": False,# Set to true if you want to load test data
    "use_z": False, # Depending on the model you may need z design matrix: For AE_conv you do not need it
    "get_pred": False, #I put it here because it is not needed in build_model_dict but we still use it to load data
    "scaling": "z_scores" # Scaling of input data: "min_max" or "z_scores"
}

train_model_dict = {
#    "batch_size": 60,  # training settings
    "batch_size": 512,
#    "epochs": 20,
    "epochs":500,
    "monitor_metric": 'val_loss',
    "patience": 30,
    "stop_criteria": "early_stopping",
    "compute_latents_callback": False,
    "sample_size":10000,
    "model_type":"ae" #This sample size is used in the clustering scores callback
}



get_scores_dict = {
    "encoder_latent_name":"AE_latent_50", #Modify depending on the model
    "get_pca": True,
    "n_components":50,
    "get_baseline": True #take forever
}

expt_design_dict = {'batch_col':'batch', #name of the batch column
                        'bio_col':'celltype',
                        'donor_col':'DonorID' # optional, this may be useful for plotting
                    }

# Combine all dictionaries into model_params_dict

# model_params_dict now contains all key-value pairs from the individual dictionaries
model_params_dict = {**compile_dict, **build_model_dict, **load_data_dict, **train_model_dict, **get_scores_dict,**expt_design_dict}

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
model_name = "AE"

# Folder structure setup
# saved_models_base = 'path_to_saved_models'
saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
# figures_base = 'path_to_figures'
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
# latent_space_base = 'path_to_latent_space'
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)


# Define the run name (ensure model_params_dict is defined before this point)
#"layer_units"


constant_keys = ['batch_col','bio_col','donor_col',"layer_units_latent_classifier", "name", "monitor_metric", "stop_criteria","get_pca","get_baseline",'use_z','encoder_latent_name','sigmoid_eval_test','last_activation','get_pred',"eval_test","optimizer","loss","loss_weights","metrics"]
# run_name = generate_run_name(model_params_dict, constant_keys, name='run_HPO')
run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
print("run_name",run_name)

# define dict of base paths: # base_paths = {"models:'/path/to/saved_models', "figures":'/path/to/figures',"latent": '/path/to/latent_spaces'}
base_paths_dict = {"models":saved_models_base,"figures":figures_base,"latent":latent_space_base}
# set save_model to False/True
save_model = True
print("save model set to ",save_model)


# Get the source path of the config_file.py
source_file = os.path.abspath(__file__)
# initialize model manager
# model_manager = ModelManager(model_params_dict =model_params_dict, base_paths_dict=base_paths_dict, run_name=run_name, save_model=save_model)
# Update parameters if needed:
# model_manager.update_params({'new_param': 'new_value'})



