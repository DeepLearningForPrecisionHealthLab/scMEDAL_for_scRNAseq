import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics/utils")
from model_train_utils import generate_run_name
import os
import tensorflow as tf


# Define individual dictionaries
compile_dict = {"loss_recon":tf.keras.losses.MeanSquaredError(), #recon loss
    "loss_multiclass":tf.keras.losses.CategoricalCrossentropy(), #class loss
    "metric_multiclass":tf.keras.metrics.CategoricalAccuracy(name='acc'),
    "opt_autoencoder":tf.keras.optimizers.Adam(lr=0.0001), #optimizer AEC
    "opt_adversary":tf.keras.optimizers.Adam(lr=0.0001),#optimizer Adversary
    "loss_gen_weight": 1,  # compile settings
#    "loss_recon_weight": 1800,
    "loss_recon_weight":3000,#1800*3
#    "loss_recon_weight": 0,#to test if we get 4.99 (max)
    "loss_class_weight": 1
}

build_model_dict = {
    "n_latent_dims":2,  # init settings
#    "layer_units": [10],
    "layer_units": [512,132],
    "n_clusters":31,#n batches
#    "layer_units_latent_classifier": [2], #not needed for ae_da
#    "n_pred": 13,# n celltypes # not needed for ae_da
    "get_pred": False, #In this case we want AE_DA model Set to true if you want to train the model with a celltype classification loss function
#    "last_activation": "sigmoid",
    "last_activation": "linear", #last activation of the decoder (will determine how the reconstructed outputs look)
    "use_batch_norm":True, #This is batch norm for encoder. Default is False
    "name": "ae_da" # Call the model that you want to use
}

load_data_dict = {
    "eval_test": True,# Set to true if you want to load test data
    "use_z": True, # Depending on the model you may need z design matrix
    "scaling": "min_max" # Scaling of input data: "min_max" or "z_scores"
}

train_model_dict = {
    "batch_size": 512,  # training settings
#    "epochs": 20,
#    "epochs": 500,
    "epochs":500,
    "monitor_metric": 'val_total_loss',
    "patience": 30,
    "stop_criteria": "early_stopping",
    "compute_latents_callback": False,
    "sample_size":10000, #This sample size is used in the clustering scores callback
    "model_type":"ae_da"
}

get_scores_dict = {
    "encoder_latent_name":"FE_AE_latent_2", #Modify depending on the model
    "get_pca": False,
    "n_components":2,
    "get_baseline": False
}

expt_design_dict = {'batch_col':'batch', #name of the batch column
                        'bio_col':'celltype'
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




data_base_path = "/archive/bioinformatics/DLLab/AixaAndrade/data/Genomic_data/ASD/reverse_norm"

scenario_id = "log_transformed_2916hvggenes"

# Construct the data paths
# data_path = os.path.join(data_base_path, simulation_folder)
# data_seen = os.path.join(data_path, scenario_id,'splits')
data_path = os.path.join(data_base_path,scenario_id)
data_seen = os.path.join(data_base_path,scenario_id,'splits')
print(f"Parent folder: {data_seen}")



# Base output path
# outputs_path = "/path/to/outputs"
#outputs_path = "../../../../../../../outputs"
outputs_path ="/archive/bioinformatics/DLLab/AixaAndrade/results/mixedeffectsdl/results/ARMED_genomics/ASD/reverse_norm/outputs"
# dataset_type = "subsets"
folder_name = "log_transformed_2916hvggenes"
model_name = "AE_DA"

# Folder structure setup
# saved_models_base = 'path_to_saved_models'
saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
# figures_base = 'path_to_figures'
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
# latent_space_base = 'path_to_latent_space'
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)


# Define the run name (ensure model_params_dict is defined before this point)
# "layer_units"

constant_keys = ["compute_latents_callback",'n_components','batch_col','bio_col','donor_col',"loss_recon","loss_multiclass","metric_multiclass","opt_autoencoder","opt_adversary","layer_units_latent_classifier", "n_pred", "n_clusters", "name", "monitor_metric", "stop_criteria","get_pca","get_baseline",'use_z','encoder_latent_name','sigmoid_eval_test','last_activation','get_pred',"eval_test"]
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

# Get the source path of the config_file.py
source_file = os.path.abspath(__file__)



