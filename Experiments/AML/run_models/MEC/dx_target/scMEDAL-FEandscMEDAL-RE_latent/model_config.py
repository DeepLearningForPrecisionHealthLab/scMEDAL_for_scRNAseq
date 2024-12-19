import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
#from utils import read_adata,min_max_scaling,get_split_paths,plot_rep,get_OHE, plot_rep,calculate_merge_scores,plot_table
from model_train_utils import generate_run_name#load_data,train_and_save_model,get_train_val_data,get_x_y_z,get_latent_spaces_paths,load_latent_spaces, ModelManager,prepare_latent_space_inputs,get_encoder_latentandscores,get_encoder_latentandscores,save_latent_representation,evaluate_model,PlotLoss,run_model_pipeline_LatentClassifier

import os
# import glob
# import anndata as ad
# from anndata import AnnData




from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy


# Define the compile dictionary for the MixedEffectsModel
compile_dict = {
    "optimizer": Adam(learning_rate=0.001),
    "loss": CategoricalCrossentropy(name='categorical_crossentropy'),
    "loss_weights": 1,
    "metrics": [CategoricalAccuracy(name='accuracy')]
}

build_model_dict = {
                "n_latent_dims":2, 
                "layer_units":[8,4], 
                "n_pred":3,
#               "post_loc_init_scale":0.1,
#                "prior_scale": 0.25,
#                "kl_weight": 1e-5,
                "add_re_2_meclass":False, #If set to False, the following args are ignored: "post_loc_init_scale","prior_scale","kl_weight"
                "name": "mec" # Call the model that you want to use
                }

load_data_dict = {
    "eval_test": True,# Set to true if you want to load test data
    "scaling": None # Scaling of input data: "min_max" or "z_scores"
}

train_model_dict = {
    "batch_size": 512,  # training settings
    "epochs": 200,
    "monitor_metric": 'val_loss',
    "patience": 30,
    "stop_criteria": "early_stopping",
    "compute_latents_callback": False
}


get_scores_dict = {
    "encoder_latent_name":"MEC_latent", #Modify depending on the model
    "get_pca": False,
    "get_baseline": False
}

expt_design_dict = {'batch_col':'batch', #name of the batch column
                        'bio_col':"Patient_group" # this time we are predicting patient group" : AML, healthy, cell line
                    }


# set save_model to False/True
save_model = True
print("save model set to ",save_model)
# 3. Write the config for the model
LatentClassifier_config = {
    'Model': None,
    'build_model_dict': build_model_dict,
    'compile_dict': compile_dict,
    'save_model': save_model,
    'latent_keys_config': {
        'fe_latent': 'AE_DA_latent',
        're_latent': 'AE_RE_latent'
    },
    'return_metrics': True,
    'return_adata_dict': True,
    'return_trained_model': True,
    'model_type': 'mec',
    'seed':42 # fix seed for chance accuracy

}


load_latent_spaces_dict = {
    'latent_path_dict': None,
    'model_params': None,
    'base_path': None,  # Make sure base_path is defined in your context
    'fold': 2,
    'models_list': ["AE_RE", "AE_DA", "AEC_DA", "AEC", "AE"],
    'batch_col_categories': None,
    'bio_col_categories': None,  # Or set to None if that's intended
}

load_latent_spaces_dict = {**load_latent_spaces_dict,**expt_design_dict }




 

# Combine all dictionaries into model_params_dict

# model_params_dict now contains all key-value pairs from the individual dictionaries
model_params_dict = {**compile_dict, **build_model_dict, **load_data_dict, **train_model_dict, **get_scores_dict,**expt_design_dict}

# Define common plotting parameters. You will update the outpath after creating model_params with ModelManager
# plot_params = {"shape_col": "celltype",
#         "color_col": "donor",
#         "markers":["x","+","<","h","s",".",'o', 's', '^', '*','1','8','p','P','D','|',0,',','d',2],
#         "showplot": True,
#         "save_fig": False,
#         "outpath": ''}



data_base_path = "/archive/bioinformatics/DLLab/AixaAndrade/data/Genomic_data/VanGallen_2019"

scenario_id = "log_transformed_2916hvggenes"

# Construct the data paths
data_path = os.path.join(data_base_path,scenario_id)
data_seen = os.path.join(data_base_path,scenario_id,'splits')
# Update load_latent_spaces_dict
load_latent_spaces_dict['base_path'] = data_seen
print(f"Parent folder: {data_seen}")



# Base output path
# outputs_path = "/path/to/outputs"
#outputs_path = "../../../../../../../outputs"
outputs_path ="/archive/bioinformatics/DLLab/AixaAndrade/results/mixedeffectsdl/results/ARMED_genomics/VanGallen_2019/outputs"
# dataset_type = "subsets"
folder_name = "log_transformed_2916hvggenes"
model_name = "MEC"

# Folder structure setup
# saved_models_base = 'path_to_saved_models'
saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
# figures_base = 'path_to_figures'
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
# latent_space_base = 'path_to_latent_space'
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)




# Define the run name (ensure model_params_dict is defined before this point)
constant_keys = ["batch_size","batch_col","bio_col","compute_latents_callback","layer_units", "layer_units_latent_classifier", "name", "monitor_metric", "stop_criteria","get_pca","get_baseline",'use_z','encoder_latent_name','sigmoid_eval_test','last_activation','get_pred',"eval_test","optimizer","loss","loss_weights","metrics","add_re_2_meclass"]
# run_name = generate_run_name(model_params_dict, constant_keys, name='run_HPO')
#run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
run_name = generate_run_name(LatentClassifier_config['latent_keys_config'], constant_keys=None, name='run_latent_classifier')
print("run_name",run_name)

# define dict of base paths: # base_paths = {"models:'/path/to/saved_models', "figures":'/path/to/figures',"latent": '/path/to/latent_spaces'}
base_paths_dict = {"models":saved_models_base,"figures":figures_base,"latent":latent_space_base}



# Get the source path of the config_file.py
source_file = os.path.abspath(__file__)