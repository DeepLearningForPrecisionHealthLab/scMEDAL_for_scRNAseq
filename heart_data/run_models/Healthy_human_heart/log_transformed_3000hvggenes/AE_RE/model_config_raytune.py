import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
from model_train_utils import generate_run_name
import os
import tensorflow as tf

#Define model parameters

# Define individual dictionaries



compile_dict = {"loss_recon":tf.keras.losses.MeanSquaredError(),# recon loss
                "loss_multiclass":tf.keras.losses.CategoricalCrossentropy(),# multi classification loss 
                #for cluster  loss
                # multi classification metric
                "metric_multiclass":tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),   
                "optimizer":tf.keras.optimizers.Adam(lr=0.0001),
                "loss_recon_weight":110.0,    
                #"loss_class_weight":0.01, #compile params (These are default but can be called when compiling). Not relevant when "get_pred": False, 
                "loss_latent_cluster_weight":0.1,
                #"loss_recon_cluster_weight":0.001 #Not relevant If "get_recon_cluster": False,
                }

build_model_dict = {
                "n_latent_dims":2, #init parameters
                "n_clusters":147, #batches
#                "layer_units":[10],
                "layer_units": [512,132],
                "layer_units_classifier":[5],
                "n_pred":13, #n celltypes
#                "last_activation":"sigmoid",
                "last_activation": "linear", #last activation of the decoder (will determine how the reconstructed outputs look)
                "post_loc_init_scale":0.1,
                "prior_scale": 0.25,
                "kl_weight": 1e-5,
                "get_pred": False, #optional to add or remove class classification
                "get_recon_cluster": False, #optional to add or remove reconstruction cluster loss
                "name": "ae_re" # Call the model that you want to use
                }

load_data_dict = {
    "eval_test": False,# Set to true if you want to load test data
    "use_z": True # Depending on the model you may need z design matrix
}

train_model_dict = {
    "batch_size": 512,  # training settings
#    "epochs": 20,
    "epochs": 500,
    "monitor_metric": 'val_total_loss',
    "patience": 30,
    "stop_criteria": "early_stopping",
    "compute_latents_callback": False,
    "sample_size":10000,
    "model_type":"ae_re" #This sample size is used in the clustering scores callback
}



get_scores_dict = {
    "encoder_latent_name":"RE_AE_latent_50", #Modify depending on the model
    "get_pca": False,
    "n_components":50,
    "get_baseline": False
}


expt_design_dict = {'batch_col':'batch', #name of the batch column
                        'bio_col':'celltype',
                        'donor_col':'DonorID', # optional, this may be useful for plotting
                        'tissue_col':'TissueDetail'
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
model_name = "AE_RE"





# set save_model to False/True
save_model = True
print("save model set to ",save_model)

#Wethether to Run Ray
RAY_RUN = True
if RAY_RUN==False:
    # Folder structure setup
    # saved_models_base = 'path_to_saved_models'
    saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
    # figures_base = 'path_to_figures'
    figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
    # latent_space_base = 'path_to_latent_space'
    latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

    # Define the run name (ensure model_params_dict is defined before this point)
    #"layer_units"
    constant_keys = ["n_components","loss_recon","loss_multiclass","metric_multiclass", "optimizer",'model_type','tissue_col','batch_col','bio_col','donor_col',"layer_units_classifier","get_recon_cluster","prior_scale","post_loc_init_scale", "layer_units_latent_classifier", "n_pred", "n_clusters", "name", "monitor_metric", "stop_criteria","get_pca","get_baseline",'use_z','encoder_latent_name','sigmoid_eval_test','last_activation','get_pred',"eval_test"]
    run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
    print("run_name",run_name)

    # define dict of base paths: # base_paths = {"models:'/path/to/saved_models', "figures":'/path/to/figures',"latent": '/path/to/latent_spaces'}
    base_paths_dict = {"models":saved_models_base,"figures":figures_base,"latent":latent_space_base}

# initialize model manager
# model_manager = ModelManager(model_params_dict =model_params_dict, base_paths_dict=base_paths_dict, run_name=run_name, save_model=save_model)
# Update parameters if needed:
# model_manager.update_params({'new_param': 'new_value'})



