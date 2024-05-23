import pandas
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#from cv2 import transpose
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tkl

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/utilities")
from tensorflow_utilities import use_specific_gpu
from tensorflow.keras.callbacks import History 
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import scanpy as sc
import anndata as ad

print(tf.__version__)
import copy
import os
# path to utils
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
from utils import create_folder
from model_train_utils import run_all_folds#,run_model_pipeline,ModelManager,get_train_val_data,load_data,train_and_save_model,PlotLoss,get_pca_scoresandplots,get_encoder_latentandscores
# path to model_config
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AE_RE")
from model_config import build_model_dict,data_seen,run_name,base_paths_dict,outputs_path, save_model, folder_name, model_name,model_params_dict,compile_dict,data_path
# path to models
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/models")
import glob

from AE_v4 import DomainEnhancingAutoencoderClassifier
import ray
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
RAY_RUN = True
# 0. Define Fold
folds_list = list(range(1,2,1)) #there are 5 folds. Running 2 for testing

# #######################################################################################################################
# 1. Define batch and bio cols and order of the donors and celltypes

batch_col = model_params_dict['batch_col']
bio_col = model_params_dict['bio_col']
donor_col = model_params_dict['donor_col'] #this column is optional

# We will use this dictionary to plot latent spaces. The basic combination is  {"shape_col": bio_col, "color_col": batch_col}, but when we have lots of cells, we cannot distinguish shapes.
# We only plot bio_col
shape_color_dict={f"{bio_col}-{bio_col}": {"shape_col": bio_col, "color_col": bio_col},f"{donor_col}-{donor_col}": {"shape_col": donor_col, "color_col": donor_col}}
                                    

# Define the One Hot encoded (OHE) order for donor and celltype categories
# get metadata before splits
metadata_all = pd.read_csv(glob.glob(data_path+"/*meta.csv")[0])

metadata_all['celltype'] = metadata_all['celltype'].astype('category')
metadata_all['batch'] = metadata_all["sampleID"].astype('category')

print("n batches",len(np.unique(metadata_all[batch_col]).tolist()))



# Define the One Hot encoded (OHE) order for donor and celltype categories
# Seen donors are only pairs
seen_donor_ids = np.unique(metadata_all[batch_col]).tolist()
print("check ordered batches: ",seen_donor_ids)

# celltypes 
celltype_ids = np.unique(metadata_all[bio_col]).tolist()

####################################################################################################################
#Define Trials which is a wrapper over a single run
#A Class Trial has to have at least two functions:
#setup: which receive the configuration dictionary from the search space
#step: which is the function that will run each trial
# Create folder for ray
class Trial(tune.Trainable):

    def setup(self, config: dict):
        self.config = config

    def step(self):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
        sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/models")
        sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AE_RE")
        #Import all additional functions that the script will use here, it doesn't accept import * (wildcard import)
        #Please, please avoid "from something import *" as it makes it hard to track where the function is coming from
        from model_train_utils import run_all_folds,generate_run_name,get_metric2optimize_re
        from model_config import compile_dict,build_model_dict,load_data_dict,train_model_dict,get_scores_dict,data_seen,base_paths_dict, constant_keys, save_model
        from AE_v4 import DomainEnhancingAutoencoderClassifier

        # Extract compile_params, build_model_params and other parameters from config
        compile_params = self.config.get("compile_params", {})
        build_model_params = self.config.get("build_model_params", {})
        other_params = {key: self.config[key] for key in self.config if key not in ["compile_params", "build_model_params"]}

        # Update compile_dict, build_model_dict and train_model_dict (if dict not empty)
        if compile_params:
            compile_dict.update(compile_params)
        if build_model_params:
            build_model_dict.update(build_model_params)

        # Combine all dictionaries for model parameters
        model_params_dict_step = {**compile_dict, **build_model_dict, **load_data_dict, **train_model_dict, **get_scores_dict}
        if other_params:
            model_params_dict_step.update(other_params)  

        # Generate a run unique name
        run_name_step = generate_run_name(model_params_dict_step, constant_keys)
        
        # Run all folds
        mean_scores_dict = run_all_folds(Model=DomainEnhancingAutoencoderClassifier,
                        input_base_path=data_seen,
                        out_base_paths_dict=base_paths_dict,
                        folds_list=folds_list, #there are 5 folds for just running 2 for testing
                        run_name=run_name,
                        model_params_dict=model_params_dict,
                        build_model_dict=build_model_dict,
                        compile_dict=compile_dict,
                        save_model=save_model,
                        batch_col=batch_col,
                        bio_col=bio_col,
                        batch_col_categories=seen_donor_ids,
                        bio_col_categories=celltype_ids,
                        model_type="ae_re",
                        issparse=True,
                        load_dense=True,                
                        shape_color_dict=shape_color_dict,
                        sample_size=model_params_dict["sample_size"])
        # maximizing biological clustering and minimizing batch clustering
        metric2optimize = get_metric2optimize_re(mean_scores_dict, subset='val', metric='silhouette', batch_col=batch_col)
        # we want to max batch, and min bio sep (this is why we take the inverse)
        return {"metric2optimize": metric2optimize}
# Define GPU config
if not RAY_RUN:
    intGPU = 1
    # Set GPU card
    use_specific_gpu(intGPU, fraction=1)
else:
    num_GPU = 0.5
    num_CPU = 4
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    if "ip_head" not in os.environ:
        # If ip_head is not set, then set it to the IP address of current node
        os.environ["ip_head"] = os.popen(
            'hostname -I').read().strip().split(' ')[0]
    print('ip head: ', os.environ["ip_head"])
    ray.init(address='auto',
                _node_ip_address=os.environ["ip_head"].split(":")[0])



# #######################################################################################################################
# Run all folds. It returns a dataframe with 1/db, ch and silhouette score for celltypes and donors.
# higher scores, better clustering. We need low clustering for batch and high clustering for celltype.
if not RAY_RUN:
    mean_scores_dict = run_all_folds(Model=DomainEnhancingAutoencoderClassifier,
                        input_base_path=data_seen,
                        out_base_paths_dict=base_paths_dict,
                        folds_list=folds_list, #there are 5 folds for just running 2 for testing
                        run_name=run_name,
                        model_params_dict=model_params_dict,
                        build_model_dict=build_model_dict,
                        compile_dict=compile_dict,
                        save_model=save_model,
                        batch_col=batch_col,
                        bio_col=bio_col,
                        batch_col_categories=seen_donor_ids,
                        bio_col_categories=celltype_ids,
                        model_type="ae_re",
                        issparse=True,
                        load_dense=True,                
                        shape_color_dict=shape_color_dict,
                        sample_size=model_params_dict["sample_size"])

    print("\nmean_scores\n",mean_scores_dict)
else:

    ray_base = os.path.join(outputs_path, "ray", folder_name, model_name)
    create_folder(ray_base)

    #Ray will save all trials to a folder with name experiment_name in local_dir, change these values to the one you want
    #datetime was added to name to make it unique
    experiment_name = "run_HPO" + \
                        str(pd.Timestamp.now().date()) + '_' + \
                        str(pd.Timestamp.now().time()).replace(':', '-')

    local_dir = ray_base
    #Total number of models to run
    num_models = 3


    # Define the search space
    search_space = {
        "compile_params": { # Update params from compile_dict that you want to tune
        "loss_latent_cluster_weight": tune.uniform(0.05, 0.5),
        "loss_recon_weight": tune.uniform(100, 200)},
        "build_model_params":{"kl_weight": tune.uniform(1e-8, 1e-4)} # Update params from build_model_dict that you want to tune
        # "batch_size": tune.choice([32, 64, 128, 256,512])
        # Add any other parameters you want to optimize
    }


    #Define the experiment metrics, which metric you want the HPO to optimize and how (minimize or maximize)
    #The metric has to be a key in the returned dictionary of Trial.step
    # metric2optimize = bio_mean - batch_mean
    experiment_metrics = dict(metric='metric2optimize', mode="max")

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=99999999999,
        reduction_factor=4,
        **experiment_metrics)
    bohb_search = TuneBOHB(**experiment_metrics)
    # Call tune.run with the Trial class and the search space
    tune.run(
        Trial,
        config=search_space,
        name=experiment_name,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        num_samples=num_models,
        local_dir=local_dir,
        stop={"training_iteration": 1},
        resources_per_trial={"gpu": num_GPU, "cpu": num_CPU},
        max_concurrent_trials=2  # Limit to 2 concurrent trials (to avoid memory errors)
    )

