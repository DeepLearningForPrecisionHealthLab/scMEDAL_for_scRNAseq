


import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
#import shutil
import glob

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve().parents[2]  # two levels up from AML dir (scMEDAL_for_scRNAseq)
sys.path.insert(0, str(PROJECT_ROOT))

from scMEDAL_for_scRNAseq.utils.defaults import ASD_PATHS_CONFIG
data_base_path = ASD_PATHS_CONFIG.get("data_base_path")
scenario_id = ASD_PATHS_CONFIG.get("scenario_id")
outputs_path = ASD_PATHS_CONFIG.get("outputs_path")

from scMEDAL_for_scRNAseq.utils.model_train_utils import generate_run_name, ModelManager
from scMEDAL_for_scRNAseq.utils.utils import get_split_paths
from scMEDAL_for_scRNAseq.comparables.SAUCIE.saucie import run_saucie


# Detect TensorFlow version
import tensorflow as tf
tf_version = tf.__version__
print("tf_version",tf_version)





compile_dict={}
build_model_dict = {
    "n_latent_dims": 50,                  # Number of latent dimensions (last element in 'layers')
    "layers": [512, 132, 50],             # Hidden layers + latent dim (set n_latent_dims as last) 
    #"layers": [512, 256, 132, 50],   # 4 elements: When lambda_c > 0 SAUCIE expects self.layers to have four values
    "lambda_b": 0.001,                      # Batch correction regularization
    #"lambda_b": 0.0,
    "lambda_c":0.0,                         # set to zero when lambda b is zero
    #"lambda_c": 0.05,                      # Clustering regularization
    "lambda_d": 0.0,                      # Intracluster distance regularization (optional)
    "learning_rate": 0.0001,               # Learning rate
}


#--------------------------------------------------------------------------------------
# Data loading parameters
#--------------------------------------------------------------------------------------
load_data_dict = {
    "eval_test": True,      # Set to True if test data should be loaded for evaluation
    "use_z": False,         # If the model requires a design matrix Z, set True. AE_conv does not need it.
    "get_pred": False,      # If predictions are needed. Useful when using classifiers, not for a simple AE.
    "scaling": "min_max"    # Input scaling: "min_max" or "z_scores"
}


#--------------------------------------------------------------------------------------
# Training parameters
#--------------------------------------------------------------------------------------
train_model_dict = {
    "batch_size": 512,      # Training batch size
    # "epochs": 500,            # For testing; for full experiments use larger epochs (e.g., 500)
    #"epochs": 500,
    "epochs": 50,
    "monitor_metric": 'val_loss',  
    "patience": 30,         # Early stopping patience
    "stop_criteria": "early_stopping",
    # "compute_latents_callback": False,
    "sample_size": 10000,   # Used in clustering score callbacks
    "model_type": "ae"      # Type of model: 'ae' for autoencoder
}


#--------------------------------------------------------------------------------------
# Latent space and score computation parameters
#--------------------------------------------------------------------------------------
get_scores_dict = {
    "encoder_latent_name": "SAUCIE_latent_50", # Modify depending on the model used : model_name
    "get_pca": False,
    #"n_components": 50,
    "get_baseline": False   # If True, compute baseline, but it's time-consuming
}


#--------------------------------------------------------------------------------------
# Experimental design parameters
#--------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
expt_design_dict = {
    'batch_col': 'batch',         # Name of the batch column
    'bio_col': 'celltype'         # Biological condition (e.g., cell type)
}


#--------------------------------------------------------------------------------------
# Combine all dictionaries into a single dictionary
#--------------------------------------------------------------------------------------
model_params_dict = {
#    **compile_dict,
    **build_model_dict,
    **load_data_dict,
    **train_model_dict,
    **get_scores_dict,
    **expt_design_dict
}


#--------------------------------------------------------------------------------------
# Define common plotting parameters
#--------------------------------------------------------------------------------------
plot_params = {
    "shape_col": "celltype",
    "color_col": "batch",
    "markers": [
        "x", "+", "<", "h", "s", ".", 'o', 's', '^', '*', '1', '8', 'p', 'P', 'D', '|',
        0, ',', 'd', 2
    ],
    "showplot": False,
    "save_fig": True,
    "outpath": None  # Will be set by the ModelManager
}



#--------------------------------------------------------------------------------------
# Define base paths for input data
#--------------------------------------------------------------------------------------
data_path = os.path.join(data_base_path, scenario_id)
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')
print(f"Parent folder: {input_base_path}")


#--------------------------------------------------------------------------------------
# Define paths for experiment outputs
#--------------------------------------------------------------------------------------
print("Outputs saved to:", outputs_path)

folder_name = scenario_id
model_name = "SAUCIE_50dims"

# Define base paths for saving models, figures, and latent spaces
saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)


#--------------------------------------------------------------------------------------
# Generate run name for the experiment
#--------------------------------------------------------------------------------------
constant_keys = [
    'batch_col', 'bio_col', 'donor_col', "layer_units_latent_classifier", "name",
    "monitor_metric", "stop_criteria", "get_pca", "get_baseline", 'use_z',
    'encoder_latent_name', 'sigmoid_eval_test', 'last_activation', 'get_pred',
    "eval_test", "optimizer", "loss", "loss_weights", "metrics"
]

# Generate a name for this run, excluding constant_keys from the naming scheme
run_name = generate_run_name(model_params_dict, constant_keys, name='run_crossval')
print("run_name:", run_name)


#--------------------------------------------------------------------------------------
# Define dictionary of base paths
#--------------------------------------------------------------------------------------
base_paths_dict = {
    "models": saved_models_base,
    "figures": figures_base,
    "latent": latent_space_base
}


#--------------------------------------------------------------------------------------
# Set whether to save the model
#--------------------------------------------------------------------------------------
save_model = True
print("save_model set to:", save_model)


#--------------------------------------------------------------------------------------
# Get the source path of this config file
#--------------------------------------------------------------------------------------
source_file = os.path.abspath(__file__)








# ======================
# 2. Main Execution Block
# ======================
if __name__ == "__main__":
    import glob
    # Example as in your scVI block, adapt as needed!
    folds_list = list(range(1, 6))
    batch_col = model_params_dict["batch_col"]
    bio_col   = model_params_dict["bio_col"]

    shape_color_dict = {
        "{}-{}".format(bio_col, bio_col):   {"shape_col": bio_col,   "color_col": bio_col},
        "{}-{}".format(batch_col, batch_col): {"shape_col": batch_col, "color_col": batch_col},
    }
    metadata_all = pd.read_csv(glob.glob(data_path + "/*meta.csv")[0])
    # Convert cell type and batch columns to categorical
    metadata_all['celltype'] = metadata_all['celltype'].astype('category')
    metadata_all['batch'] = metadata_all["batch"].astype('category')
    seen_donor_ids = np.unique(metadata_all[batch_col]).tolist()
    celltype_ids   = np.unique(metadata_all[bio_col]).tolist()

    print("Number of batches:", len(seen_donor_ids))

    # SAUCIE fold loop

    all_scores_per_fold = {}

    for intFold in folds_list:
        print(f"\nRunning SAUCIE Fold {intFold}\n{'='*60}")
        model_manager = ModelManager(
            params_dict=model_params_dict,
            base_paths_dict=base_paths_dict,
            run_name=run_name,
            save_model=save_model,
            use_kfolds=True,
            kfold=intFold,
        )
        model_params = model_manager.params
        input_path_dict = get_split_paths(base_path=input_base_path, fold=intFold)

        results = run_saucie(
            save_model=save_model,
            batch_col=batch_col,
            bio_col=bio_col,
            input_path_dict=input_path_dict,
            model_params=model_params,
            shape_color_dict=shape_color_dict,
            sample_size=model_params.sample_size,
            plot_params=plot_params,
        )

        all_scores_per_fold[intFold] = results

    # Aggregation after all folds
    os.makedirs(model_params.latent_path_main, exist_ok=True)

    for data_split in ["train", "val", "test"]:
        if data_split == "test" and not model_params.eval_test:
            continue

        rows = []
        for fold_idx, split_dict in all_scores_per_fold.items():
            latent_key = f"{model_params.encoder_latent_name}_{data_split}"
            scores_df = split_dict[data_split].copy()
            scores_df["fold"] = fold_idx
            scores_df["dataset_type"] = latent_key
            rows.append(scores_df)

        big_df = pd.concat(rows, ignore_index=True)

        # Save detailed scores per fold
        big_df.to_csv(
            os.path.join(model_params.latent_path_main,
                        f"all_scores_{data_split}_samplesize-{model_params.sample_size}.csv"),
            index=False
        )

        numeric_cols = big_df.select_dtypes(include="number").columns

        mean_scores = (
            big_df.groupby("dataset_type")[numeric_cols]
            .mean()
            .reset_index(drop=True)
        )

        std_scores = (
            big_df.groupby("dataset_type")[numeric_cols]
            .std(ddof=1)
            .reset_index(drop=True)
        )

        sem_scores = std_scores / (len(folds_list) ** 0.5)

        summary_df = mean_scores.copy()
        summary_df.loc['mean'] = mean_scores.iloc[0]
        summary_df.loc['std'] = std_scores.iloc[0]
        summary_df.loc['sem'] = sem_scores.iloc[0]

        summary_df = summary_df.T
        summary_df = summary_df.loc[:, summary_df.columns != 0]

        # Save summarized mean, std, sem scores
        summary_df.to_csv(
            os.path.join(
                model_params.latent_path_main,
                f"mean_scores_{data_split}_samplesize-{model_params.sample_size}.csv",
            ),
            header=True
        )
