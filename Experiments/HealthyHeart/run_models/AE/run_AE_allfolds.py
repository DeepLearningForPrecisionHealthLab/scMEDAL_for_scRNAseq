import sys
import glob
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

from scMEDAL.utils.utils import *
from scMEDAL.utils.model_train_utils import run_all_folds
from model_config import *  # Imports all model configurations and parameters
from scMEDAL.models.scMEDAL import AE

print("TensorFlow version:", tf.__version__)
# Environment: run_models_env

# ---------------------------------------------------------------------------------------
# 0. Define 5-fold cross-validation
# ---------------------------------------------------------------------------------------
folds_list = list(range(1, 6, 1))  # [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------------------
# 1. Define batch and biological columns, and order of donors and cell types
# ---------------------------------------------------------------------------------------
# If you need a quick test, change model_params_dict["epochs"]. Note that this won't update
# the folder name for outputs, but can help you test faster.

batch_col = model_params_dict['batch_col']
bio_col = model_params_dict['bio_col']
donor_col = model_params_dict['donor_col']  # This column is optional

# For plotting latent spaces, different combinations of shape and color columns can be used.
# Basic combination is {"shape_col": bio_col, "color_col": bio_col}. When dealing with many
# cells, it might be hard to distinguish shapes, so consider only plotting one type of metadata.
shape_color_dict = {
    f"{bio_col}-{bio_col}": {"shape_col": bio_col, "color_col": bio_col},
    f"{donor_col}-{donor_col}": {"shape_col": donor_col, "color_col": donor_col},
    # Additional combinations could be uncommented if needed:
    # f"{bio_col}-{batch_col}": {"shape_col": bio_col, "color_col": batch_col},
    # f"{batch_col}-{batch_col}": {"shape_col": batch_col, "color_col": batch_col},
}


# ---------------------------------------------------------------------------------------
# 2. Load metadata to determine category ordering for donors and cell types
# ---------------------------------------------------------------------------------------
metadata_all = pd.read_csv(glob.glob(data_path + "/*meta.csv")[0])
metadata_all['celltype'] = metadata_all['celltype'].astype('category')
metadata_all['batch'] = metadata_all["sampleID"].astype('category')

num_batches = len(np.unique(metadata_all[batch_col]))
print("Number of batches:", num_batches)

# Get ordered donor IDs (seen batch categories)
seen_donor_ids = np.unique(metadata_all[batch_col]).tolist()
print("Ordered batches (seen donor IDs):", seen_donor_ids)

# Get ordered cell type IDs
celltype_ids = np.unique(metadata_all[bio_col]).tolist()


# ---------------------------------------------------------------------------------------
# 3. Run all folds
# ---------------------------------------------------------------------------------------
# This will return a dataframe with clustering scores (1/db, ch, silhouette) for cell types
# and donors. Higher score = better clustering. We aim for low clustering on batch and high
# clustering on biological conditions (cell type).

mean_scores = run_all_folds(
    Model=AE,
    input_base_path=input_base_path,
    out_base_paths_dict=base_paths_dict,
    folds_list=folds_list,  # For testing you can reduce the number of folds
    run_name=run_name,
    model_params_dict=model_params_dict,
    build_model_dict=build_model_dict,
    compile_dict=compile_dict,
    save_model=save_model,
    batch_col=batch_col,
    bio_col=bio_col,
    batch_col_categories=seen_donor_ids,
    bio_col_categories=celltype_ids,
    model_type="ae",
    issparse=True,
    load_dense=True,
    shape_color_dict=shape_color_dict,
    sample_size=model_params_dict["sample_size"]
)


# ---------------------------------------------------------------------------------------
# 4. Save the configuration file for reproducibility
# ---------------------------------------------------------------------------------------
destination_path = os.path.join(saved_models_base, run_name, 'model_config.py')

print("\nCopying config.py file to:", destination_path)
shutil.copy(source_file, destination_path)
