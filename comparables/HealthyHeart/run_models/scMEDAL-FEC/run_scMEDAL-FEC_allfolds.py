import os
import glob
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

from scMEDAL.utils.model_train_utils import run_all_folds
from model_config import *
from scMEDAL.models.scMEDAL import DomainAdversarialAE  # scMEDAL-FEC configuration

print("TensorFlow version:", tf.__version__)


# --------------------------------------------------------------------------------------
# 0. Define folds for cross-validation
# --------------------------------------------------------------------------------------
folds_list = list(range(1, 6, 1))  # e.g., 5 folds. Adjust if needed.


# --------------------------------------------------------------------------------------
# 1. Define batch and biological columns, and set category ordering
# --------------------------------------------------------------------------------------
# If you need a quick test, modify the number of epochs in model_params_dict.
# Note: This won't change the output folder name, only the training duration.

batch_col = model_params_dict['batch_col']
bio_col = model_params_dict['bio_col']
donor_col = model_params_dict['donor_col']  # Optional column for plotting

# For plotting latent spaces, a basic combination is {"shape_col": bio_col, "color_col": batch_col}.
# However, with many cells, shapes may be hard to distinguish. Here we only plot bio_col or donor_col.
shape_color_dict = {
    f"{bio_col}-{bio_col}": {"shape_col": bio_col, "color_col": bio_col},
    f"{donor_col}-{donor_col}": {"shape_col": donor_col, "color_col": donor_col},
    # Additional options:
    # f"{bio_col}-{batch_col}": {"shape_col": bio_col, "color_col": batch_col},
    # f"{batch_col}-{batch_col}": {"shape_col": batch_col, "color_col": batch_col},
}


# --------------------------------------------------------------------------------------
# 2. Load metadata and determine category ordering for donors and celltypes
# --------------------------------------------------------------------------------------
metadata_files = glob.glob(data_path + "/*meta.csv")
if not metadata_files:
    raise FileNotFoundError(f"No metadata CSV file found in {data_path}")

metadata_all = pd.read_csv(metadata_files[0])
metadata_all['celltype'] = metadata_all['celltype'].astype('category')
metadata_all['batch'] = metadata_all['sampleID'].astype('category')

num_batches = len(np.unique(metadata_all[batch_col]))
print("Number of batches:", num_batches)

# Ordered categories for batches/donors
seen_donor_ids = np.unique(metadata_all[batch_col]).tolist()
print("Ordered batches (seen donors):", seen_donor_ids)

# Ordered categories for cell types
celltype_ids = np.unique(metadata_all[bio_col]).tolist()


# --------------------------------------------------------------------------------------
# 3. Run all folds of the experiment
# --------------------------------------------------------------------------------------
# run_all_folds returns a DataFrame with clustering metrics (1/db, ch, silhouette) for
# celltypes and donors. Higher score = better clustering.
# We aim for low batch clustering and high celltype clustering.

mean_scores = run_all_folds(
    Model=DomainAdversarialAE,
    input_base_path=input_base_path,
    out_base_paths_dict=base_paths_dict,
    folds_list=folds_list,
    run_name=run_name,
    model_params_dict=model_params_dict,
    build_model_dict=build_model_dict,
    compile_dict=compile_dict,
    save_model=save_model,
    batch_col=batch_col,
    bio_col=bio_col,
    batch_col_categories=seen_donor_ids,
    bio_col_categories=celltype_ids,
    model_type="ae_da",   # scMEDAL-FEC configuration
    issparse=True,
    load_dense=True,
    shape_color_dict=shape_color_dict,
    sample_size=model_params_dict["sample_size"]
)


# --------------------------------------------------------------------------------------
# 4. Save config.py file for reproducibility
# --------------------------------------------------------------------------------------
destination_path = os.path.join(saved_models_base, run_name, 'model_config.py')
print("\nCopying config.py file to:", destination_path)

# If needed, ensure destination directory exists:
# os.makedirs(os.path.dirname(destination_path), exist_ok=True)

shutil.copy(source_file, destination_path)
