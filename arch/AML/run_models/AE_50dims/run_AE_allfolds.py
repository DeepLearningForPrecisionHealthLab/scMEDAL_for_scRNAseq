import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import shutil
import os

from scMEDAL.utils.model_train_utils import run_all_folds
from model_config import *  # Import all model configurations and parameters
from scMEDAL.models.scMEDAL import AE

print("TensorFlow version:", tf.__version__)

# --------------------------------------------------------------------------------------
# 0. Define Fold Parameters
# --------------------------------------------------------------------------------------
folds_list = list(range(1, 6))  # Define folds (10 total)

# --------------------------------------------------------------------------------------
# 1. Define Batch and Bio Columns
# --------------------------------------------------------------------------------------
batch_col = model_params_dict['batch_col']
bio_col = model_params_dict['bio_col']

# Define shape-color dictionary for plotting latent spaces
shape_color_dict = {
    f"{bio_col}-{bio_col}": {"shape_col": bio_col, "color_col": bio_col},
    f"{batch_col}-{batch_col}": {"shape_col": batch_col, "color_col": batch_col},
    # Uncomment for additional combinations if needed
    # f"{bio_col}-{batch_col}": {"shape_col": bio_col, "color_col": batch_col},
    # f"{batch_col}-{batch_col}": {"shape_col": batch_col, "color_col": batch_col},
}

# --------------------------------------------------------------------------------------
# 2. Load Metadata and Define Categories
# --------------------------------------------------------------------------------------
# Load metadata before splits
metadata_all = pd.read_csv(glob.glob(data_path + "/*meta.csv")[0])

# Convert columns to categorical types
metadata_all['celltype'] = metadata_all['celltype'].astype('category')
metadata_all['batch'] = metadata_all['batch'].astype('category')

# Print the number of unique batches
print("Number of batches:", len(np.unique(metadata_all[batch_col])))

# Define One Hot Encoded (OHE) order for donor and celltype categories
seen_donor_ids = np.unique(metadata_all[batch_col]).tolist()
print("Ordered batches (donors):", seen_donor_ids)

celltype_ids = np.unique(metadata_all[bio_col]).tolist()

# --------------------------------------------------------------------------------------
# 3. Run All Folds
# --------------------------------------------------------------------------------------
# Run folds and compute clustering metrics
mean_scores = run_all_folds(
    Model=AE,
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
    model_type="ae",
    issparse=False,
    load_dense=False,
    shape_color_dict=shape_color_dict,
    sample_size=model_params_dict["sample_size"]
)

# --------------------------------------------------------------------------------------
# 4. Save Configuration File
# --------------------------------------------------------------------------------------
# Define the destination path for the configuration file
destination_path = os.path.join(saved_models_base, run_name, 'model_config.py')

print("\nCopying config.py file to:", destination_path)

# Copy the configuration file to the destination
shutil.copy(source_file, destination_path)
