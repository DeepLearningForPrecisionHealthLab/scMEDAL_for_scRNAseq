import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
import shutil

from scMEDAL.utils.model_train_utils import run_all_folds
from model_config import *  # Import all model configurations and parameters
from scMEDAL.models.scMEDAL import DomainEnhancingAutoencoderClassifier

print(f"TensorFlow version: {tf.__version__}")

# --------------------------------------------------------------------------------------
# 0. Define Fold Parameters
# --------------------------------------------------------------------------------------
folds_list = list(range(1, 6))  # Define folds (10 total; using 5 for testing)

# --------------------------------------------------------------------------------------
# 1. Define Batch and Bio Columns
# --------------------------------------------------------------------------------------
batch_col = model_params_dict['batch_col']
bio_col = model_params_dict['bio_col']

# Dictionary for plotting latent spaces
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
# Locate and load the metadata file
metadata_path = glob.glob(os.path.join(data_path, "*meta.csv"))[0]
metadata_all = pd.read_csv(metadata_path)

# Ensure required columns are treated as categorical
metadata_all['celltype'] = metadata_all['celltype'].astype('category')
metadata_all['batch'] = metadata_all['batch'].astype('category')

# Print the number of unique batches
n_batches = len(metadata_all[batch_col].unique())
print(f"Number of batches: {n_batches}")

# Define unique categories for batches and cell types
seen_donor_ids = metadata_all[batch_col].unique().tolist()
print(f"Ordered batches (donors): {seen_donor_ids}")

celltype_ids = metadata_all[bio_col].unique().tolist()

# --------------------------------------------------------------------------------------
# 3. Run All Folds
# --------------------------------------------------------------------------------------
# Execute model training and evaluation across folds
mean_scores = run_all_folds(
    Model=DomainEnhancingAutoencoderClassifier,
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
    model_type="ae_re",
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

# Ensure the destination directory exists
os.makedirs(os.path.dirname(destination_path), exist_ok=True)

# Copy the configuration file to the specified destination
print(f"\nCopying config.py file to: {destination_path}")
shutil.copy(source_file, destination_path)
