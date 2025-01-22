
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import shutil
import os
import glob

from scMEDAL.utils.model_train_utils import run_all_folds
from model_config import *  # Import all model configurations and parameters
from scMEDAL.models.scMEDAL import DomainAdversarialAE

print(f"TensorFlow version: {tf.__version__}")

# --------------------------------------------------------------------------------------
# 0. Define Fold Parameters
# --------------------------------------------------------------------------------------
# Specify the number of folds to use (e.g., for cross-validation)
folds_list = list(range(1, 6))  # Using 5 folds for testing

# --------------------------------------------------------------------------------------
# 1. Define Batch and Bio Columns
# --------------------------------------------------------------------------------------
# Columns in the metadata that specify batch and biological condition
batch_col = model_params_dict['batch_col']
bio_col = model_params_dict['bio_col']

# Dictionary to map how shapes and colors are assigned in latent space plots
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
# Locate the metadata file dynamically
metadata_path = glob.glob(data_path + "/*meta.csv")[0]
metadata_all = pd.read_csv(metadata_path)

# Ensure the necessary columns are treated as categorical
metadata_all['celltype'] = metadata_all['celltype'].astype('category')
metadata_all['batch'] = metadata_all['batch'].astype('category')

# Number of unique batches and their order
n_batches = len(np.unique(metadata_all[batch_col]))
print(f"Number of batches: {n_batches}")

# Define the unique categories for batches and cell types
seen_donor_ids = np.unique(metadata_all[batch_col]).tolist()
print(f"Ordered batches (donors): {seen_donor_ids}")

celltype_ids = np.unique(metadata_all[bio_col]).tolist()

# --------------------------------------------------------------------------------------
# 3. Run All Folds
# --------------------------------------------------------------------------------------
# Execute the model training and evaluation across all folds
mean_scores = run_all_folds(
    Model=DomainAdversarialAE,  # Model class to use
    input_base_path=input_base_path,  # Path to input data
    out_base_paths_dict=base_paths_dict,  # Output paths for results
    folds_list=folds_list,  # List of folds to run
    run_name=run_name,  # Unique name for this run
    model_params_dict=model_params_dict,  # Model parameters
    build_model_dict=build_model_dict,  # Model architecture
    compile_dict=compile_dict,  # Compilation settings
    save_model=save_model,  # Whether to save the trained model
    batch_col=batch_col,  # Batch column in the data
    bio_col=bio_col,  # Biological condition column
    batch_col_categories=seen_donor_ids,  # Categories for batch column
    bio_col_categories=celltype_ids,  # Categories for bio column
    model_type="ae_da",  # Model type
    issparse=False,  # Specify if input data is sparse
    load_dense=False,  # Specify if data should be loaded as dense
    shape_color_dict=shape_color_dict,  # Plotting configurations
    sample_size=model_params_dict["sample_size"]  # Sample size for evaluation
)

# --------------------------------------------------------------------------------------
# 4. Save Configuration File
# --------------------------------------------------------------------------------------
# Define the path to save the configuration file
destination_path = os.path.join(saved_models_base, run_name, 'model_config.py')

# Ensure the directory exists before copying the configuration file
os.makedirs(os.path.dirname(destination_path), exist_ok=True)

# Copy the configuration file to the output folder
print(f"\nCopying config.py file to: {destination_path}")
shutil.copy(source_file, destination_path)
