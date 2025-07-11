# Setup experiment in paths_config.py
# Add the path to the configuration file
# Path to paths_config: /scMEDAL_for_scRNAseq/Experiments/HealthyHeart/paths_config.py
import sys
sys.path.append("../../../")
from paths_config import run_names_dict, results_path_dict, compare_models_path

import os
import numpy as np

from scMEDAL.utils.compare_results_utils import (
    aggregate_paths,
    read_and_aggregate_scores,
    filter_min_max_silhouette_scores,
    process_all_results,
    process_confidence_intervals,
)

"""
This script reorganizes clustering scores into a single table. 
It also adds 95% confidence intervals (CI) to the results.
Environment: run_models_env
"""

# --------------------------------------------------------------------------------------
# 1. Define dataset type and output directory
# --------------------------------------------------------------------------------------
# Set the dataset type: Use 'val' for development or 'test' for final results
dataset_type = 'val'

# Define output directory for results
out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])

# Ensure the directory exists
if not os.path.exists(out_name):
    os.makedirs(out_name)
print(f"Directory created or already exists: {out_name}")

# --------------------------------------------------------------------------------------
# 2. Get paths for mean and all scores
# --------------------------------------------------------------------------------------
# Aggregate paths for mean scores
df_all_paths = aggregate_paths(results_path_dict, pattern=f'mean_scores_{dataset_type}_samplesize')
print("\n\nMean scores paths:")
print(df_all_paths.head())

# Aggregate paths for all scores
df_all_paths_allscores = aggregate_paths(results_path_dict, pattern=f'all_scores_{dataset_type}_samplesize')
print("\nAll scores paths:")
print(df_all_paths_allscores)

print("\nColumns in all scores paths:")
print(df_all_paths_allscores.columns)

# --------------------------------------------------------------------------------------
# 3. Read all scores and save to a CSV file
# --------------------------------------------------------------------------------------
df_allscores = read_and_aggregate_scores(df_all_paths_allscores)
print("\nAggregated scores DataFrame:")
print(df_allscores)

# Save all scores to CSV
df_allscores.to_csv(os.path.join(out_name, f"{dataset_type}_allscores.csv"))

# --------------------------------------------------------------------------------------
# 4. Filter minimum and maximum silhouette scores for the batch
# --------------------------------------------------------------------------------------
for sample_size in np.unique(df_allscores["sample_size"]):
    df_min_silhouette, df_max_silhouette = filter_min_max_silhouette_scores(df_allscores, batch_col="batch")
    
    # Save the filtered results to CSV
    df_min_silhouette.to_csv(os.path.join(out_name, f"{dataset_type}_scores_{sample_size}_min_silhouette_batch.csv"))
    df_max_silhouette.to_csv(os.path.join(out_name, f"{dataset_type}_scores_{sample_size}_max_silhouette_batch.csv"))

    # Print the resulting DataFrames
    print("DataFrame with minimum silhouette scores:")
    print(df_min_silhouette)

    print("\nDataFrame with maximum silhouette scores:")
    print(df_max_silhouette)

# --------------------------------------------------------------------------------------
# 5. Process results depending on the model configuration
# --------------------------------------------------------------------------------------
# Define how to process results for each model
# If `get_pca=True` for any model in the pipeline, use "preprocess_results_model_pca_format".
# Otherwise, use "process_single_model_format" or leave empty if no processing is required.
# NOTE: Keys in models2process_dict must match those in run_names_dict from paths_config.py.
# Update the run names: run_names_dict={
#                "AEC_reconloss100_classloss0.1": "scMEDAL-FEC_run_name"}
models2process_dict = {
    "AEC_reconloss100_classloss0.1": "preprocess_results_model_pca_format",
    "AEC_reconloss1_classloss0.1": "preprocess_results_model_pca_format",
    "AEC_reconloss10_classloss0.1": "preprocess_results_model_pca_format",
}



print("\n\nGet df_sample_size")
df_sample_size = process_all_results(df_all_paths, models2process_dict, out_name, dataset_type)

# --------------------------------------------------------------------------------------
# 6. Calculate 95% confidence intervals (CI) for results
# --------------------------------------------------------------------------------------
print("\n\nGet 95% CI")
for sample_size, df_all in df_sample_size.items():
    df_mean_ci_results = process_confidence_intervals(df_all, out_name, dataset_type, sample_size)
    print(f"Sample size: {sample_size}\nConfidence interval results:")
    print(df_mean_ci_results)