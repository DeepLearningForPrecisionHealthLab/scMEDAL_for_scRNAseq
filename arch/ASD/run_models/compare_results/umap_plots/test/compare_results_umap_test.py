"""
Script to load latent and input paths, merge them, filter models, and compute
UMAP projections for scMEDAL experiments.

Configuration paths are defined in `paths_config.py`.
"""

import os
import sys
import pandas as pd

# Extend Python path to find project-specific modules
sys.path.append("../../../")

from paths_config import (
    run_names_dict,
    results_path_dict,
    compare_models_path,
    input_base_path
)

from scMEDAL.utils.compare_results_utils import (
    get_input_paths_df,
    get_latent_paths_df,
    filter_models_by_type_and_split,
    DimensionalityReductionProcessor
)

# Load latent and input paths
df_latent = get_latent_paths_df(results_path_dict)
df_inputs = get_input_paths_df(input_base_path)

print("\ndf_latent columns:", df_latent.columns, "\ndf_latent:\n", df_latent)
print("\ndf_inputs columns:", df_inputs.columns, "\ndf_inputs:\n", df_inputs)

# Merge latent and input paths by "Split" and "Type"
df = pd.merge(df_latent, df_inputs, on=["Split", "Type"], how="left")

df["latent_prefix"] = [
    f"{model}_latent_{Type}_{Split}"
    for model, Split, Type in zip(df["Key"], df["Split"], df["Type"])
]
df["input_prefix"] = [
    f"{Type}_{Split}"
    for Split, Type in zip(df["Split"], df["Type"])
]

print("Reading paths, df head:\n", df.head(5))

# Define output directories
out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])
if not os.path.exists(out_name):
    os.makedirs(out_name)

umap_path = os.path.join(out_name, "umap_31batches_seed_5")
if not os.path.exists(umap_path):
    os.makedirs(umap_path)


# Specify folds (splits) for filtering data you want to plot
filter_folds = {
    "AE_50dims": 2,
    # "AEC": 2,
    # "scMEDAL-FEC": 2,
    "scMEDAL-FE_50dims": 2,
    "scMEDAL-RE_50dims": 2
}

# Filter data to include only specific models and splits for "train" data
filtered_df = filter_models_by_type_and_split(df, filter_folds, Type='train')
print(filtered_df.columns)
print(filtered_df.loc[0,"LatentPath"],"\n")
