

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

# Create prefixes for downstream plotting and analysis
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

umap_path = os.path.join(out_name, "umap_20batches")
if not os.path.exists(umap_path):
    os.makedirs(umap_path)

# Specify folds (splits) for filtering data you want to plot
filter_folds = {
    "AE_50dims": 2,
    "AEC_50dims": 2,
    "scMEDAL-FEC_50dims": 2,
    "scMEDAL-FE_50dims": 2,
    "scMEDAL-RE_50dims": 2,
    "scVI_50dims":2,
    "scANVI_50dims":2,
    "scanorama_50dims":2,
    "harmony_50dims":2,
    "SAUCIE_50dims":2,
}


            
            

# Filter data to include only specific models and splits for "train" data
filtered_df = filter_models_by_type_and_split(df, filter_folds, Type='train')

print("\n\nfiltered_df",filtered_df)

# Common plotting parameters
# Add colors if you decide to plot more batches
plot_params = {
    "markers": ["x", "+", "<", "h", "s", ".", 'o', 's', '^', '*', '1', '8', 'p',
                'P', 'D', '|', 0, ',', 'd', 2],
    "shape_col": "celltype",
    "color_col": "celltype",
    "use_rep": "X_umap",
    "clustering_scores": None,
    "save_fig": True,
    "showplot": False,
    "palette_choice": [
        '#e6194b',  # Red
        '#3cb44b',  # Green
        '#ffe119',  # Yellow
        '#4363d8',  # Blue
        '#f58231',  # Orange
        '#911eb4',  # Purple
        '#46f0f0',  # Cyan
        '#f032e6',  # Magenta
        '#000000',  # Black
        '#fabebe',  # Light pink
        '#008080',  # Teal
        '#e6beff',  # Lavender
        '#9a6324',  # Brown
        '#d2f53c',  # Lime
        '#ff69b4',  # Hot pink
        '#000080',  # Navy
        '#800000',  # Maroon
        '#808000',  # Olive
        '#800080',  # Dark purple
        '#808080'   # Gray
    ]
}

print("Computing UMAPs...")





processor = DimensionalityReductionProcessor(
    df=filtered_df,
    output_path=umap_path,
    plot_params=plot_params,
#    sample_size=None,          # take every cell
    sample_size=None,          # take every cell
    n_neighbors=15,
    scaling="min_max",
    n_batches_sample=20,
    batch_col="batch",         # subsample 19 technical batches
    plot_tsne=False,
    n_pca_components=50,
    rng_seed=5,                # reproducible RNG seed
    extra_color_cols=["DonorID","TissueDetail","protocol"]  #Make sure this columns exist in the data
    # min_dist keeps its default 0.5
)

# generate UMAPs
processor.get_dimensionality_reduction_plots(
    process_allbatches=False,  # only batch-filtered plots
    issparse=True,             # inputs are already dense
)