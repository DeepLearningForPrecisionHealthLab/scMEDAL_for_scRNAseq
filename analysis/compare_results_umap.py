"""
Script to load latent and input paths, merge them, filter models, and compute
UMAP projections for scMEDAL experiments.

Configuration paths are defined in `paths_config.py`.
"""

import os
import sys
import pandas as pd

from configs.configs import PlotConfigs
from typing import Optional, List

from utils.compare_results_utils import (
    get_input_paths_df,
    get_latent_paths_df,
    filter_models_by_type_and_split,
    DimensionalityReductionProcessor
)

def get_umap(
        run_names_dict,
        results_path_dict,
        compare_models_path,
        input_base_path,
        analysis_name,
        n_pca_components:int=50,
        n_batches:int=19,
        n_neighbors:int=15,
        rng_seed:int=5,
        scaling:str="min_max",
        models:Optional[List[str]]=None,
        types:Optional[List[str]]=None,
        splits:Optional[List[int]]=None,
        batch_col="batch",
        shape_col:str="celltype",
        color_col:str="celltype",
        use_rep:str="X_umap",
        clustering_scores=None, 
        issparse=False,
        extra_color_cols=["Patient_group"]
    ):
    
    # Define common plotting parameters
    plot_params = {
        "shape_col": shape_col,
        "color_col":color_col,
        "use_rep": use_rep,
        "clustering_scores": clustering_scores,
        **PlotConfigs()._asdict()
    }

    print("\plot configs",PlotConfigs()._asdict().items())

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
    out_name = os.path.join(compare_models_path, analysis_name)
    if not os.path.exists(out_name):
        os.makedirs(out_name)

    umap_path = os.path.join(out_name, f"umap_{n_batches}batches_seed_{rng_seed}")
    if not os.path.exists(umap_path):
        os.makedirs(umap_path)

    if models is None:
        models = list(run_names_dict.keys())  # Add all your models to this list
    if types is None:
        types = ["train", "test", "val"]
    if splits is None:
        splits = list(range(1,6))             # Get data from fold 2
 
    # Filter data to include only specific models and splits for "train" data
    processors = []
    for type in types:
        for split in splits:
            filter_folds = {mod:split for mod in models}
            filtered_df = filter_models_by_type_and_split(df, filter_folds, Type=type)

            print("Computing UMAP projections...")

            # Initialize dimensionality reduction processor
            processor = DimensionalityReductionProcessor(
                filtered_df,
                umap_path,
                plot_params,
                sample_size=None,
                n_neighbors=n_neighbors,
                scaling=scaling,
                n_batches_sample=n_batches,
                batch_col=batch_col,
                plot_tsne=False,
                n_pca_components=n_pca_components,
                rng_seed=rng_seed,
                extra_color_cols=extra_color_cols
            )

            print(f"{df.columns}")
            
            # Generate UMAP plots
            processor.get_dimensionality_reduction_plots(process_allbatches=False, issparse=issparse)
            processors.append(processor)
    return processors