#!/usr/bin/env python
"""
scVI Training & Cross‑Validation Pipeline

"""

# ======================================================
# 0. Imports
# ======================================================
# --- Standard library
import os

import gc
import warnings
from typing import Dict, Any, List

# --- Third‑party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

#import torch
import tensorflow as tf
import anndata
from anndata import AnnData
from matplotlib.lines import Line2D


from scMEDAL_for_scRNAseq.comparables.comparables_utils import calculate_merge_scores, plot_rep 


from scMEDAL_for_scRNAseq.utils.utils import (

    get_split_paths,

)

from scMEDAL_for_scRNAseq.utils.model_train_utils import ModelManager, load_data  #, compute_scores



def _extract_and_store(ad: AnnData, data_type: str, model_params=None):
    """
    Save Scanorama latent, PCA latent, and batch-corrected counts ("recon") for ad.
    """
    latent_path = model_params.latent_path
    pca_path = os.path.join(latent_path, 'pca')
    os.makedirs(pca_path, exist_ok=True)

    latent_key = f"{model_params.encoder_latent_name}_{data_type}"
    
    # Explicitly add latent to obsm!
    latent = ad.obsm["X_scanorama"]
    ad.obsm[latent_key] = latent  # <-- CRITICAL STEP
    np.save(os.path.join(latent_path, f"{latent_key}.npy"), latent)

    pca_latent_key = f"pca_latent_{latent.shape[1]}_{data_type}"
    ad.obsm[pca_latent_key] = ad.obsm["X_pca"]
    np.save(os.path.join(pca_path, f"{pca_latent_key}.npy"), ad.obsm["X_pca"])

    recon = ad.X
    np.save(os.path.join(latent_path, f"recon_{data_type}.npy"), recon)

def run_scanorama_pipeline(
    Model,
    input_path_dict: Dict[str, str],
    build_model_dict: Dict[str, Any],
    compile_dict: Dict[str, Any],
    model_params,
    save_model: bool,
    batch_col: str,
    bio_col: str,
    batch_col_categories=None,
    bio_col_categories=None,
    return_scores=False,
    return_adata_dict=False,
    return_trained_model=False,
    model_type="scanorama",
    issparse=False,
    load_dense=True,
    shape_color_dict=None,
    sample_size=None,
    return_history=False,
    n_batch2plot=None,
    seed=5,
    plot_params=None,
):
    import scanpy.external as sce

    # 1. Load splits
    adata_dict = load_data(
        input_path_dict,
        eval_test=model_params.eval_test,
        scaling=model_params.scaling,
        issparse=issparse,
        load_dense=load_dense,
    )

    # Add unique indexing to preserve original cell order
    for split_name, adata in adata_dict.items():
        adata.obs["original_index"] = [f"{split_name}_{i}" for i in range(adata.n_obs)]

    split_order, split_data = [], []
    for split_name in ["train", "val", "test"]:
        if split_name in adata_dict:
            split_order.append(split_name)
            split_data.append(adata_dict[split_name].copy())

    # Remove any pre-existing "split" column
    for ad in split_data:
        if "split" in ad.obs.columns:
            ad.obs.drop(columns=["split"], inplace=True)

    # 2. Concatenate splits
    adata_full = anndata.concat(
        split_data,
        label="split",
        keys=split_order,
        index_unique=None
    )

    adata_full.obs[batch_col] = adata_full.obs[batch_col].astype(str)
    adata_full.obs_names_make_unique()

    # 3. Sort to ensure contiguous batches (required by Scanorama)
    adata_full = adata_full[adata_full.obs[batch_col].argsort()].copy()

    # 4. Run PCA and Scanorama integration
    sc.pp.pca(adata_full, n_comps=model_params.n_latent_dims)
    sce.pp.scanorama_integrate(adata_full, key=batch_col)

    # 5. Restore original cell ordering based on saved index
    adata_dict_reordered = {}
    for split_name in split_order:
        split_ad = adata_full[adata_full.obs["split"] == split_name].copy()

        # Set original index as obs_names
        split_ad.obs_names = split_ad.obs["original_index"]

        # Reorder explicitly based on the original indices
        original_order = [f"{split_name}_{i}" for i in range(split_ad.n_obs)]
        split_ad = split_ad[original_order].copy()

        adata_dict_reordered[split_name] = split_ad

    # 6. Save latent spaces
    os.makedirs(model_params.latent_path, exist_ok=True)
    for split, ad in adata_dict_reordered.items():
        _extract_and_store(ad, split, model_params)

    # 7. Calculate and save clustering scores
    df_scores_dict = None
    if return_scores:
        df_scores_dict = {}
        for split, ad in adata_dict_reordered.items():
            latent_list = [
                f"{model_params.encoder_latent_name}_{split}",
                f"pca_latent_{ad.obsm['X_scanorama'].shape[1]}_{split}"
            ]
            df_scores = calculate_merge_scores(
                latent_list=latent_list,
                adata=ad,
                labels=[batch_col, bio_col],
                sample_size=sample_size,
            )
            df_scores.to_csv(
                os.path.join(model_params.latent_path, f"scores_{split}_samplesize-{sample_size}.csv")
            )
            df_scores_dict[split] = df_scores

    # 8. Generate plots if requested
    if shape_color_dict:
        from pathlib import Path
        Path(model_params.plots_path).mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)

        for split_name, ad_full in adata_dict_reordered.items():
            for combo, cfg in shape_color_dict.items():
                pp = plot_params.copy()
                pp["outpath"] = model_params.plots_path
                pp["file_name"] = f"{combo}_X_scanorama_{split_name}"
                pp["shape_col"] = cfg["shape_col"]
                pp["color_col"] = cfg["color_col"]

                ad = ad_full
                if n_batch2plot is not None:
                    col_vals = ad_full.obs[pp["color_col"]].unique()
                    if len(col_vals) > n_batch2plot:
                        chosen = rng.choice(col_vals, size=n_batch2plot, replace=False)
                        ad = ad_full[ad_full.obs[pp["color_col"]].isin(chosen)].copy()

                plot_rep(
                    adata=ad,
                    use_rep="X_scanorama",
                    **pp,
                )

    gc.collect()

    results = {}
    if return_trained_model:
        results["model"] = None
    if return_scores:
        results["scores"] = df_scores_dict
    if return_adata_dict:
        results["adata"] = adata_dict_reordered
    if return_history:
        results["history"] = pd.DataFrame()

    return results


# ======================================================
# 2. k‑fold cross‑validation wrapper
# ======================================================

def run_scanorama_across_folds(
    input_base_path: str,
    out_base_paths_dict: Dict[str, str],
    folds_list: List[int],
    run_name: str,
    model_params_dict: Dict[str, Any],
    build_model_dict: Dict[str, Any],
    compile_dict: Dict[str, Any],  # ignored by scVI
    save_model: bool,
    batch_col: str,
    bio_col: str,
    batch_col_categories: List[str] | None = None,
    bio_col_categories: List[str] | None = None,
    model_type: str = "scanorama",
    issparse: bool = True,
    load_dense: bool = True,
    shape_color_dict: Dict[str, Any] | None = None,
    sample_size: int | None = None,
    return_scores_temp: bool = False,
    n_batch2plot: int | None = None,
    seed: int = 5,  # for plots only
    plot_params: dict | None = None,
    ):
    """Run scVI across multiple folds, collecting outputs."""
    all_scores, all_folds_adata, all_folds_model_params = {}, {}, {}
    all_history_df = pd.DataFrame()

    all_scores_per_fold = {}     # new scores accumulator

    for intFold in folds_list:
        print(f"\nRunning Fold {intFold}\n".center(60, "-"))

        # 1) Configure paths & parameters for this fold
        model_manager = ModelManager(
            params_dict=model_params_dict,
            base_paths_dict=out_base_paths_dict,
            run_name=run_name,
            save_model=save_model,
            use_kfolds=True,
            kfold=intFold,
        )
        model_params = model_manager.params



        # 2) Build dictionary of .h5ad paths
        input_path_dict = get_split_paths(base_path=input_base_path, fold=intFold)
        print("input_path_dict:", input_path_dict)

        # 3) Execute single‑fold pipeline
        fold_results = run_scanorama_pipeline(
            Model                   = None,
            input_path_dict         = input_path_dict,
            build_model_dict        = build_model_dict,
            compile_dict            = compile_dict,   # ignored safely
            model_params            = model_params,
            save_model              = save_model,
            batch_col               = batch_col,
            bio_col                 = bio_col,
            batch_col_categories    = batch_col_categories,
            bio_col_categories      = bio_col_categories,
            return_scores           = return_scores_temp,
            return_adata_dict       = True,
            return_trained_model    = False,
            model_type              = model_type,
            issparse                = issparse,
            load_dense              = load_dense,
            shape_color_dict        = shape_color_dict,
            sample_size             = sample_size,
            return_history          = True,
            n_batch2plot = n_batch2plot,
            seed = seed,
            plot_params = plot_params
        )

        # 4) Accumulate outputs
        all_folds_adata[intFold]        = fold_results["adata"]
        all_folds_model_params[intFold] = model_params
        # save scores
        if return_scores_temp:
            all_scores_per_fold[intFold] = fold_results["scores"]   # dict(train/val/test)



    # --------------------------------------------------
    # 5) Aggregate & write CSVs  (train / val / test)
    # --------------------------------------------------
    if return_scores_temp and all_scores_per_fold:
        os.makedirs(model_params.latent_path_main, exist_ok=True)   # one dir per run
        for data_split in ["train", "val", "test"]:
            if data_split == "test" and not model_params.eval_test:
                continue                                           # skip if no test set

            rows = []
            for fold_idx, split_dict in all_scores_per_fold.items():
                latent_key = f"{model_params.encoder_latent_name}_{data_split}"

                # )
                scores_df = split_dict[data_split].copy()          # keep original structure
                scores_df["fold"] = fold_idx
                scores_df["dataset_type"] = latent_key       # latent name ? column
                rows.append(scores_df)    
                
            big_df = pd.concat(rows, ignore_index=True)

            # ---- save every?fold detail
            big_df.to_csv(
                os.path.join(model_params.latent_path_main,
                            f"all_scores_{data_split}_samplesize-{sample_size}.csv"),
                index=False
            )


            numeric_cols = big_df.select_dtypes(include="number").columns



            #)
            # Compute mean scores grouped by 'dataset_type', then remove 'dataset_type' after grouping
            mean_scores = (
                big_df.groupby("dataset_type")[numeric_cols]
                .mean()
                .reset_index(drop=True)  # Dropping 'dataset_type' here after calculating mean
            )

            # Calculate standard deviation (sample standard deviation, ddof=1)
            std_scores = (
                big_df.groupby("dataset_type")[numeric_cols]
                .std(ddof=1)
                .reset_index(drop=True)
            )

            # Calculate standard error of the mean (SEM)
            sem_scores = std_scores / (len(folds_list) ** 0.5)



            # Add summary rows manually for 'mean', 'std', 'sem'
            summary_df = mean_scores.copy()
            summary_df.loc['mean'] = mean_scores.iloc[0]  # Add mean explicitly for clarity
            summary_df.loc['std'] = std_scores.iloc[0]  # Assuming single dataset_type group
            summary_df.loc['sem'] = sem_scores.iloc[0]
            

            summary_df = summary_df.T
            summary_df = summary_df.loc[:, summary_df.columns != 0]  # Drop any column with name 0

            # Save summary DataFrame
            summary_df.to_csv(
                os.path.join(
                    model_params.latent_path_main,
                    f"mean_scores_{data_split}_samplesize-{sample_size}.csv",
                ),
                header=True
            )  # Columns will be: mean, std, sem

        




    return {
        "all_scores_per_fold":all_scores_per_fold,
        #"mean_scores": merged_scores,
        "all_adata":   all_folds_adata,
        "all_params":  all_folds_model_params,
        "history":     all_history_df,
        #"model_params_dict":model_params_dict
    }





