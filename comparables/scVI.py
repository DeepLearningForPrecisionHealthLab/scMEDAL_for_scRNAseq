
"""
scVI Training & CrossValidation Pipeline
"""

# ======================================================
# 0. Imports
# ======================================================
# --- Standard library
import os

import gc
import warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scvi
import torch
import tensorflow as tf
from anndata import AnnData
from matplotlib.lines import Line2D


from scMEDAL_for_scRNAseq.utils.utils import (get_split_paths)
from scMEDAL_for_scRNAseq.comparables.comparables_utils import calculate_merge_scores, plot_rep 
from scMEDAL_for_scRNAseq.utils.model_train_utils import ModelManager, load_data  #, compute_scores

# Silence potential tensorflow warnings for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning)
print("tf_version", tf.__version__)




# ======================================================
# 1. scVI single‑fold routine
# ======================================================

def run_scvi_pipeline(
    Model,
    input_path_dict: Dict[str, str],
    build_model_dict: Dict[str, Any],
    compile_dict: Dict[str, Any],  # kept for API compatibility (unused)
    model_params,
    save_model: bool,
    batch_col: str,
    bio_col: str,
    batch_col_categories: List[str] | None = None,
    bio_col_categories: List[str] | None = None,
    return_scores: bool = False,
    return_adata_dict: bool = False,
    return_trained_model: bool = False,
    model_type: str = "scvi",  # placeholder – no effect on SCVI
    issparse: bool = False,
    load_dense: bool = True,
    shape_color_dict: Dict[str, Any] | None = None,
    sample_size: int | None = None,
    return_history: bool = False,
    n_batch2plot: int | None = None,
    seed: int = 5,  # for plots only
    plot_params: dict | None = None,
    ):
    """Train an scVI model on a single data split and extract results."""
    # --------------------------------------------------
    # 1) Load data
    # --------------------------------------------------
    adata_dict = load_data(
        input_path_dict,
        eval_test=model_params.eval_test,
        scaling=model_params.scaling,
        issparse=issparse,
        load_dense=load_dense,
    )
    adata_train = adata_dict["train"]
    adata_val   = adata_dict["val"]
    adata_test  = adata_dict.get("test") if model_params.eval_test else None


    # --------------------------------------------------
    # 2) Setup + train scVI
    # --------------------------------------------------
    # register BOTH splits with the same keys
    scvi.model.SCVI.setup_anndata(adata_train, batch_key=batch_col)
    scvi.model.SCVI.setup_anndata(adata_val,   batch_key=batch_col)     

    model = Model(
        adata_train,
        n_latent       = model_params.n_latent_dims,
        n_layers       = model_params.n_layers,
        n_hidden       = model_params.n_hidden,
        gene_likelihood= model_params.gene_likelihood,
        dispersion     = model_params.dispersion,
    )

    model.train(
        max_epochs              = model_params.epochs,
        batch_size              = model_params.batch_size,
        early_stopping          = True,
        check_val_every_n_epoch = model_params.patience,
        validation_size=0.1 # i cannot pass my own splits
    )


    # Save model
    if save_model:
        model.save(model_params.model_path, overwrite=True)

    # --------------------------------------------------
    # 3) Training history
    # --------------------------------------------------
    history_raw = model.history
    try:
        dfs = []
        for k, v in history_raw.items():
            if isinstance(v, pd.Series):
                dfs.append(v.to_frame(name=k))
            elif isinstance(v, pd.DataFrame):
                dfs.append(v.rename(columns={v.columns[0]: k}))
        history_df = pd.concat(dfs, axis=1).reset_index()
    except Exception:
        history_df = pd.DataFrame(history_raw)

    os.makedirs(model_params.latent_path, exist_ok=True)
    history_df.to_csv(os.path.join(model_params.latent_path, "history_scvi.csv"), index=False)

    # Save a PNG of the loss curves for this fold/split
    plot_scvi_history(
        history_df,
        save_path=os.path.join(model_params.plots_path, "scvi_history.png"),  # one per pipeline call
        show=False,  # keep silent inside the CV loop
    )

    # --------------------------------------------------
    # 4) Latent representations & reconstructions
    # --------------------------------------------------

    def _extract_and_store(ad: AnnData, data_type: str) -> None:
        scvi.model.SCVI.setup_anndata(ad, batch_key=batch_col)
        latent = model.get_latent_representation(ad)
        
        latent_key = f"{model_params.encoder_latent_name}_{data_type}"
        ad.obsm["X_scvi"] = latent
        np.save(os.path.join(model_params.latent_path, f"{latent_key}.npy"), latent)

        recon = model.get_normalized_expression(ad, return_numpy=True)
        np.save(os.path.join(model_params.latent_path, f"recon_{data_type}.npy"), recon)

    _extract_and_store(adata_train, "train")
    _extract_and_store(adata_val, "val")
    if adata_test is not None:
        _extract_and_store(adata_test, "test")

    # --------------------------------------------------
    # 5) Optional: clustering / batch metrics
    # --------------------------------------------------
    df_scores_dict = None
    if return_scores:
        df_scores_dict = {}

        # folder to save score CSVs
        os.makedirs(model_params.plots_path, exist_ok=True)

        for split, ad in adata_dict.items():
            print(f"Computing clustering scores for {split}")

            latent_list = list(ad.obsm.keys())
            # for latent_name in adata_subset.obsm.keys():
            print(f"\n\nProcessing clustering scores {latent_list} for dataset {split}..")
            df_scores = calculate_merge_scores(latent_list=latent_list, 
                                                        adata=ad, 
                                                        labels=[batch_col, bio_col], 
                                                        sample_size=sample_size)

            # save for inspection
            df_scores.to_csv(
                os.path.join(model_params.latent_path, f"scores_{split}_samplesize-{sample_size}.csv")
            )
            df_scores_dict[split] = df_scores

    # --------------------------------------------------
    # 6) Quick latent plots  PNG
    # --------------------------------------------------
    if shape_color_dict:
        from pathlib import Path
        Path(model_params.plots_path).mkdir(parents=True, exist_ok=True)


        rng = np.random.default_rng(seed) if seed is not None else np.random

        for split_name, ad_full in adata_dict.items():
            for combo, cfg in shape_color_dict.items():
                # 1) build kwargs for this plot
                pp = plot_params.copy()
                pp["outpath"]   = model_params.plots_path
                pp["file_name"] = f"{combo}_X_scvi_{split_name}"
                pp["shape_col"] = cfg["shape_col"]   # ? keep originals
                pp["color_col"] = cfg["color_col"]

                # 2) OPTIONAL colour?category subsampling
                ad = ad_full
                if n_batch2plot is not None:
                    col_vals = ad_full.obs[pp["color_col"]].unique()
                    if len(col_vals) > n_batch2plot:
                        chosen = rng.choice(col_vals,
                                            size=n_batch2plot,
                                            replace=False)
                        ad = ad_full[ad_full.obs[pp["color_col"]].isin(chosen)].copy()

                # 3) plot
                plot_rep(
                    adata   = ad,
                    use_rep = "X_scvi",
                    **pp,
                )
                
    gc.collect()

    # --------------------------------------------------
    # 6) Return requested objects
    # --------------------------------------------------
    results = {}
    if return_trained_model:
        results["model"] = model
    if return_scores:
        results["scores"] = df_scores_dict
    if return_adata_dict:
        results["adata"] = adata_dict
    if return_history:
        results["history"] = history_df

    return results


# ======================================================
# 2. k‑fold cross‑validation wrapper
# ======================================================

def run_scvi_across_folds(
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
    model_type: str = "scvi",
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
        fold_results = run_scvi_pipeline(
            Model                   = scvi.model.SCVI,
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
            
            # Transpose and drop column named '0' if it exists
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


# ======================================================
# 3. Visualisation utilities
# ======================================================



def plot_scvi_history(history_df: pd.DataFrame, save_path: str | None = None, show: bool = True) -> None:
    """Visualise training losses & metrics stored in *history_df*."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()

    if {"train_loss_step", "validation_loss"}.issubset(history_df.columns):
        ax[0].plot(history_df["epoch"], history_df["train_loss_step"], label="Train Loss")
        ax[0].plot(history_df["epoch"], history_df["validation_loss"], label="Validation Loss")
        ax[0].set_title("Total Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

    if "reconstruction_loss_validation" in history_df:
        ax[1].plot(history_df["epoch"], history_df["reconstruction_loss_validation"], label="Reconstruction Loss (Val)")
        ax[1].set_title("Reconstruction Loss (Validation)")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend()

    if "kl_local_validation" in history_df:
        ax[2].plot(history_df["epoch"], history_df["kl_local_validation"], label="KL Local (Val)")
        ax[2].set_title("KL Local (Validation)")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("KL")
        ax[2].legend()

    if "kl_global_validation" in history_df:
        ax[3].plot(history_df["epoch"], history_df["kl_global_validation"], label="KL Global (Val)")
        ax[3].set_title("KL Global (Validation)")
        ax[3].set_xlabel("Epoch")
        ax[3].set_ylabel("KL")
        ax[3].legend()

    plt.tight_layout()

    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:                     # ← only create if not empty
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved history plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# def postprocess_cv_results(cv_results: Dict[str, Any], shape_color_dict: Dict[str, Any], plot_params: Dict[str, Any]) -> None:
#     """Plot training history and latent spaces from *cv_results*."""
#     # 1) History plot
#     plot_scvi_history(
#         cv_results["history"],
#         save_path=os.path.join(plot_params["outpath"], "scvi_history.png"),
#         show=True,
#     )

#     # 2) Latent space visualisations
#     for fold, adata_splits in cv_results["all_adata"].items():
#         print(f"\nPlotting latent space for fold {fold}")
#         for split_name, adata in adata_splits.items():
#             print(f"  Split: {split_name}")
#             for combo_name, combo_params in shape_color_dict.items():
#                 file_name = f"{combo_name}_X_scvi_fold{fold}_{split_name}"
#                 plot_rep(
#                     adata=adata,
#                     use_rep="X_scvi",
#                     shape_col=combo_params["shape_col"],
#                     color_col=combo_params["color_col"],
#                     file_name=file_name,
#                     **plot_params,
#                 )

                
#     print("Finished all plots")



