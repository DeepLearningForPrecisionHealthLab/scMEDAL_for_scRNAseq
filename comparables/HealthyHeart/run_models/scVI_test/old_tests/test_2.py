#!/usr/bin/env python
"""
scVI Training & Cross‑Validation Pipeline

"""

# ======================================================
# 0. Imports
# ======================================================
# --- Standard library
import os
import sys
import glob
import gc
import warnings
from typing import Dict, Any, List

# --- Third‑party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scvi
import torch
import tensorflow as tf
import anndata
from anndata import AnnData
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scvi.dataloaders import DataSplitter

# --- Project‑specific modules
# NOTE: keep sys.path adjustments *before* relative imports
sys.path.append(
    "/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/HealthyHeart/run_models/scVI"
)
sys.path.append(
    "/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/HealthyHeart"
)

from model_config import *  # noqa: F403  (imports model_params_dict, build_model_dict, etc.)
from scMEDAL.utils.utils import (
    create_folder,
    read_adata,
    get_OHE,
    min_max_scaling,
    plot_rep,
    calculate_merge_scores,
    get_split_paths,
    calculate_zscores,
    get_clustering_scores_optimized,
)
from scMEDAL.utils.callbacks import ComputeLatentsCallback
from scMEDAL.utils.model_train_utils import ModelManager, load_data  #, compute_scores

# Silence potential tensorflow warnings for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning)
print("tf_version", tf.__version__)

import time

def get_clustering_scores_optimized(adata, use_rep, labels, sample_size=None):
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
    """
    Optimized version of getting clustering scores from an AnnData object with optional subsampling for all scores.

    Args:
    - adata: An AnnData object.
    - use_rep: Representation to use. Any key for .obsm is valid. If 'X', then adata.X is used.
    - labels: Labels for clustering. Any key for .obs is valid.
    - sample_size: Optional integer. If specified, subsamples this number of instances for score calculation.

    Returns:
    - A pandas DataFrame with Davies-Bouldin (DB), inverse DB, Calinski-Harabasz (CH), and Silhouette scores for each label.
    """
    
    start_time = time.time()
    print("Computing scores..")
    if sample_size is not None:
        print("sample_size:",sample_size)
    else:
        print("all cells used")
        
    print("\nDB: lower values --> better clustering")
    print("1/DB, CH, Silhouette: higher values --> better clustering")

    # Determine the data representation
    data_rep = adata.X if use_rep == "X" else adata.obsm[use_rep]

    dict_scores = {}
    time_per_score = {}

    # Subsample the data if a sample size is specified
    if sample_size and sample_size < data_rep.shape[0]:
        indices = np.random.choice(data_rep.shape[0], sample_size, replace=False)
        subsampled_data_rep = data_rep[indices]
    else:
        subsampled_data_rep = data_rep

    # Compute scores for each label and store in dict
    for label in labels:
        labels_array = adata.obs[label].to_numpy()
        if sample_size and sample_size < data_rep.shape[0]:
            subsampled_labels = labels_array[indices]
        else:
            subsampled_labels = labels_array

        score_start_time = time.time()
        db_score = davies_bouldin_score(subsampled_data_rep, subsampled_labels)
        db_time = time.time() - score_start_time

        score_start_time = time.time()
        ch_score = calinski_harabasz_score(subsampled_data_rep, subsampled_labels)
        ch_time = time.time() - score_start_time

        score_start_time = time.time()
        silhouette = silhouette_score(subsampled_data_rep, subsampled_labels)
        silhouette_time = time.time() - score_start_time

        dict_scores[label] = [db_score, 1/db_score, ch_score, silhouette]
        time_per_score[label] = [db_time, ch_time, silhouette_time]

    # Create DataFrame from dict
    df_scores = pd.DataFrame(dict_scores, index=['db', '1/db', 'ch', 'silhouette'])
    total_time = time.time() - start_time

    print(f"Total computation time: {total_time} seconds")
    for label, times in time_per_score.items():
        print(f"Time per score for {label} - DB: {times[0]:.4f}, CH: {times[1]:.4f}, Silhouette: {times[2]:.4f} seconds")

    return df_scores




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

    # 2)  CONCAT train?+?val  ? new_adata_train   (with split tags)

    adata_train.obs["split_cv"] = "train"          # outer?CV assignment
    adata_val.obs["split_cv"]   = "val"

    new_adata_train = anndata.concat(
        [adata_train, adata_val],
        join="inner",             # keep only shared genes (safer)
        merge="unique",           # keep identical .uns entries once
    )
    print("Combined object:", new_adata_train.shape)

    # ------------------------------------------------------------------
    # determine split_base from the first entry in input_path_dict
    # (guaranteed to be ??/splits/split_<k>? no matter how base_path
    #  was constructed or whether there are ?..? segments)
    # ------------------------------------------------------------------
    split_base = os.path.abspath(os.path.join(input_path_dict["train"], os.pardir))
    #            e.g. ?/splits/split_1

    concat_dir  = os.path.join(split_base, "train_val_scvi")
    os.makedirs(concat_dir, exist_ok=True)
    combined_h5 = os.path.join(concat_dir, "train_val_concat.h5ad")
    new_adata_train.write(combined_h5)
    print("Saved concatenated train + val to ?", combined_h5)

    # optional ? free memory
    del adata_train, adata_val
    gc.collect()


    # 3)  Set?up & TRAIN scVI on the combined object

    scvi.settings.seed = seed                       # reproducible split
    scvi.model.SCVI.setup_anndata(new_adata_train, batch_key=batch_col)

    model = Model(
        new_adata_train,
        n_latent        = model_params.n_latent_dims,
        n_layers        = model_params.n_layers,
        n_hidden        = model_params.n_hidden,
        gene_likelihood = model_params.gene_likelihood,
        dispersion      = model_params.dispersion,
    )

    model.train(
        max_epochs              = model_params.epochs,
        batch_size              = model_params.batch_size,
        early_stopping          = True,
        check_val_every_n_epoch = model_params.patience,
        validation_size         = 0.10,      # scvi carves this from new_adata_train
        train_size              = 0.90,
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
    def _extract_and_store(ad: AnnData, split_name: str) -> None:
        scvi.model.SCVI.setup_anndata(ad, batch_key=batch_col)
        latent = model.get_latent_representation(ad)
        ad.obsm["X_scvi"] = latent
        np.save(os.path.join(model_params.latent_path, f"z_{split_name}.npy"), latent)

        recon = model.get_normalized_expression(ad, return_numpy=True)
        np.save(os.path.join(model_params.latent_path, f"recon_{split_name}.npy"), recon)

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
            df_scores = get_clustering_scores_optimized(
                adata   = ad,
                use_rep = "X_scvi",
                labels  = [bio_col, batch_col],
                sample_size = sample_size,
            )
            # save for inspection
            df_scores.to_csv(
                os.path.join(model_params.latent_path, f"scores_{split}.csv")
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
    # 5) Aggregate scores across folds  ?  one CSV with
    #    all per?fold scores *and* one with their mean,
    #    saved separately for train / val / test
    # --------------------------------------------------
    #merged_scores = {}                          # keep per?split means in a dict



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
                rows.append(
                    reshape_scores(split_dict[data_split],
                                labels=[batch_col, bio_col],
                                fold=fold_idx,
                                dataset_type=data_split)
                )
            big_df = pd.concat(rows, ignore_index=True)

            # ---- save every?fold detail
            big_df.to_csv(
                os.path.join(model_params.latent_path_main,
                            f"all_scores_{data_split}_samplesize-{sample_size}.csv"),
                index=False
            )

            # ---- save fold?mean
            metric_cols = [c for c in big_df.columns
                            if c not in ("fold", "dataset_type")]
            mean_df = (
                big_df[metric_cols]
                .groupby(big_df["dataset_type"])   # single group, but keeps shape
                .mean()
                .reset_index(drop=True)
            )
            mean_df.to_csv(
                os.path.join(model_params.latent_path_main,
                            f"mean_scores_{data_split}_samplesize-{sample_size}.csv"),
                index=False
            )
        




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

def reshape_scores(df_scores: pd.DataFrame,
                   labels: list[str],
                   fold: int,
                   dataset_type: str) -> pd.DataFrame:
    """
    df_scores     the 4*len(labels) frame returned by
                    get_clustering_scores_optimized
    labels        in the order you want to appear (batch, celltype)
    fold          int, e.g. 1 ? 5
    dataset_type  'train' | 'val' | 'test'
    -------------------------------------------------------------
    returns a 1?row dataframe with columns
      batch_ch, batch_1/db, batch_silhouette,
      celltype_ch, ?, celltype_silhouette, fold, dataset_type
    """
    row = {}
    for lbl in labels:                       # fixed column order
        row[f"{lbl}_ch"]         = df_scores.loc["ch",         lbl]
        row[f"{lbl}_1/db"]       = df_scores.loc["1/db",       lbl]
        row[f"{lbl}_silhouette"] = df_scores.loc["silhouette", lbl]

    row["fold"]         = fold
    row["dataset_type"] = dataset_type
    return pd.DataFrame([row])

def plot_rep(
    adata: AnnData,
    shape_col: str = "celltype",
    color_col: str = "celltype",
    use_rep: str = "X_pca",
    markers: tuple = ("o", "v", "^", "<", "*"),
    clustering_scores=None,
    save_fig: bool = True,
    outpath: str = "",
    showplot: bool = False,
    palette_choice: str = "tab20",
    file_name: str = "latent",
    figsize: tuple = (7, 7)):
    """Plot 2D embedding with separate colour and shape legends."""
    print("Plotting latent representation:", use_rep)

    plt.ioff()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # --------------------------------------------
    # Determine unique categories
    # --------------------------------------------
    unique_shapes = np.unique(adata.obs[shape_col])
    unique_colors = np.unique(adata.obs[color_col])

    # --------------------------------------------
    # Build colour palette
    # --------------------------------------------
    if isinstance(palette_choice, list):
        color_palette = palette_choice
    elif palette_choice == "hsv":
        color_palette = sns.color_palette("hsv", len(unique_colors))
    elif palette_choice == "tab20":
        color_palette = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(unique_colors))]
    elif palette_choice == "Set2":
        color_palette = sns.color_palette("Set2", len(unique_colors))
    else:
        raise ValueError("palette_choice must be 'hsv', 'tab20', 'Set2', or a list of colours")

    color_map = {c: color_palette[i] for i, c in enumerate(unique_colors)}
    shape_map = {s: markers[i % len(markers)] for i, s in enumerate(unique_shapes)}

    # --------------------------------------------
    # Coordinates
    # --------------------------------------------
    coords = adata.X if use_rep == "X" else adata.obsm[use_rep]
    x, y = coords[:, 0], coords[:, 1]

    for s in unique_shapes:
        for c in unique_colors:
            mask = (adata.obs[shape_col] == s) & (adata.obs[color_col] == c)
            ax.scatter(
                x[mask],
                y[mask],
                color=color_map[c],
                marker=shape_map[s],
                alpha=0.7,
                s=1,
            )

    # --------------------------------------------
    # Legends
    # --------------------------------------------
    colour_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_map[c], markeredgecolor="none", markersize=10)
        for c in unique_colors
    ]
    legend_colour = ax.legend(
        handles=colour_handles,
        labels=list(unique_colors),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title=color_col,
    )
    ax.figure.add_artist(legend_colour)

    shape_handles = [
        Line2D([0], [0], marker=shape_map[s], linestyle="", color="black", markerfacecolor="none", markersize=10, markeredgewidth=1.5)
        for s in unique_shapes
    ]
    legend_shape = ax.legend(
        handles=shape_handles,
        labels=list(unique_shapes),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.65),
        ncol=3,
        title=shape_col,
    )

    extra_artists = (legend_colour, legend_shape)

    # --------------------------------------------
    # Axis labels
    # --------------------------------------------
    if "pca" in use_rep and "variance_ratio" in adata.uns.get("pca", {}):
        vr = adata.uns["pca"]["variance_ratio"]
        ax.set_xlabel(f"PC 1 ({vr[0]*100:.2f} %)")
        ax.set_ylabel(f"PC 2 ({vr[1]*100:.2f} %)")
    else:
        ax.set_xlabel(f"{use_rep} 1")
        ax.set_ylabel(f"{use_rep} 2")

    # --------------------------------------------
    # Save / show
    # --------------------------------------------
    if save_fig and outpath:
        os.makedirs(outpath, exist_ok=True)
        fig.savefig(
            os.path.join(outpath, f"{use_rep}_{file_name}.png"),
            bbox_extra_artists=extra_artists,
            bbox_inches="tight",
        )
    if showplot:
        plt.show()
    else:
        plt.close(fig)


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


def postprocess_cv_results(cv_results: Dict[str, Any], shape_color_dict: Dict[str, Any], plot_params: Dict[str, Any]) -> None:
    """Plot training history and latent spaces from *cv_results*."""
    # 1) History plot
    plot_scvi_history(
        cv_results["history"],
        save_path=os.path.join(plot_params["outpath"], "scvi_history.png"),
        show=True,
    )

    # 2) Latent space visualisations
    for fold, adata_splits in cv_results["all_adata"].items():
        print(f"\nPlotting latent space for fold {fold}")
        for split_name, adata in adata_splits.items():
            print(f"  Split: {split_name}")
            for combo_name, combo_params in shape_color_dict.items():
                file_name = f"{combo_name}_X_scvi_fold{fold}_{split_name}"
                plot_rep(
                    adata=adata,
                    use_rep="X_scvi",
                    shape_col=combo_params["shape_col"],
                    color_col=combo_params["color_col"],
                    file_name=file_name,
                    **plot_params,
                )

                
    print("Finished all plots")


# ======================================================
# 4. Main execution block
# ======================================================

# -------------------------------------
# 4.1 CV & metadata preparation
# -------------------------------------
folds_list = list(range(1, 6))  # 5‑fold CV
# folds_list = [1,2]

batch_col = model_params_dict["batch_col"]
bio_col   = model_params_dict["bio_col"]
donor_col = model_params_dict["donor_col"]  # optional in plotting only

shape_color_dict = {
    f"{bio_col}-{bio_col}":   {"shape_col": bio_col,   "color_col": bio_col},
    f"{batch_col}-{batch_col}": {"shape_col": batch_col, "color_col": batch_col},
    }

metadata_all = pd.read_csv(glob.glob(data_path + "/*meta.csv")[0])
metadata_all["celltype"] = metadata_all["celltype"].astype("category")
metadata_all["batch"]    = metadata_all["sampleID"].astype("category")

seen_donor_ids = np.unique(metadata_all[batch_col]).tolist()
celltype_ids   = np.unique(metadata_all[bio_col]).tolist()

print("Number of batches:", len(seen_donor_ids))

# -------------------------------------
# 4.2 Run CV
# -------------------------------------
cv_results = run_scvi_across_folds(
    input_base_path        = input_base_path,
    out_base_paths_dict    = base_paths_dict,
    folds_list             = folds_list,
    run_name               = run_name,
    model_params_dict      = model_params_dict,
    build_model_dict       = build_model_dict,
    compile_dict           = compile_dict,
    save_model             = save_model,
    batch_col              = batch_col,
    bio_col                = bio_col,
    batch_col_categories   = seen_donor_ids,
    bio_col_categories     = celltype_ids,
    model_type             = "scvi",
    issparse               = True,
    load_dense             = True,
    shape_color_dict       = shape_color_dict,
    return_scores_temp     = True,
    sample_size = 10000,
    n_batch2plot = 20,
    seed = 5,
    plot_params = plot_params
    )






