# ======================
# 0. Imports
# ======================
import os
import sys
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

from model import SAUCIE  # Make sure this is in your PYTHONPATH and TensorFlow 1.x is active
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

from model_config import *
from scMEDAL.utils.utils import (
    create_folder,
    plot_rep,
    calculate_merge_scores,
    get_split_paths,
    get_OHE,
)
from scMEDAL.utils.model_train_utils import ModelManager, load_data

# Silence TF warnings for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================
# Utility Functions
# ======================

def adata_to_numpy(adata, batch_col=None, bio_col=None):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    batch = None
    if batch_col:
        batch = adata.obs[batch_col]
        # Convert to categorical codes if not already int
        if not np.issubdtype(batch.dtype, np.integer):
            batch = batch.astype('category').cat.codes
        batch = batch.to_numpy()
    celltype = None
    if bio_col:
        celltype = adata.obs[bio_col]
        if not np.issubdtype(celltype.dtype, np.integer):
            celltype = celltype.astype('category').cat.codes
        celltype = celltype.to_numpy()
    return X, batch, celltype
def add_saucie_to_adata(adata, embedding, clusters, prefix="saucie"):
    adata.obsm["X_{}".format(prefix)] = embedding
    adata.obs["{}_clusters".format(prefix)] = pd.Categorical(clusters.astype(int))
    return adata

def get_clustering_scores_optimized(adata, use_rep, labels, sample_size=None):
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
    data_rep = adata.X if use_rep == "X" else adata.obsm[use_rep]
    dict_scores = {}
    if sample_size and sample_size < data_rep.shape[0]:
        indices = np.random.choice(data_rep.shape[0], sample_size, replace=False)
        subsampled_data_rep = data_rep[indices]
    else:
        subsampled_data_rep = data_rep
    for label in labels:
        labels_array = adata.obs[label].to_numpy()
        if sample_size and sample_size < data_rep.shape[0]:
            subsampled_labels = labels_array[indices]
        else:
            subsampled_labels = labels_array
        db_score = davies_bouldin_score(subsampled_data_rep, subsampled_labels)
        ch_score = calinski_harabasz_score(subsampled_data_rep, subsampled_labels)
        silhouette = silhouette_score(subsampled_data_rep, subsampled_labels)
        dict_scores[label] = [db_score, 1/db_score, ch_score, silhouette]
    df_scores = pd.DataFrame(dict_scores, index=['db', '1/db', 'ch', 'silhouette'])
    return df_scores

# ======================
# 1. SAUCIE All Folds Routine (main function)
# ======================

from typing import Optional, Dict, List

def run_saucie(
    save_model,
    batch_col,
    bio_col,
    Model=None,
    input_path_dict=None,
    build_model_dict=None,
    compile_dict=None,
    model_params=None,
    batch_col_categories=None,  # <-- this will be used as batch_categories if provided
    bio_col_categories=None,
    return_scores=False,
    return_adata_dict=False,
    return_trained_model=False,
    model_type="saucie",
    issparse=False,
    load_dense=True,
    shape_color_dict=None,
    sample_size=None,
    return_history=False,
    n_batch2plot=None,
    seed=5,
    plot_params=None,
):
    # -------- 1. Load data --------
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

    # -------- 2. Find all unique batch categories across splits --------
    import pandas as pd
    if batch_col_categories is not None:
        batch_categories = batch_col_categories
    else:
        all_batches = pd.concat([
            adata_train.obs[batch_col],
            adata_val.obs[batch_col],
        ] + ([adata_test.obs[batch_col]] if adata_test is not None else []))
        batch_categories = sorted(all_batches.unique())

    # -------- 3. One-hot encode batch column with the same categories --------
    ohe_train = get_OHE(adata_train, categories=batch_categories, col=batch_col)
    ohe_val   = get_OHE(adata_val, categories=batch_categories, col=batch_col)
    ohe_test  = get_OHE(adata_test, categories=batch_categories, col=batch_col) if adata_test is not None else None

    # -------- 4. Prepare X and one-hot labels --------
    X_train = adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else np.array(adata_train.X)
    X_val   = adata_val.X.toarray() if hasattr(adata_val.X, "toarray") else np.array(adata_val.X)
    X_test  = adata_test.X.toarray() if (adata_test is not None and hasattr(adata_test.X, "toarray")) else (np.array(adata_test.X) if adata_test is not None else None)

    # -------- 5. Train SAUCIE model --------
    from model import SAUCIE
    from loader import Loader

    saucie = SAUCIE(
        input_dim=X_train.shape[1],
        layers=model_params.layers,
        lambda_b=model_params.lambda_b,
        lambda_c=model_params.lambda_c,
        lambda_d=model_params.lambda_d,
        learning_rate=model_params.learning_rate
    )

    train_loader = Loader(X_train, labels=adata_train.obs[batch_col], shuffle=True)
    saucie.train(train_loader, steps=model_params.epochs)
    # After: saucie.train(train_loader, steps=n_steps)

    # -------- 6. Organize splits for evaluation --------
    splits = {
        "train": (X_train, ohe_train.values, adata_train),
        "val":   (X_val, ohe_val.values, adata_val)
    }
    if adata_test is not None:
        splits["test"] = (X_test, ohe_test.values, adata_test)

    latent_path = model_params.latent_path
    os.makedirs(latent_path, exist_ok=True)

    df_scores_dict = {}

    for split, (X, batch_ohe, adata) in splits.items():
        loader = Loader(X, labels=batch_ohe, shuffle=False)
        embedding = saucie.get_embedding(loader)
        n_clusters, clusters = saucie.get_clusters(loader)
        recon = saucie.get_reconstruction(loader)

        add_saucie_to_adata(adata, embedding, clusters, prefix="saucie")

        # Save numpy arrays for each split
        np.save(os.path.join(latent_path, f"saucie_embedding_{split}.npy"), embedding)
        np.save(os.path.join(latent_path, f"saucie_clusters_{split}.npy"), clusters)
        np.save(os.path.join(latent_path, f"saucie_recon_{split}.npy"), recon)

        # Save 2D plots, if requested
        if shape_color_dict:
            for combo, cfg in shape_color_dict.items():
                plot_rep(
                    adata=adata,
                    use_rep="X_saucie",
                    shape_col=cfg["shape_col"],
                    color_col=cfg["color_col"],
                    file_name=f"{combo}_X_saucie_{split}_fold{seed}",
                    **(plot_params if plot_params is not None else {})
                )

        # Save clustering metrics
        if return_scores:
            adata.obsm["X_saucie"] = embedding
            latent_list = ["X_saucie"]
            df_scores = calculate_merge_scores(
                latent_list=latent_list,
                adata=adata,
                labels=[batch_col, bio_col],
                sample_size=sample_size
            )
            df_scores.to_csv(
                os.path.join(latent_path, f"saucie_scores_{split}_samplesize-{sample_size}.csv")
            )
            df_scores_dict[split] = df_scores

    results = {}
    if return_scores:
        results["scores"] = df_scores_dict
    if return_adata_dict:
        results["adata"] = adata_dict
    return results

# ======================
# 2. Main Execution Block
# ======================
if __name__ == "__main__":
    import glob
    # Example as in your scVI block, adapt as needed!
    folds_list = list(range(1, 6))
    batch_col = model_params_dict["batch_col"]
    bio_col   = model_params_dict["bio_col"]

    shape_color_dict = {
        "{}-{}".format(bio_col, bio_col):   {"shape_col": bio_col,   "color_col": bio_col},
        "{}-{}".format(batch_col, batch_col): {"shape_col": batch_col, "color_col": batch_col},
    }
    metadata_all = pd.read_csv(glob.glob(data_path + "/*meta.csv")[0])
    # Convert cell type and batch columns to categorical
    metadata_all['celltype'] = metadata_all['celltype'].astype('category')
    metadata_all['batch'] = metadata_all["batch"].astype('category')
    seen_donor_ids = np.unique(metadata_all[batch_col]).tolist()
    celltype_ids   = np.unique(metadata_all[bio_col]).tolist()

    print("Number of batches:", len(seen_donor_ids))

    for intFold in folds_list:
        print("\nRunning SAUCIE Fold {}\n{}".format(intFold, "="*60))
        model_manager = ModelManager(
            params_dict=model_params_dict,
            base_paths_dict=base_paths_dict,
            run_name=run_name,
            save_model=save_model,
            use_kfolds=True,
            kfold=intFold,
        )
        model_params = model_manager.params
        input_path_dict = get_split_paths(base_path=input_base_path, fold=intFold)
        print("input_path_dict:", input_path_dict)

        results = run_saucie(
            save_model=save_model,
            batch_col=batch_col,
            bio_col=bio_col,
            batch_col_categories   = seen_donor_ids,
            bio_col_categories     = celltype_ids,
            Model=SAUCIE,
            input_path_dict=input_path_dict,
            build_model_dict=build_model_dict,
            compile_dict=compile_dict,
            model_params=model_params,
            shape_color_dict=shape_color_dict,
            sample_size=10000,
            plot_params=plot_params,
            # add any more arguments you use
        )
        # You can collect/store your results as needed here
