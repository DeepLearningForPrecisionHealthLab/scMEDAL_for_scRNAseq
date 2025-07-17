# ======================
# 0. Imports
# ======================
import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from scMEDAL_for_scRNAseq.utils.utils  import (
    create_folder,
    plot_rep,
    calculate_merge_scores,
)
from scMEDAL_for_scRNAseq.utils.model_train_utils import  load_data

import tensorflow as tf
from .model import SAUCIE
from .loader import Loader
from scMEDAL_for_scRNAseq.utils.utils import create_folder, plot_rep, calculate_merge_scores



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

# Corrected SAUCIE pipeline
# SAUCIE batch?correction pipeline (MMD)
# ---------------------------------------------------------
# * lambda_b > 0   activates MMD batch alignment
# * exactly 3 layer widths  SAUCIE will create a 2D embedding
# ---------------------------------------------------------
def run_saucie(
    save_model,
    batch_col,
    bio_col,
    input_path_dict,
    model_params,
    shape_color_dict=None,
    sample_size=None,
    plot_params=None,
):
    """Train SAUCIE in *batch?correction* mode (lambda_b > 0).
    Only two label values are used per sample: reference batch = 0, all others = 1.
    """

    # -----------------------------------------------------
    # Imports (local to avoid TF graph contamination)
    # -----------------------------------------------------


    # -----------------------------------------------------
    # 1 ? load AnnData splits
    # -----------------------------------------------------
    adata_dict = load_data(
        input_path_dict,
        eval_test=model_params.eval_test,
        scaling=model_params.scaling,
        issparse=False,
        load_dense=True,
    )

    # fresh TF graph for every fold
    tf.reset_default_graph()
    tf.keras.backend.clear_session()

    adata_train, adata_val = adata_dict["train"], adata_dict["val"]
    adata_test = adata_dict.get("test")

    # -----------------------------------------------------
    # 2 ? make sure the batch column is *categorical* everywhere
    # -----------------------------------------------------
    for _ad in [adata_train, adata_val] + ([adata_test] if adata_test is not None else []):
        _ad.obs[batch_col] = _ad.obs[batch_col].astype("category")

    # pick *one* reference batch (first category)
    ref_batch = adata_train.obs[batch_col].cat.categories[0]
    # -----------------------------------------------------
    ref_batch = adata_train.obs[batch_col].unique()[0]  # first batch value

    def make_loader(adata, shuffle):
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X.copy()
        # binary labels: 0 = reference, 1 = non?reference
        labels = (adata.obs[batch_col] != ref_batch).astype("int32").values
        return Loader(X, labels=labels, shuffle=shuffle)

    train_loader = make_loader(adata_train, shuffle=True)

    # -----------------------------------------------------
    # 3 ? build + train SAUCIE (batch mode)
    #    * 3 layer widths ? embedding automatically 2?D
    # -----------------------------------------------------
    saucie = SAUCIE(
        input_dim=adata_train.X.shape[1],
        layers=model_params.layers,      # e.g. [512,256,128]
        lambda_b=model_params.lambda_b,  # >0 for batch correction
        lambda_c=0.0,
        lambda_d=0.0,
        learning_rate=model_params.learning_rate,
    )

    batch_size      = model_params.batch_size
    epochs_desired  = model_params.epochs
    steps_per_epoch = int(np.ceil(adata_train.n_obs / batch_size))
    total_steps     = steps_per_epoch * epochs_desired

    saucie.train(train_loader, steps=total_steps)

    # evaluation splits
    splits = {"train": adata_train, "val": adata_val}
    if adata_test is not None:
        splits["test"] = adata_test

    results = {}
    latent_path = model_params.latent_path
    create_folder(latent_path)

    for split, ad in splits.items():
        loader_eval = make_loader(ad, shuffle=False)

        emb, _           = saucie.get_embedding(loader_eval)       # 2D embedding
        _,   clusters    = saucie.get_clusters(loader_eval)        # still works (binary codes)
        recon, _         = saucie.get_reconstruction(loader_eval)
        print("\n\n latent dims:",emb.shape)

        ad.obsm["X_saucie"]         = emb
        ad.obs["saucie_clusters"] = pd.Categorical(clusters.astype(int))



        latent_key = f"{model_params.encoder_latent_name}_{split}"
        np.save(os.path.join(latent_path, f"{latent_key}.npy"), emb)
        np.save(os.path.join(latent_path, f"recon_{split}.npy"),     recon)
        np.save(os.path.join(latent_path, f"saucie_clusters_{split}.npy"),   clusters)

        # optional plots
        if shape_color_dict:
            for tag, cfg in shape_color_dict.items():
                pp = dict(plot_params or {})
                pp.update({
                    "outpath": model_params.plots_path,
                    "file_name": f"{tag}_SAUCIE_{split}",
                    "shape_col": cfg["shape_col"],
                    "color_col": cfg["color_col"],
                })
                plot_rep(ad, use_rep="X_saucie", **pp)

        # scores
        scores = calculate_merge_scores(
            latent_list=["X_saucie"],
            adata=ad,
            labels=[batch_col, bio_col],
            sample_size=sample_size,
        )

        scores.to_csv(os.path.join(latent_path, f"saucie_scores_{split}_samplesize-{sample_size}.csv"))
        results[split] = scores

    return results


