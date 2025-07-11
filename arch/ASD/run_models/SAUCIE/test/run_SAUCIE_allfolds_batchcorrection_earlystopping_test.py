# ======================
# 0. Imports
# ======================
import os
import sys
import gc
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

import tensorflow as tf
from model import SAUCIE
from loader import Loader
from scMEDAL.utils.utils import create_folder, plot_rep, calculate_merge_scores



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
# * lambda_b > 0  ? activates MMD batch alignment
# * exactly 3 layer widths ? SAUCIE will create a 2?D embedding
# ---------------------------------------------------------
# SAUCIE batch?correction pipeline with **early stopping** and **history plot**

# ------------------------------------------------------------------
# helper: build SAUCIE loader with binary batch labels (0 = reference, 1 = other)
# ------------------------------------------------------------------

def make_loader(adata, batch_col, ref_batch, shuffle):
    """Return a Loader with binary batch labels required by SAUCIE?s MMD loss."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X.copy()
    labels = (adata.obs[batch_col] != ref_batch).astype("int32").values
    return Loader(X, labels=labels, shuffle=shuffle)


# ------------------------------------------------------------------
# main routine ------------------------------------------------------
# ------------------------------------------------------------------

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
    """Train SAUCIE for batch correction with early stopping and history plot."""

    # ---- 1. load AnnData splits -------------------------------------------------
    adict = load_data(
        input_path_dict,
        eval_test=model_params.eval_test,
        scaling=model_params.scaling,
        issparse=False,
        load_dense=True,
    )
    ad_train, ad_val = adict["train"], adict["val"]
    ad_test = adict.get("test")

    # ensure batch column is categorical everywhere
    for ad in [ad_train, ad_val] + ([ad_test] if ad_test is not None else []):
        ad.obs[batch_col] = ad.obs[batch_col].astype("category")

    ref_batch = ad_train.obs[batch_col].cat.categories[0]

    # ---- 2. fresh graph per run -----------------------------------
    tf.reset_default_graph()
    tf.keras.backend.clear_session()

    # ---- 3. build model (batch?correction mode) -------------------
    saucie = SAUCIE(
        input_dim=ad_train.X.shape[1],
        layers=model_params.layers,         # e.g. [512, 256, 128]
        lambda_b=model_params.lambda_b,     # > 0 activates MMD
        lambda_c=0.0,
        lambda_d=0.0,
        learning_rate=model_params.learning_rate,
    )

    # ---- 4. training parameters ----------------------------------
    bs = model_params.batch_size
    max_epochs = model_params.epochs
    patience = getattr(model_params, "patience", 30)
    steps_per_epoch = int(np.ceil(ad_train.n_obs / bs))

    hist_tr, hist_val = [], []
    best_val, wait = np.inf, 0

    train_loader = make_loader(ad_train, batch_col, ref_batch, shuffle=True)
    val_loader = make_loader(ad_val, batch_col, ref_batch, shuffle=False)

    # ------------------------------------------------------------------
    # Loop over epochs. Each iteration **trains SAUCIE for a single epoch**
    # (i.e. runs `steps_per_epoch` gradient?update steps) and then evaluates
    # losses on training and validation sets, updating early?stopping state.
    # ------------------------------------------------------------------
    for epoch in range(max_epochs):
        saucie.train(train_loader, steps=steps_per_epoch)

        # Calculate **training** loss immediately after finishing the epoch
        tr_loss = float(saucie.get_loss(train_loader))
        # Calculate **validation** loss on the held?out validation split
        va_loss = float(saucie.get_loss(val_loader))
        hist_tr.append(tr_loss)
        hist_val.append(va_loss)

        # If validation loss improved beyond a small tolerance, treat this epoch
        # as the new best model.
        if va_loss < best_val - 1e-4:
            # Reset early?stopping counter because we have an improvement.
            best_val = va_loss
            wait = 0
            if save_model:
                # Optionally save the model weights corresponding to the best validation loss
                saucie.save(folder=os.path.join(model_params.models_path, "best"))
        else:
            # No improvement ? increment the early?stopping patience counter
            wait += 1
            if wait >= patience:
                print(
                    f"Early stop at epoch {epoch + 1} ? no val improvement in {patience} epochs"
                )
                break

        # Print progress every 10 epochs for monitoring
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{max_epochs}  train {tr_loss:.4f}  val {va_loss:.4f}"
            )

    # ---- 5. history plot -----------------------------------------
    # Plot and save the training vs. validation loss curves after training finishes
    create_folder(model_params.plots_path)
    plt.figure(figsize=(6, 4))
    plt.plot(hist_tr, label="train")
    plt.plot(hist_val, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(model_params.plots_path, "training_history.png"), dpi=120
    )
    plt.close()

    # ---- 6. evaluation on each split -----------------------------
    splits = {"train": ad_train, "val": ad_val}
    if ad_test is not None:
        splits["test"] = ad_test

    create_folder(model_params.latent_path)
    results = {}
    for split, ad in splits.items():
        loader_eval = make_loader(ad, batch_col, ref_batch, shuffle=False)
        emb, _ = saucie.get_embedding(loader_eval)
        _, clusters = saucie.get_clusters(loader_eval)
        recon, _ = saucie.get_reconstruction(loader_eval)

        ad.obsm["X_saucie"] = emb
        ad.obs["saucie_clusters"] = pd.Categorical(clusters)

                # ---- save latent, recon, clusters using caller's naming convention ----
        latent_key = f"{model_params.encoder_latent_name}_{split}"
        np.save(os.path.join(model_params.latent_path, f"{latent_key}.npy"), emb)
        np.save(f"{model_params.latent_path}/recon_{split}.npy", recon)
        np.save(f"{model_params.latent_path}/clust_{split}.npy", clusters)

        # optional 2?D scatter plots
        if shape_color_dict:
            for tag, cfg in shape_color_dict.items():
                pp = dict(plot_params or {})
                pp.update(
                    {
                        "outpath": model_params.plots_path,
                        "file_name": f"{tag}_Saucie_{split}",
                        "shape_col": cfg["shape_col"],
                        "color_col": cfg["color_col"],
                    }
                )
                plot_rep(ad, use_rep="X_saucie", **pp)

        # batch / bio mixing scores
        sc = calculate_merge_scores(
            ["X_saucie"], ad, [batch_col, bio_col], sample_size
        )
        sc.to_csv(f"{model_params.latent_path}/scores_{split}.csv")
        results[split] = sc

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


        # You can collect/store your results as needed here
        results = run_saucie(
                save_model=save_model,
                batch_col=batch_col,
                bio_col = bio_col,
                input_path_dict = input_path_dict,
                model_params = model_params,
                shape_color_dict=shape_color_dict,
                sample_size=1000,
                plot_params=plot_params)
