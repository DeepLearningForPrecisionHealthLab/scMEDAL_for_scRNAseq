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
# Corrected SAUCIE pipeline

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
    from model import SAUCIE
    from loader import Loader
    from scMEDAL.utils.utils import (
        create_folder,
        plot_rep,
        calculate_merge_scores,
    )

    # Load data
    adata_dict = load_data(
        input_path_dict,
        eval_test=model_params.eval_test,
        scaling=model_params.scaling,
        issparse=False,
        load_dense=True,
    )

    tf.reset_default_graph()
    tf.keras.backend.clear_session()

    adata_train, adata_val = adata_dict['train'], adata_dict['val']
    adata_test = adata_dict.get('test')

    # Prepare data (ensure batches are integer-encoded)
    def prepare_loader(adata, batch_col, shuffle=True):
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        batch_labels = adata.obs[batch_col]
        if batch_labels.dtype != np.int32 and batch_labels.dtype != np.int64:
            batch_labels = batch_labels.astype('category').cat.codes
        return Loader(X, labels=batch_labels.values, shuffle=shuffle)

    train_loader = prepare_loader(adata_train, batch_col)



    # Initialize and train SAUCIE
    saucie = SAUCIE(
        input_dim=adata_train.X.shape[1],
        layers=[512, 256, 132, 50],
        lambda_b=0.0,
        lambda_c=0.05,
        lambda_d=0.0,
        learning_rate=model_params.learning_rate
    )
    saucie.train(train_loader, steps=model_params.epochs)

    splits = {'train': adata_train, 'val': adata_val}
    if adata_test is not None:
        splits['test'] = adata_test

    results = {}

    for split_name, adata in splits.items():
        loader_eval = prepare_loader(adata, batch_col, shuffle=False)

        # Embedding and clustering
        latent,_ = saucie.get_embedding(loader_eval)
        print("latent:",latent)
        print("latent type:",type(latent))
        print("latent shape",latent.shape)
        n_clusters, clusters = saucie.get_clusters(loader_eval)
        print("\n\nsaucie n_clusters:",n_clusters)
        recon,_ = saucie.get_reconstruction(loader_eval)
        print("\n\nsaucie recon shape",recon.shape)
        print("saucie recon:",recon.shape)

        adata.obsm["X_saucie"] = latent
        
        adata.obs["saucie_clusters"] = pd.Categorical(clusters.astype(int))

        # Save latent space arrays
        latent_path = model_params.latent_path
        create_folder(latent_path)



        
        latent_key = f"{model_params.encoder_latent_name}_{split_name}"
        adata.obsm["X_saucie"] = latent
        np.save(os.path.join(model_params.latent_path, f"{latent_key}.npy"), latent)

        np.save(os.path.join(model_params.latent_path, f"recon_{split_name}.npy"), recon)

        np.save(os.path.join(model_params.latent_path, f"saucie_clusters_{split_name}.npy"), clusters)



        # Plot embeddings
        # Plot embeddings
        if shape_color_dict:
            for combo, cfg in shape_color_dict.items():
                pp = plot_params.copy()
                pp["outpath"] = model_params.plots_path
                pp["file_name"] = f"{combo}_X_saucie_{split_name}.png"
                pp["shape_col"] = cfg["shape_col"]
                pp["color_col"] = cfg["color_col"]

                plot_rep(
                    adata=adata,
                    use_rep="X_saucie",
                    **pp
                )
        # Calculate scores
        scores = calculate_merge_scores(
            latent_list=["X_saucie"],
            adata=adata,
            labels=[batch_col, bio_col],
            sample_size=sample_size
        )
        scores.to_csv(f"{latent_path}/saucie_scores_{split_name}_samplesize-{sample_size}.csv")
        results[split_name] = scores

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
