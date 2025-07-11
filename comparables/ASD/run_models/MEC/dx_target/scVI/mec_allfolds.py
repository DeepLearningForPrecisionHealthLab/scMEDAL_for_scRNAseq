import gc
import warnings
from typing import Dict, Any, List

# --- Third‑party libraries
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scanpy as sc
# import scvi
import torch
import tensorflow as tf
#import anndata
# from anndata import AnnData
# from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, balanced_accuracy_score
# from scvi.dataloaders import DataSplitter


import os
# import random
import gc
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import shutil
import sys
import pickle


# --- Project‑specific modules
# NOTE: keep sys.path adjustments *before* relative imports
sys.path.append(
    "/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/ASD/run_models/scVI"
)
sys.path.append(
    "/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/ASD"
)

from model_config import *  # imports model_params_dict, build_model_dict, etc.)
# from scMEDAL.utils.utils import (
#     create_folder,
#     read_adata,
#     get_OHE,
#     min_max_scaling,
#     plot_rep,
#     calculate_merge_scores,
#     get_split_paths,
#     calculate_zscores,
#     get_clustering_scores_optimized,
# )
from scMEDAL.utils.callbacks import ComputeLatentsCallback
from scMEDAL.utils.model_train_utils import ModelManager, load_data  #, compute_scores

# Silence potential tensorflow warnings for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning)
print("tf_version", tf.__version__)

# import time
import glob



from paths_config import results_path_dict, input_base_path

from scMEDAL.utils.model_train_utils import (
    ModelManager,
    calculate_metrics_with_ci,load_latent_spaces,prepare_latent_space_inputs)
from model_config import (
    data_path, model_params_dict, base_paths_dict, run_name,
    LatentClassifier_config, load_latent_spaces_dict,
    saved_models_base, source_file
)

from scMEDAL.utils.compare_results_utils import (
    get_input_paths_df,
    get_latent_paths_df,
    create_latent_dict_from_df
)


# Def functions, they are already in model_train_utils but that version do not work with python 12, so I had to rewrite them



def random_forest_accuracy_and_predictions(inputs, adata_dict, model_params, eval_test=False,n_estimators=100, seed=42,save_model=True):
    """
    Trains a RandomForest classifier using concatenated latent space features, evaluates it on train, validation, 
    and optionally the test datasets, and returns accuracy metrics along with updated AnnData objects 
    containing RandomForest predictions.

    Parameters:
    - inputs (dict): Dictionary containing input data for each dataset type ('train', 'val', 'test'). 
                     It includes keys 'fe_latent' and optionally 're_latent' for feature and regularization latent spaces.
    - adata_dict (dict): Dictionary containing AnnData objects or ground truth y values for 'train', 'val', and 'test' datasets.
    - model_params: Object containing model parameters, including the path to save predictions.
    - eval_test (bool): Boolean flag indicating whether to evaluate the model on the test set (default: False).
    - n_estimators (int): # random forest estimators
    - seed (int): random state

    Returns:
    - dict: A dictionary containing:
        - "metrics": DataFrame with accuracy and balanced accuracy metrics for the RandomForest classifier across train, validation, and optionally test datasets.
        - "adata_dict": Updated AnnData dictionary with true and predicted labels for each dataset, labeled with RandomForest predictions.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier

    def concatenate_features(dataset_type):
        if "re_latent" in inputs[dataset_type]:
            return np.concatenate((inputs[dataset_type]["fe_latent"], inputs[dataset_type]["re_latent"]), axis=1)
        else:
            return inputs[dataset_type]["fe_latent"]

    def process_dataset(dataset_type, X, y_true):
        # Make predictions
        y_pred = clf.predict(X)

        # Convert predictions to one-hot encoding
        y_pred_ohe = np.eye(num_classes)[y_pred]

        # Save true and predicted labels to adata_dict
        outputs_data = adata_dict[f'{dataset_type}_y']
        y_pred_df = pd.DataFrame(y_pred_ohe, columns=outputs_data.columns)
        adata_dict[dataset_type].obs["true_labels_rf"] = outputs_data.columns[outputs_data.values.argmax(axis=1)]
        adata_dict[dataset_type].obs["pred_labels_rf"] = y_pred_df.columns[y_pred_df.values.argmax(axis=1)]
        adata_dict[dataset_type].obs.to_csv(os.path.join(model_params.latent_path, f"y_pred_{dataset_type}_rf.csv"))

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Calculate balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # Return accuracy and balanced accuracy
        return {"accuracy": accuracy, "balanced_accuracy": balanced_acc, "adata_dict": adata_dict}

    # Initialize scaler and label encoder
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    # Prepare and standardize the training set
    X_train = concatenate_features("train")
    X_train = scaler.fit_transform(X_train)
    y_train = label_encoder.fit_transform(adata_dict["train_y"].values.argmax(axis=1))

    # Train the RandomForest classifier
    clf = RandomForestClassifier(n_estimators=n_estimators,random_state=seed)
    clf.fit(X_train, y_train)
    num_classes = adata_dict["train_y"].shape[1]
    # Optionally save the model
    if save_model:
        model_path = os.path.join(model_params.model_path, "random_forest_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

    # Standardize the validation set using the fitted scaler
    X_val = concatenate_features("val")
    X_val = scaler.transform(X_val)
    y_val = label_encoder.transform(adata_dict["val_y"].values.argmax(axis=1))

    # Process the train and validation sets
    train_results = process_dataset("train", X_train, y_train)
    val_results = process_dataset("val", X_val, y_val)

    # Initialize the metrics DataFrame
    metrics_df = pd.DataFrame({
        "Dataset": ["train", "val"],
        "RFAccuracy": [train_results["accuracy"], val_results["accuracy"]],
        "RFBalancedAccuracy": [train_results["balanced_accuracy"], val_results["balanced_accuracy"]]
    })

    # Evaluate on the test set if eval_test is True
    if eval_test:
        X_test = concatenate_features("test")
        X_test = scaler.transform(X_test)
        y_test = label_encoder.transform(adata_dict["test_y"].values.argmax(axis=1))
        test_results = process_dataset("test", X_test, y_test)

        try:
            # For pandas < 2.0
            metrics_df = metrics_df.append({
                "Dataset": "test",
                "RFAccuracy": test_results["accuracy"],
                "RFBalancedAccuracy": test_results["balanced_accuracy"]
            }, ignore_index=True)
        except AttributeError:
            # For pandas >= 2.0
            test_metrics = pd.DataFrame([{
                "Dataset": "test",
                "RFAccuracy": test_results["accuracy"],
                "RFBalancedAccuracy": test_results["balanced_accuracy"]
            }])
            metrics_df = pd.concat([metrics_df, test_metrics], ignore_index=True)


    adata_dict["train"] = train_results["adata_dict"]["train"]
    adata_dict["val"] = val_results["adata_dict"]["val"]

    return {"metrics": metrics_df, "adata_dict": adata_dict}


def dummy_classifier_chance_accuracy(inputs, adata_dict, model_params, eval_test=False,seed = 42,save_model=True):
    """
    Trains a DummyClassifier using the 'stratified' strategy to calculate the chance accuracy (baseline accuracy) 
    using concatenated latent space features. Evaluates it on train, validation, and optionally the test datasets, 
    and returns chance accuracy metrics.

    Parameters:
    - inputs (dict): Dictionary containing input data for each dataset type ('train', 'val', 'test'). 
                     It includes keys 'fe_latent' and optionally 're_latent' for feature and regularization latent spaces.
    - adata_dict (dict): Dictionary containing AnnData objects or ground truth y values for 'train', 'val', and 'test' datasets.
    - model_params: Object containing model parameters, including the path to save predictions.
    - eval_test (bool): Boolean flag indicating whether to evaluate the model on the test set (default: False).
    - seed (int) : seed set for repreducible results of dummy classifier with strategy: stratified

    Returns:
    - dict: A dictionary containing:
        - "metrics": DataFrame with chance accuracy and balanced accuracy metrics for the DummyClassifier across train, validation, and optionally test datasets.
        - "adata_dict": Updated AnnData dictionary with true and predicted labels for each dataset, labeled with DummyClassifier predictions.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.dummy import DummyClassifier

    def concatenate_features(dataset_type):
        if "re_latent" in inputs[dataset_type]:
            return np.concatenate((inputs[dataset_type]["fe_latent"], inputs[dataset_type]["re_latent"]), axis=1)
        else:
            return inputs[dataset_type]["fe_latent"]

    def process_dataset(dataset_type, X, y_true):
        # Make predictions
        y_pred = clf.predict(X)

        # Convert predictions to one-hot encoding
        y_pred_ohe = np.eye(num_classes)[y_pred]

        # Save true and predicted labels to adata_dict
        outputs_data = adata_dict[f'{dataset_type}_y']
        y_pred_df = pd.DataFrame(y_pred_ohe, columns=outputs_data.columns)
        adata_dict[dataset_type].obs["pred_labels_dummyclass"] = y_pred_df.columns[y_pred_df.values.argmax(axis=1)]
        adata_dict[dataset_type].obs.to_csv(os.path.join(model_params.latent_path, f"y_pred_{dataset_type}_dummy.csv"))

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Return accuracy and balanced accuracy
        return {"chance_accuracy": accuracy, "adata_dict": adata_dict}

    # Initialize scaler and label encoder
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    # Prepare and standardize the training set
    X_train = concatenate_features("train")
    X_train = scaler.fit_transform(X_train)
    y_train = label_encoder.fit_transform(adata_dict["train_y"].values.argmax(axis=1))

    # Train the DummyClassifier with the 'stratified' strategy
    clf = DummyClassifier(strategy="stratified",random_state = seed)
    clf.fit(X_train, y_train)
    num_classes = adata_dict["train_y"].shape[1]

    if save_model:
        model_path = os.path.join(model_params.model_path, "dummy_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

    # Standardize the validation set using the fitted scaler
    X_val = concatenate_features("val")
    X_val = scaler.transform(X_val)
    y_val = label_encoder.transform(adata_dict["val_y"].values.argmax(axis=1))

    # Process the train and validation sets
    train_results = process_dataset("train", X_train, y_train)
    val_results = process_dataset("val", X_val, y_val)

    # Initialize the metrics DataFrame
    metrics_df = pd.DataFrame({
        "Dataset": ["train", "val"],
        "ChanceAccuracy": [train_results["chance_accuracy"], val_results["chance_accuracy"]],
    })

    # Evaluate on the test set if eval_test is True
    if eval_test:
        X_test = concatenate_features("test")
        X_test = scaler.transform(X_test)
        y_test = label_encoder.transform(adata_dict["test_y"].values.argmax(axis=1))
        test_results = process_dataset("test", X_test, y_test)

        try:
            # pandas < 2.0
            metrics_df = metrics_df.append({
                "Dataset": "test",
                "ChanceAccuracy": test_results["chance_accuracy"]
            }, ignore_index=True)
        except AttributeError:
            # pandas ? 2.0
            test_metrics = pd.DataFrame([{
                "Dataset": "test",
                "ChanceAccuracy": test_results["chance_accuracy"]
            }])
            metrics_df = pd.concat([metrics_df, test_metrics], ignore_index=True)

        adata_dict["test"] = test_results["adata_dict"]["test"]


    adata_dict["train"] = train_results["adata_dict"]["train"]
    adata_dict["val"] = val_results["adata_dict"]["val"]

    return {"metrics": metrics_df, "adata_dict": adata_dict}

def run_model_pipeline_LatentClassifier_v3(
    Model:None,                              # kept for API-compatibility ? not used
    latent_path_dict: Dict[str, Any],
    build_model_dict:None,                   # not used
    compile_dict:None,                       # not used
    model_params,
    save_model: True,                   
    batch_col: str,
    bio_col: str,
    base_path: str,
    fold: int,
    models_list: List[str],
    latent_keys_config: Dict[str, str],
    batch_col_categories: Optional[List[str]] = None,
    bio_col_categories: Optional[List[str]] = None,
    return_metrics: bool = True,
    return_adata_dict: bool = False,
    return_trained_model: bool = False,   # not produced
    model_type: str = "mec",              # not used
    issparse: bool = False,
    load_dense: bool = False,
    seed: Optional[int] = None,
):
    """
    Steps

        1. load_latent_spaces
        2. prepare_latent_space_inputs
        3. Random-Forest vs. Dummy baseline
        4. merge & save metrics

    you can call:

        run_model_pipeline_LatentClassifier_v3(**pipeline_LatentClassifier_config)
    """

    # ------------------------------------------------------------------
    # 1) Load latent spaces directly (returns an adata_dict)
    # ------------------------------------------------------------------
    adata_dict = load_latent_spaces(
        base_path=base_path,
        fold=fold,
        models_list=models_list,
        latent_path_dict=latent_path_dict,
        model_params=model_params,
        batch_col=batch_col,
        bio_col=bio_col,
        batch_col_categories=batch_col_categories,
        bio_col_categories=bio_col_categories,
        issparse=issparse,
        load_dense=load_dense,
    )

    # ------------------------------------------------------------------
    # 2) Build classifier inputs
    # ------------------------------------------------------------------
    print("Batches available:", np.unique(adata_dict["train"].obs[batch_col]))
    inputs = prepare_latent_space_inputs(
        adata_dict,
        latent_keys_config,
        eval_test=model_params.eval_test,
    )

    # ------------------------------------------------------------------
    # 3) Random-Forest evaluation
    # ------------------------------------------------------------------
    rf_results = random_forest_accuracy_and_predictions(
        inputs,
        adata_dict,
        model_params=model_params,
        eval_test=model_params.eval_test,
        seed=seed,
        save_model = save_model,
    )
    adata_dict = rf_results["adata_dict"]
    rf_metrics = rf_results["metrics"]

    # Chance (Dummy) baseline
    chance_results = dummy_classifier_chance_accuracy(
        inputs,
        adata_dict,
        model_params=model_params,
        eval_test=model_params.eval_test,
        seed=seed,
        save_model = save_model,
    )
    adata_dict = chance_results["adata_dict"]
    chance_metrics = chance_results["metrics"]

    # ------------------------------------------------------------------
    # 4) Merge & save metrics
    # ------------------------------------------------------------------
    metrics_df = pd.merge(rf_metrics, chance_metrics, on="Dataset")
    os.makedirs(model_params.latent_path, exist_ok=True)
    metrics_df.to_csv(
        os.path.join(model_params.latent_path, "metrics.csv"),
        index=False,
    )

    # ------------------------------------------------------------------
    # 5) Optional returns
    # ------------------------------------------------------------------
    results = {}
    if return_metrics:
        results["metrics"] = metrics_df
    if return_adata_dict:
        results["adata"] = adata_dict
    # return_trained_model is ignored by design

    return results if results else None
# --------------------------------------------------------------------------------------
# 1. Get Input Paths and Latent Paths
# --------------------------------------------------------------------------------------
df_latent = get_latent_paths_df(results_path_dict, k_folds=5)
df_inputs = get_input_paths_df(input_base_path, k_folds=5, eval_test=True)

# Merge latent and input paths
df = pd.merge(df_latent, df_inputs, on=["Split", "Type"], how="left")
print("Reading paths,\ndf paths:\n", df.head(5))

# --------------------------------------------------------------------------------------
# 2. Define Variables Necessary to Load Data and Train Model
# --------------------------------------------------------------------------------------
batch_col = model_params_dict['batch_col']
bio_col = model_params_dict['bio_col']

# Load metadata
metadata_all = pd.read_csv(glob.glob(os.path.join(data_path, "*meta.csv"))[0])
metadata_all[bio_col] = metadata_all[bio_col].astype('category')

# Define batch column (original metadata does not include batch yet)
metadata_all[batch_col] = metadata_all["batch"].astype('category')

print("n batches:", len(np.unique(metadata_all[batch_col]).tolist()))

# Define categories for batch and bio columns
batch_col_categories = np.unique(metadata_all[batch_col]).tolist()
print("check ordered batches:", batch_col_categories)

bio_col_categories = np.unique(metadata_all[bio_col]).tolist()
print("bio categories:", bio_col_categories)

# --------------------------------------------------------------------------------------
# 3. Update the Config for the Model
# --------------------------------------------------------------------------------------
load_latent_spaces_dict['batch_col_categories'] = batch_col_categories
load_latent_spaces_dict['bio_col_categories'] = bio_col_categories

# Update model
# LatentClassifier_config['Model'] = MixedEffectsModel (we are not going to use it, only RF classifier)

# Create latent path dictionary
latent_path_dict = create_latent_dict_from_df(df_latent)
load_latent_spaces_dict["latent_path_dict"] = latent_path_dict


# --------------------------------------------------------------------------------------
# 4. Run the Classifier for All Folds Latent Space
# --------------------------------------------------------------------------------------
# folds_list = [1] 
folds_list = list(range(1, 6))  # Folds 1 to 5
all_folds_metrics_df = pd.DataFrame()

pipeline_LatentClassifier_config = {}
pipeline_LatentClassifier_config['issparse']=False
pipeline_LatentClassifier_config['load_dense']=True
pipeline_LatentClassifier_config['save_model']=save_model

for fold in folds_list:
    print("fold", fold)
    load_latent_spaces_dict["fold"] = fold

    # Initialize model manager
    model_manager = ModelManager(
        model_params_dict,
        base_paths_dict,
        run_name,
        save_model=LatentClassifier_config["save_model"],
        use_kfolds=True,
        kfold=load_latent_spaces_dict["fold"]
    )

    # Update LatentClassifier config
    load_latent_spaces_dict["model_params"] = model_manager.params
    pipeline_LatentClassifier_config = {**pipeline_LatentClassifier_config,**load_latent_spaces_dict, **LatentClassifier_config}

    # Run pipeline
    results = run_model_pipeline_LatentClassifier_v3(**pipeline_LatentClassifier_config)
    results["metrics"]["fold"] = fold

    # Append metrics
    all_folds_metrics_df = pd.concat([all_folds_metrics_df, results["metrics"]], ignore_index=True)

    # Clear memory
    gc.collect()


# --------------------------------------------------------------------------------------
# 5. Save All Folds Metrics Results
# --------------------------------------------------------------------------------------
output_path = os.path.join(
    load_latent_spaces_dict["model_params"].latent_path_main,
    "metrics_allfolds.csv"
)
all_folds_metrics_df.to_csv(output_path)
print("\nall_folds_metrics_df:", all_folds_metrics_df)

# --------------------------------------------------------------------------------------
# 6. Calculate and Save 95% CI
# --------------------------------------------------------------------------------------
results_df = calculate_metrics_with_ci(all_folds_metrics_df)
output_path = os.path.join(
    load_latent_spaces_dict["model_params"].latent_path_main,
    "metrics_allfolds_95CI.csv"
)
results_df.to_csv(output_path)
print("\nresults_df:", results_df)


# --------------------------------------------------------------------------------------
# 7. Save Configuration File
# --------------------------------------------------------------------------------------
destination_path = os.path.join(saved_models_base, run_name, 'model_config.py')
print("\nCopying config.py file to:", destination_path)
shutil.copy(source_file, destination_path)






