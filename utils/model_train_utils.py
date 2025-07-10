import numpy as np
import anndata as ad
import sys
from collections import namedtuple
from typing import Dict, List, Any
from .utils import create_folder,read_adata,get_OHE,min_max_scaling,plot_rep,calculate_merge_scores,get_split_paths,calculate_zscores,get_clustering_scores_optimized
# from utils import create_folder,read_adata,get_OHE,min_max_scaling,plot_rep,calculate_merge_scores,get_split_paths,calculate_zscores,get_clustering_scores_optimized
from .callbacks import ComputeLatentsCallback
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import glob
from anndata import AnnData
import scipy
from sklearn.metrics import accuracy_score, balanced_accuracy_score







def generate_run_name(model_params_dict, constant_keys, name='run'):
    """
    Generates a run name by concatenating key-value pairs from the model parameters dictionary.

    Parameters:
    - model_params_dict (dict): Dictionary containing model parameters.
    - constant_keys (list): List of keys to be excluded from the run name.
    - name (str): Prefix for the run name.

    Returns:
    - str: A string representing the run name.
    """
    def safe_round(value):
        try:
            return np.round(value, 2)
        except TypeError:
            return value

    def format_value(key, value):
        if isinstance(value, list):
            return '-'.join(map(str, value))
        if isinstance(value, dict):
            tp = namedtuple(key, field_names=value.keys())
            return tp._make(value)
        return str(safe_round(value))
    
    # If constant_keys is None, set it to an empty list
    if constant_keys is None:
        constant_keys = []

    run_name_parts = [f"{key}-{format_value(key, value)}" for key, value in model_params_dict.items() if key not in constant_keys]
    # Format the timestamp to include only date, hours, and minutes
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')
    
    run_name = f"{name}_" + "_".join(run_name_parts) + '_' + timestamp
    return run_name



def get_x_y_z(adata, batch_col, bio_col, batch_col_categories, bio_col_categories, use_rep="X"):
    """
    Extracts and returns features, labels, and batch information from an AnnData object.

    Parameters:
    - adata: AnnData object containing the dataset.
    - batch_col (str): The name of the batch column.
    - bio_col (str): The name of the biological column.
    - batch_col_categories (list): Categories for the batch column to be used in one-hot encoding.
    - bio_col_categories (list): Categories for the biological column to be used in one-hot encoding.
    - use_rep (str, optional): The representation of data to be used. 'X' for default representation or key for .obsm.

    Returns:
    - x_y_z_dict (dict): Dictionary containing:
        - 'x': Features from the AnnData object (either .X or specified .obsm).
        - 'y': One-hot encoded labels.
        - 'z': One-hot encoded batch information.
    """

    # Choose the data representation based on 'use_rep'
    if use_rep != "X" and use_rep not in adata.obsm:
        raise KeyError(f"The key '{use_rep}' is not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    x = adata.X if use_rep == "X" else adata.obsm[use_rep]

    # Retrieve one-hot encoded batch information
    z_ohe = get_OHE(adata, categories=batch_col_categories, col=batch_col)

    # Retrieve one-hot encoded labels
    y_ohe = get_OHE(adata, categories=bio_col_categories, col=bio_col)

    # Create a dictionary to store 'x', 'y', and 'z' components
    x_y_z_dict = {
        "x": x,
        "y": y_ohe,
        "z": z_ohe
    }

    return x_y_z_dict


def process_data(adata, batch_col, bio_col, get_pred, use_z, 
                     batch_col_categories, bio_col_categories, return_outputs=True,use_rep="X"):
    """
    Processes data to prepare inputs and outputs for modeling based on specified conditions.

    Parameters:
    - adata: AnnData object containing the dataset.
    - batch_col (str): The name of the batch column.
    - bio_col (str): The name of the biological column.
    - get_pred (bool): Determines whether to process predictions (outputs) or not.
    - use_z (bool): Flag to determine if one-hot encoding (OHE) of the batch column should be used in inputs.
    - batch_col_categories: Categories for the batch column to be used in OHE.
    - bio_col_categories: Categories for the biological column to be used in OHE.
    - return_outputs (bool, optional): Determines whether to return outputs. Default is True.
    - use_rep (str, optional): The representation of data to be used. 'X' for default representation or key for .obsm.

    Returns:
    - If return_outputs is True, returns a tuple of (inputs, outputs).
    - If return_outputs is False, returns only the inputs.
    """
    # Choose the data representation based on 'use_rep'
    if use_rep != "X" and use_rep not in adata.obsm:
        raise KeyError(f"The key '{use_rep}' is not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    x = adata.X if use_rep == "X" else adata.obsm[use_rep]
    # Process inputs
    if use_z:
        z_ohe = get_OHE(adata, categories=batch_col_categories, col=batch_col)
        # inputs = (x,z)
        inputs = (x, z_ohe.values)
    else:
        inputs = x
    if return_outputs:
        # Process outputs
        if get_pred:
            y_ohe = get_OHE(adata, categories=bio_col_categories, col=bio_col)
            # outputs = (x,y)
            outputs = (x, y_ohe.values)
        else:
            outputs = x

        return inputs, outputs
    else:
        return inputs
    
def get_z_ohe_dict(adata_dict, batch_col, batch_col_categories):
    """
    Generates a dictionary of one-hot encoded representations for a specified column in AnnData objects.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects. Each key is a dataset type (e.g., 'train', 'val', 'test').
    - batch_col (str): Name of the column in AnnData.obs to be one-hot encoded.
    - batch_col_categories (list): List of categories for one-hot encoding of the specified column.

    Returns:
    - dict: A dictionary where each key corresponds to a key in adata_dict, and each value is a DataFrame of one-hot encoded values.
    """
    z_ohe_dict = {}
    valid_data_types = ['train', 'val', 'test']
    for key, adata in adata_dict.items():
        print("getting z_ohe for ",key)
        # Proceed only if key is one of the valid types
        if key in valid_data_types:
        
            # Ensure the batch column is of type string for proper OHE
            if adata.obs[batch_col].dtype != 'object':
                adata.obs[batch_col] = adata.obs[batch_col].astype(str)

            # Perform one-hot encoding
            z_ohe = get_OHE(adata, categories=batch_col_categories, col=batch_col)
            z_ohe_dict[key] = z_ohe
        else:
            print(key," not added")

    return z_ohe_dict


def get_train_val_data(adata_dict, batch_col, bio_col, get_pred, use_z, 
                       batch_col_categories=None, bio_col_categories=None,use_rep="X",eval_test=False):
    """
    Prepare input and output data for training and validation, including optional one-hot encoding.

    This function processes the datasets in the provided dictionary to form appropriate input and output pairs
    for training and validation, potentially including one-hot encoded auxiliary information.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects keyed by 'train' and 'val' for training and validation datasets.
    - batch_col (str): Column name in `adata.obs` for batch information.
    - bio_col (str): Column name in `adata.obs` for biological labels.
    - get_pred (bool): Flag to determine if biological labels should be used as output for training.
    - use_z (bool): Flag to determine if batch information should be included in the model input.
    - batch_col_categories (list, optional): Categories for one-hot encoding of batch information.
    - bio_col_categories (list, optional): Categories for one-hot encoding of biological labels.
    - use_rep (str, optional): The base representation key to be used. 'X' for default representation or key for .obsm.
        For obsm, the function will append '_train' or '_val' based on the dataset.
    - eval_test (bool): whether or not the test data is also processed and added to the data_dict

    Returns:
    - data_dict: Contains the following keys. train_in, train_out,val_in, val_out
    """
    # Adjust use_rep for training and validation datasets based on the specific naming conventions in the .obsm attribute of the AnnData objects
    # If use_rep is "X", it indicates that the default data representation (adata.X) is being used.
    # In this case, there's no need to rename the latent space, so train_rep and val_rep are set to "X".
    # If use_rep is not "X", it implies a specific representation in adata.obsm is being used,
    # and the names are adjusted to include '_train' or '_val' to match the naming convention for training and validation datasets.
    train_rep = f"{use_rep}_train" if use_rep != "X" else "X"
    val_rep = f"{use_rep}_val" if use_rep != "X" else "X"
    test_rep = f"{use_rep}_test" if use_rep != "X" else "X"


    # Prepare data for training
    # Process the 'train' dataset from adata_dict and get the input and output data
    train_in, train_out = process_data(adata_dict['train'], batch_col, bio_col, get_pred, use_z, 
                                       batch_col_categories, bio_col_categories,use_rep=train_rep)


    # Prepare data for validation
    # Process the 'val' dataset from adata_dict and get the input and output data
    val_in, val_out = process_data(adata_dict['val'], batch_col, bio_col, get_pred, use_z, 
                                   batch_col_categories, bio_col_categories,use_rep=val_rep)

    # Prepare data for testing
    # Process the 'val' dataset from adata_dict and get the input and output data
    if eval_test:
        test_in, test_out = process_data(adata_dict['test'], batch_col, bio_col, get_pred, use_z, 
                                   batch_col_categories, bio_col_categories,use_rep=test_rep)
    print("check that the shapes make sense")
    if isinstance(train_out, tuple):
        print("train out shapes x,y:", train_out[0].shape, train_out[1].shape)
    else:
        print("train out shape x:", train_out.shape)

    if isinstance(val_out, tuple):
        print("val out shapes x,y:", val_out[0].shape, val_out[1].shape)
    else:
        print("val out shape x:", val_out.shape)

    if use_z:
        print("train in shapes x,z:", train_in[0].shape, train_in[1].shape)
        print("val in shapes x,z:", val_in[0].shape, val_in[1].shape)
    else:
        print("train in shape x:", train_in.shape)
        print("val in shape x:", val_in.shape)

    
    #return (train_in, train_out), (val_in, val_out)

    # Return structured data
    data_dict = {
        'train_in': train_in,
        'train_out': train_out,
        'val_in': val_in,
        'val_out': val_out
    }
    if eval_test:
        data_dict['test_in'] = test_in
        data_dict['test_out'] = test_out

    return data_dict
    


def load_data(paths_dict, eval_test, scaling="min_max",issparse=False, load_dense=False):
    """
    Load and optionally scale datasets for training, validation, and testing.

    This function reads datasets from given paths in the paths dictionary and applies scaling if specified.
    It supports loading either two (train and validation) or three datasets (including test),
    based on the `eval_test` flag.

    Parameters:
    - paths_dict (dict): A dictionary containing paths with keys 'train', 'val', and optionally 'test'.
    - eval_test (bool): Flag indicating whether to load the test dataset. If True, expects 'test' key in `paths_dict`.
    - scaling (str, optional): The type of scaling to apply to the data. Default is "min_max". You can also select "z_scores".
                              Other options should be handled within the function.
    - issparse(bool): True if X is in sparse array, False if its dense
    - load_dense (bool): If True, forces conversion of sparse arrays to dense format.


    Returns:
    - dict: A dictionary containing the loaded AnnData objects with keys 'train', 'val', and optionally 'test'.
    """
    def read_subset(subset_path,issparse=issparse, load_dense=load_dense):
        # Replace with the actual function to read AnnData
        X, var, obs = read_adata(subset_path, issparse=issparse)
        
        # Convert X to dense if it's sparse and load_dense is True.
        if load_dense and issparse and scipy.sparse.issparse(X):
            X = X.toarray()
        print("X.shape before scaling",X.shape)

        # Apply scaling to X based on the scaling parameter.
        if scaling == "min_max":
            # Placeholder for the actual min_max_scaling function; this needs to be defined or imported.
            X = min_max_scaling(X)
        elif scaling =="z_scores":
            X = calculate_zscores(X)
            print("X.shape after scaling",X.shape)

        # Create an AnnData object with the scaled data.
        adata = ad.AnnData(X, obs=obs, var=var)
        return adata 

    adata_dict = {
        'train': read_subset(paths_dict['train']),
        'val': read_subset(paths_dict['val'])
    }
    
    if eval_test:
        adata_dict['test'] = read_subset(paths_dict['test'])

    return adata_dict



class ModelManager:
    """
    A class that manages model parameters and the creation of directories for checkpoints, plots, and latent spaces.
    """

    def __init__(self, params_dict, base_paths_dict, run_name, save_model=False,use_kfolds=True,kfold=None,run_model=True, get_baseline_scores=False):
        """
        Initializes the ModelManager with given parameters, base paths, and run name.
        
        Parameters:
        - params_dict (dict): Dictionary of model parameters.
        - base_paths_dict (dict): dict containing the base paths for saved models, figures, and latent space directories.
        - run_name (str): The name of the current run.
        - save_model (bool): Flag indicating whether to save model checkpoints.
        - run_model (bool): Flag indicating whether to run the model.
        - get_baseline_scores (bool): Flag indicating whether to get baseline scores.
        """
        self.params = self.Namespace(params_dict)
        self.use_kfolds=use_kfolds
        self.kfold=kfold
        if run_model:
            self.create_directories(base_paths_dict, run_name, save_model)
        elif get_baseline_scores:
            self.create_baseline_scores_directory(base_paths_dict, run_name)


    class Namespace:
        """
        A class that converts a dictionary into a namespace object.
        This allows accessing the dictionary keys as class attributes.
        """
        def __init__(self, adict):
            self.__dict__.update(adict)

    def create_baseline_scores_directory(self, base_paths_dict, run_name):
        """Creates a directory for baseline scores"""

        # Extracting individual paths from the dictionary
        Baseline_scores_path= base_paths_dict["Baseline_scores"]
        # Create and return directory paths
        Baseline_scores_path = os.path.join(Baseline_scores_path, run_name)
        self.params.Baseline_scores_path = Baseline_scores_path

        create_folder(self.params.Baseline_scores_path)

    def create_directories(self, base_paths_dict, run_name, save_model):
        """
        Creates directories for model checkpoints, plots, and latent space representations 
        based on a provided base path and run name. If `save_model` is set to True, 
        it also prepares a checkpoint path.
        
        Parameters:
        - base_paths_dict (dict): A dict containing the base paths for saved models,
                            figures, and latent space directories.
        - run_name (str): The name of the current run. This is used to create a 
                        unique subdirectory for the run under each base path.
        - save_model (bool)
        - use_kfolds (bool): Indicates whether to use k-fold cross-validation.
        - kfold (int or None): The specific k-fold number for which to create directories.
        
        Returns:
        - model_path (str): The path to the directory where the model is saved.
        - plots_path (str): The path to the directory where plots will be saved.
        - latent_path (str): The path to the directory for saving latent space representations.
        - checkpoint_path (str or None): The path to the checkpoint directory, or None if 
                                        `save_model` is False.
        """
        # Extracting individual paths from the dictionary
        saved_models_base = base_paths_dict['models']
        figures_base = base_paths_dict['figures']
        latent_space_base = base_paths_dict['latent']
        # Destructure the base_paths tuple
        # saved_models_base, figures_base, latent_space_base = base_paths

        
        # Create and return directory paths
        saved_models_base = os.path.join(saved_models_base, run_name)
        figures_base = os.path.join(figures_base, run_name)
        latent_space_base = os.path.join(latent_space_base, run_name)

        

        # Create k-fold specific directories if required
        if self.use_kfolds and self.kfold is not None:
            # define main folders (parent folders)
            self.params.model_path_main = saved_models_base
            self.params.plots_path_main = figures_base
            self.params.latent_path_main = latent_space_base
            # define splits folders
            saved_models_base = os.path.join(saved_models_base, "splits_" + str(self.kfold))
            create_folder(saved_models_base)
            figures_base = os.path.join(figures_base, "splits_" + str(self.kfold))
            create_folder(figures_base)
            latent_space_base = os.path.join(latent_space_base, "splits_" + str(self.kfold))
            create_folder(latent_space_base)
        self.params.model_path = saved_models_base
        self.params.plots_path = figures_base
        self.params.latent_path = latent_space_base

        # self.params.model_path = os.path.join(saved_models_base, run_name)
        # self.params.plots_path = os.path.join(figures_base, run_name)
        # self.params.latent_path = os.path.join(latent_space_base, run_name)
        
        # Check if the directories exist and create them if they don't
        for path in [self.params.model_path, self.params.plots_path, self.params.latent_path]:
            create_folder(path)
        
        # Define the checkpoint path if save_model is enabled
        # Set save_model to True if you wish to save model checkpoints

        # Define the checkpoint path if save_model is enabled
        self.params.checkpoint_path = None
        if save_model:
            self.params.checkpoint_path = os.path.join(self.params.model_path, "cp-{epoch:04d}.ckpt")
            self.params.checkpoint_dir = os.path.dirname(self.params.checkpoint_path)

    
        # Return the paths for later use
        return self.params.model_path, self.params.plots_path, self.params.latent_path, self.params.checkpoint_path

    def update_params(self, new_params):
        """
        Updates the model parameters with new values.
        
        Parameters:
        - new_params (dict): Dictionary of new parameters to update.
        """
        self.params.__dict__.update(new_params)

    def print_params(self):
        """
        Prints the current model parameters.
        """
        print("Current Model Parameters:")
        for key, value in self.params.__dict__.items():
            print(f"{key}: {value}")




def plot_clustering_scores_curve(train_output_dir, val_output_dir, label_col, save_path=None):
    """
    Plots the Davies-Bouldin Index, its inverse, and the Calinski-Harabasz Index over epochs from two CSV files,
    comparing training and validation datasets. The inverse Davies-Bouldin Index is plotted on a twin axis due
    to its different scale.

    Parameters:
    - train_output_dir (str): Directory path where the CSV file for training data is stored.
    - val_output_dir (str): Directory path where the CSV file for validation data is stored.
    - label_col (str): Label column for which the clustering scores are recorded. Used for identifying the dataset.
    - save_path (str, optional): Path to save the resulting plot. If None, the plot is not saved to a file.

    Outputs:
    - A matplotlib plot displaying the three clustering scores over epochs for both training and validation datasets.
      The plot includes a legend and is displayed via `plt.show()`.

    Example:
    plot_combined_clustering_scores('/path/to/train/output', '/path/to/val/output', 'batch', '/path/to/save/plot.png')
    """
    # Load training data
    train_score_path = os.path.join(train_output_dir, f'clustering_scores_{label_col}.csv')
    train_data = pd.read_csv(train_score_path)

    # Load validation data
    val_score_path = os.path.join(val_output_dir, f'clustering_scores_{label_col}.csv')
    val_data = pd.read_csv(val_score_path)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Main axis for DB and CH indexes
    ax1.plot(train_data['epoch'], train_data['DB'], label='Train DB Index', color='blue', marker='o')
    ax1.plot(val_data['epoch'], val_data['DB'], label='Val DB Index', color='cyan', marker='x')
    ax1.plot(train_data['epoch'], train_data['CH'], label='Train CH Index', color='red', linestyle='--')
    ax1.plot(val_data['epoch'], val_data['CH'], label='Val CH Index', color='magenta', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('DB and CH Scores')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Twin axis for 1/DB index
    ax2 = ax1.twinx()
    ax2.plot(train_data['epoch'], train_data['1/DB'], label='Train Inverse DB Index', color='green', marker='o', linestyle=':')
    ax2.plot(val_data['epoch'], val_data['1/DB'], label='Val Inverse DB Index', color='lightgreen', marker='x', linestyle=':')
    ax2.set_ylabel('Inverse DB Score')
    ax2.legend(loc='upper right')

    plt.title(f'Combined Clustering Scores Over Epochs for {label_col}')

    # Show plot
    plt.show()

    # Save plot to file if save_path is provided
    if save_path:
        os.path.join(save_path, f'clustering_scores_{label_col}.png')
        fig.savefig(os.path.join(save_path, f'clustering_scores_{label_col}.png'))
        print(f"Plot saved to {save_path}")






def train_and_save_model(model, train_in, train_out, val_in, val_out, model_params, save_model=False,metadata_dict=None, **kwargs):
    """
    Trains the provided model with given data and saves the model parameters and best weights.

    Parameters:
    - model: The model to be trained.
    - train_in: Input data for training.
    - train_out: Output data for training.
    - val_in: Input data for validation.
    - val_out: Output data for validation.
    - model_params: Namespace object with model parameters and hyperparameters:
        monitor_metric, patience, checkpoint_path, epochs, batch_size, model_path.
    - save_model (bool): Flag to determine if the model should be saved.
    - metadata_dict (dict): Dictionary with pd.DataFrames containing the metadata for 'train' and 'val' keys

    Returns:
    - model: The trained model.
    - history: Training history object.
    """
    # initialize time tracker
    import time
    start_time = time.time()
    callbacks = [History()]

    if model_params.stop_criteria == "early_stopping":
        # EarlyStopping callback
        early_stopping_callback = EarlyStopping(
            monitor=model_params.monitor_metric,
            patience=model_params.patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_callback)

    # ModelCheckpoint callback
    if save_model:
        cp_callback = ModelCheckpoint(
            filepath=model_params.checkpoint_path, 
            verbose=1, 
            save_best_only=True,
            monitor=model_params.monitor_metric,
            mode='min',
            save_weights_only=True
        )
        callbacks.append(cp_callback)

        # Save initial weights at epoch 0 using the `checkpoint_path` format
        model.save_weights(model_params.checkpoint_path.format(epoch=0))

    # Compute latent callback: gets latent and computes clustering scores
    if model_params.compute_latents_callback and metadata_dict:
        print("Adding ComputeLatentsCallback, which calcuates clustering scores on sample size: ", model_params.sample_size)
        # Assuming `inputs` can be either a tuple or another type, and `modeltype` is a string
        # define outpus path:
        train_output_dir = os.path.join(model_params.latent_path, "train")
        val_output_dir = os.path.join(model_params.latent_path, "val")


        LatentsCallback_train = ComputeLatentsCallback(model,
                                                        X=train_in,
                                                        metadata = metadata_dict["train"],
                                                        labels_list=[model_params.batch_col, model_params.bio_col],
                                                        output_dir = train_output_dir,
                                                        sample_size = model_params.sample_size,
                                                        batch_size=model_params.batch_size,
                                                        model_type = model_params.model_type)
        LatentsCallback_val = ComputeLatentsCallback(model,
                                                        X=val_in,
                                                        metadata = metadata_dict["val"],
                                                        labels_list=[model_params.batch_col, model_params.bio_col],
                                                        output_dir = val_output_dir,
                                                        sample_size = model_params.sample_size,
                                                        batch_size = model_params.batch_size,
                                                        model_type = model_params.model_type)
        callbacks.append(LatentsCallback_train)
        callbacks.append(LatentsCallback_val)
    print(os.system('nvidia-smi'))
    # Train the model with the callbacks
    history = model.fit(train_in, train_out, epochs=model_params.epochs,
                        batch_size=model_params.batch_size, 
                        validation_data=(val_in, val_out), 
                        callbacks=callbacks)
    print(os.system('nvidia-smi'))

    if model_params.compute_latents_callback and metadata_dict:
        # plot clustering scores curves
        plot_clustering_scores_curve(train_output_dir, val_output_dir, model_params.batch_col, save_path=model_params.latent_path)
        plot_clustering_scores_curve(train_output_dir, val_output_dir, model_params.bio_col, save_path=model_params.latent_path)

    model_params_dict = model_params.__dict__
    if model_params.stop_criteria == "early_stopping":
        print("\nAdding early stopping params..")
        # Update model_params with early stopping info
        stopped_epoch = early_stopping_callback.stopped_epoch
        best_epoch = stopped_epoch - early_stopping_callback.patience
        
        model_params_dict["best_epoch"] = best_epoch
        model_params_dict["checkpoint_name"] = best_epoch + 1
        model_params_dict["stopped_epoch"] = stopped_epoch
        print(f"Training stopped at epoch {stopped_epoch}.")
        print(f"Best epoch was {best_epoch}, which corresponds to checkpoint {best_epoch + 1}")
    if save_model:
        print("\nSaving model..")
        keys2drop = ["optimizer","loss","loss_weights","metrics"]
        dict2save = {f"{key}:{value}" for key, value in model_params_dict.items() if key not in keys2drop}
        # Check if 'loss_weights' is a key in the model_params_dict before proceeding
        if "loss_weights" in model_params_dict:
            # Check if the type of model_params_dict["loss_weights"] is a dictionary
            if isinstance(model_params_dict["loss_weights"], dict):
                # Getting the loss_weights sub-dictionary and prefixing keys with 'loss_weights_'
                prefixed_loss_weights = {'loss_weights_' + key: value for key, value in model_params_dict["loss_weights"].items()}
            else:
                # If it's not a dictionary, treat it as a single value under the key 'loss_weights'
                prefixed_loss_weights = {"loss_weights": model_params_dict["loss_weights"]}


            # Adding the updated loss_weights dictionary to dict2save
            dict2save.update(prefixed_loss_weights)
        # Correct once chatgpt is back (I want to add the tf elements)f
        # Save model parameters to a YAML file
        with open(model_params.model_path + '/model_params.yaml', 'w') as f:
            
            yaml.dump(dict2save, f)
            #yaml.dump(model_params_dict, f)
        print("Model saved")
    # compute total time to save and train model
    total_time = time.time() - start_time
    print(f"\nTotal time to train and save model: {total_time} seconds")

    return model, history



def filter_keys(model_params_dict, keys):
    """
    Filters a subset of parameters from the provided dictionary based on specified keys.

    This function is useful for extracting a specific set of parameters from a larger
    dictionary, for example, to obtain only the parameters relevant for a certain operation.
    You can use it if you want to extract the parameters to compile or train the model 

    Parameters:
    - model_params_dict (dict): The full dictionary of model parameters.
    - keys (list of str): A list of keys that are to be filtered from the model_params_dict.

    Returns:
    - dict: A subset of model_params_dict containing only the keys specified in the 'keys' list.
    """

    # Filter out only the keys-related parameters
    subset_params = {k: model_params_dict[k] for k in keys if k in model_params_dict}
    return subset_params


class PlotLoss:
    """
    A class for plotting the training loss of a model.

    Attributes:
    - history (dict): The history object from model training containing loss values.
    - model_params (Namespace object): The parameters of the model, used for scaling loss values. Created using ModelManager class.
    - save_model (bool): Flag indicating whether to save the plot.
    - model_type (str): Type of the model. Options: ["ae_da", "ae", "ae_re","aec","mec"].
    - showplot (bool): Whether to display the plot.
    - average curve (bool): wether the average curve is being plotted or its a plot per fold
    """

    def __init__(self, history, model_params, save_model, model_type="ae_da", showplot=False,average_curve=False):
        self.history = history.history if isinstance(history, History) else history
        self.model_params = model_params
        self.save_model = save_model
        self.model_type = model_type
        self.showplot = showplot
        self.average_curve = average_curve

        if self.model_type in ["scmedalfe", "scmedalfec"]:
            self.plot_ae_da()
        elif self.model_type == "ae":
            self.plot_ae()
        elif self.model_type == "scmedalre":
            self.plot_ae_re()
        elif self.model_type == "aec":
            self.plot_aec()
        elif self.model_type == "mec":
            self.plot_mec()
        else:
            print("No training loss plotted. Type a valid model_type [ae_da, ae, ae_re,aec,mec]")
        print("Loss plots completed")


    def plot_ae_re(self):
        """
        Plot training and validation loss for an Random effects Autoencoder (AE RE) model with reconstruction and latent clustering loss.
        """
        # First plot (cluster class loss)
        epochs = range(len(self.history["total_loss"]))
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()  # twin axis
            # Safely get loss_latent_cluster_weightwith a default value
        #loss_latent_cluster_weight = getattr(self.model_params, "loss_latent_cluster_weight", None)

        for key, val in self.history.items():
            if "recon_loss" in key:
                color = "b" 
                #marker = '--' if "val" in key else None
                ax1.plot(epochs, self.model_params.loss_recon_weight * np.array(val), color=color,linestyle='--' if "val" in key else '-', label=key)

            elif "la_clus" in key:# and isinstance(loss_latent_cluster_weight, (float, int)):
                color = "yellowgreen" 
                #marker = '--' if "val" in key else None
                ax2.plot(epochs, self.model_params.loss_latent_cluster_weight * np.array(val), color=color, linestyle='--' if "val" in key else '-', label=key)

            elif "total_loss" in key:
                color = "g"
                #marker = '--' if "val" in key else None
                ax1.plot(epochs, val, color=color, linestyle='--' if "val" in key else '-', label=key, alpha=0.5)

        ax1.set_ylabel("Total Loss / Recon Loss", color="g")
        ax2.set_ylabel("Latent Cluster Loss", color="yellowgreen")

        legend1 = ax1.legend(loc='upper right', bbox_to_anchor=(1.55, 1))
        legend2 = ax2.legend(loc='upper right', bbox_to_anchor=(1.6, 0.6))

        # if self.save_model:
        #     plt.savefig(self.model_params.plots_path + "/loss.png", bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
        if self.save_model:
            if not self.average_curve:
                plt.savefig(self.model_params.plots_path + "/loss.png", bbox_extra_artists=(legend1,legend2), bbox_inches='tight')
        
            else:
                plt.savefig(self.model_params.plots_path_main + "/avg_loss.png", bbox_extra_artists=(legend1,legend2), bbox_inches='tight')

        if self.showplot:
            plt.show()

        plt.clf()

        # Second plot (KLD)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        for key, val in self.history.items():
            # recon and total loss in first axis

            if "recon_loss" in key:
                color = "b" 
                #marker = '--' if "val" in key else None
                ax1.plot(epochs, self.model_params.loss_recon_weight * np.array(val), color=color, linestyle='--' if "val" in key else '-', label=key)

            elif "total_loss" in key:
                color = "g"
                #marker = '-' if "val" in key else None
                ax1.plot(epochs, val, color=color, linestyle='--' if "val" in key else '-', label=key, alpha=0.5)
            # kld has no val loss (point estimate)
            if key == "kld":

                ax2.plot(epochs, val,color= "darkgoldenrod", label=key)

        ax1.set_ylabel("Total loss/recon loss", color="g")
        ax2.set_ylabel("Kld loss", color ="darkgoldenrod")

        legend1 = ax1.legend(loc='upper right',bbox_to_anchor=(1.55,1))
        legend2 = ax2.legend(loc='upper right',bbox_to_anchor=(1.4,.6))
        # if self.save_model:
        #     plt.savefig(self.model_params.plots_path+"/loss_KLD.png", bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
        if self.save_model:
            if not self.average_curve:
                plt.savefig(self.model_params.plots_path + "/loss_KLD.png", bbox_extra_artists=(legend1,legend2), bbox_inches='tight')
        
            else:
                plt.savefig(self.model_params.plots_path_main + "/avg_loss_KLD.png", bbox_extra_artists=(legend1,legend2), bbox_inches='tight')
        if self.showplot:
            plt.show()


    def plot_ae(self):
        """
        Plot training and validation loss for an Autoencoder (AE) model.
        """
        train_loss = self.history["loss"]
        val_loss = self.history["val_loss"]

        fig, ax = plt.subplots(figsize=(5, 5), sharey=True)
        ax.plot(range(len(train_loss)), train_loss, "g-", label="train_loss", alpha=0.1)
        ax.plot(range(len(val_loss)), val_loss, "g--", label="val_loss")
        legend = plt.legend()
        plt.yscale("log")
        plt.xlabel("epochs")
        plt.ylabel("mse loss (log)")
        if self.save_model:
            if not self.average_curve:
                plt.savefig(self.model_params.plots_path + "/loss.png", bbox_extra_artists=(legend,), bbox_inches='tight')
        
            else:
                plt.savefig(self.model_params.plots_path_main + "/avg_loss.png", bbox_extra_artists=(legend,), bbox_inches='tight')

        if self.showplot:
            plt.show()


    def plot_ae_da(self):
        """
        Plots the losses for a Domain Adversarial Autoencoder model.
        """
        # Define the epochs based on the history length
        epochs = range(len(self.history["total_loss"]))

        # Creating a figure for the reconstruction loss plot
        fig_recon, ax_recon = plt.subplots()

        # Plotting only the reconstruction loss
        for key, val in self.history.items():
            if "recon_loss" in key:
                ax_recon.plot(epochs, self.model_params.loss_recon_weight * np.array(val), linestyle='--' if "val" in key else '-', color='b', label=key)

        # Setting axis labels and colors for the reconstruction loss plot
        ax_recon.set_ylabel("Reconstruction Loss", color='b')
        ax_recon.set_xlabel("Epochs")
        
        # Adding legend
        legend_recon = ax_recon.legend(loc='upper right')

        # Title for the reconstruction loss plot
        plt.title("Reconstruction Loss")

        # Saving the reconstruction loss plot if the model is being saved
        if self.save_model:
            if not self.average_curve:
                plt.savefig(self.model_params.plots_path + "/recon_loss.png", bbox_extra_artists=(legend_recon,), bbox_inches='tight')
                print(f"Saved reconstruction loss plot at: {self.model_params.plots_path}")
            else:
                plt.savefig(self.model_params.plots_path_main + "/avg_recon_loss.png", bbox_extra_artists=(legend_recon,), bbox_inches='tight')
                print(f"Saved average reconstruction loss plot at: {self.model_params.plots_path_main}")

        # Displaying the reconstruction loss plot
        if self.showplot:
            plt.show()

        # Creating the figure and axes for the main plot
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # Create a twin axis for different scale plotting

        # Iterate through the history items and plot them
        for key, val in self.history.items():
            # Scaling the values based on loss weights
            scaled_val = np.array(val)

            if "adv" in key:
                # Plotting adversarial loss on ax1
                ax1.plot(epochs, self.model_params.loss_gen_weight * scaled_val, linestyle='--' if "val" in key else '-', color='blueviolet', label=key)

            elif "recon_loss" in key:
                # Plotting reconstruction loss on ax1
                ax1.plot(epochs, self.model_params.loss_recon_weight * scaled_val, linestyle='--' if "val" in key else '-', color='b', label=key)  # 'b' is a valid color shorthand for blue

            elif "total_loss" in key:
                # Plotting total loss on ax1
                ax1.plot(epochs, scaled_val, linestyle='--' if "val" in key else '-', color='g', label=key)

            elif "class_loss" in key:
                ax2.plot(epochs, self.model_params.loss_class_weight * scaled_val, linestyle='--' if "val" in key else '-', color='skyblue', label=key)

        # Setting axis labels and colors for the main plot
        ax1.set_ylabel("Total/Recon/Adv Loss", color="g")
        ax2.set_ylabel("Class Loss", color="skyblue")

        # Positioning the legends for the main plot
        legend1 = ax1.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
        legend2 = ax2.legend(loc='upper right', bbox_to_anchor=(1.5, 0.6))

        # Title for the main plot
        plt.title("Losses Adjusted by Weights")

        # Saving the main plot if the model is being saved
        if self.save_model:
            if not self.average_curve:
                plt.savefig(self.model_params.plots_path + "/loss.png", bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
                print(f"Saved regular loss plot at: {self.model_params.plots_path}")
            else:
                plt.savefig(self.model_params.plots_path_main + "/avg_loss.png", bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
                print(f"Saved average loss plot at: {self.model_params.plots_path}")

        # Displaying the main plot
        if self.showplot:
            plt.show()




    def plot_aec(self):

        """
        Plots the training and validation losses for the AE_classifier model.
        The function creates subplots for reconstruction loss and classification loss,
        with specific colors and line styles for differentiating between training and validation losses.
        """
        # Define the epochs based on the history length
        epochs = range(len(self.history['loss']))

        # Creating the figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plotting Reconstruction Loss and Total Loss on the first subplot (ax1)
        for key in ['reconstruction_output_loss', 'val_reconstruction_output_loss']:
            if key in self.history:
                color = 'blue'
                linestyle = '-' if 'val' not in key else '--'
                ax1.plot(epochs, self.model_params.loss_weights['reconstruction_output']*np.array(self.history[key]), label=key, color=color, linestyle=linestyle)
        # Total Loss on the same axis (ax1)
        ax1.plot(epochs, self.history['loss'], label='Total Train Loss', color='green', linestyle='-')
        ax1.plot(epochs, self.history['val_loss'], label='Total Val Loss', color='green', linestyle='--')
        ax1.set_title('Reconstruction Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        legend1 = ax1.legend(loc='upper right')

        # Plotting Classification Loss on the second subplot (ax2)
        for key in ['classification_output_loss', 'val_classification_output_loss']:
            if key in self.history:
                color = 'skyblue'
                linestyle = '-' if 'val' not in key else '--'
                ax2.plot(epochs, self.model_params.loss_weights['classification_output']*np.array(self.history[key]), label=key, color=color, linestyle=linestyle)
        ax2.set_title('Classification Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        legend2 = ax2.legend(loc='upper right')

        plt.tight_layout()  # Adjust the layout

        # Save the plot if required
        # if self.save_model:
        #     plt.savefig(self.model_params.plots_path + "/aec_loss_plot.png", bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
        if self.save_model:
            if not self.average_curve:
                plt.savefig(self.model_params.plots_path + "/loss.png", bbox_extra_artists=(legend1,legend2), bbox_inches='tight')
        
            else:
                plt.savefig(self.model_params.plots_path_main + "/avg_loss.png", bbox_extra_artists=(legend1,legend2), bbox_inches='tight')

        # Show or close the plot based on the flag
        if self.showplot:
            plt.show()
        else:
            plt.close()

    def plot_mec(self):
        """
        Plot training and validation loss for an mixed effects classifier model.
        """
        train_loss = self.history["loss"]
        val_loss = self.history["val_loss"]

        fig, ax = plt.subplots(figsize=(5, 5), sharey=True)
        ax.plot(range(len(train_loss)), train_loss,linestyle="-", color='skyblue', label="Train Loss", alpha=0.7)
        ax.plot(range(len(val_loss)), val_loss, linestyle='--', label="Val Loss", alpha=0.7)
        legend = plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("CCE")
        plt.title("Training and Validation Loss")

        # if self.save_model:
        #     plt.savefig(self.model_params.plots_path + "/loss.png", bbox_extra_artists=(legend,), bbox_inches='tight')
        if self.save_model:
            if not self.average_curve:
                plt.savefig(self.model_params.plots_path + "/loss.png", bbox_extra_artists=(legend,), bbox_inches='tight')
        
            else:
                plt.savefig(self.model_params.plots_path_main + "/avg_loss.png", bbox_extra_artists=(legend,), bbox_inches='tight')

        if self.showplot:
            plt.show()





def get_pca_andplot(adata_dict, plot_params, eval_test=False, shape_color_dict={"celltype_vs_donor": {"shape_col": "celltype", "color_col": "donor"}},n_components=2):
    from sklearn.decomposition import PCA
    import time
    """
    Performs Principal Component Analysis (PCA) on the given AnnData objects and plots the results.
    
    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys 'train', 'val', and optionally 'test'.
    - plot_params (dict): Parameters to be used for plotting.
    - eval_test (bool, optional): A flag indicating whether test data is included and should be processed. Defaults to False.
    - shape_color_dict (dict, optional): Dictionary with shape_col and color_col combinations for plotting.
    - n_components (int): number of pca components

    Returns:
    - dict: A dictionary of updated AnnData objects with PCA transformations applied.
    """
    start_time = time.time()
    print("Calculating PCA ncomponents=",n_components)
    pca = PCA(n_components=n_components)
    variance_ratio = None  # Initialize variance_ratio

    # Loop through each dataset type: 'train', 'val', 'test'
    for key in ['train', 'val', 'test']:
        if key in adata_dict:
            adata = adata_dict[key]
            # Perform PCA; fit_transform for 'train', transform for 'val'/'test'
            if key == 'train':
                X_pca = pca.fit_transform(adata.X)
                variance_ratio = {"variance_ratio": pca.explained_variance_ratio_}
            else:
                X_pca = pca.transform(adata.X)

            # Update AnnData object with PCA results and variance ratio
            pca_key = f"X_pca_{key}"
            adata.obsm[pca_key] = X_pca
            adata.uns['pca'] = variance_ratio

            # Plotting
            if shape_color_dict:
                for combo_name, combo_params in shape_color_dict.items():
                    shape_col = combo_params["shape_col"]
                    color_col = combo_params["color_col"]
                    file_name = f"{shape_col}-{color_col}_{key}"
                    specific_plot_params = {**plot_params, "file_name": file_name}
                    plot_rep(adata, use_rep=pca_key, shape_col=shape_col, color_col=color_col, **specific_plot_params)

            # Additional generic plot for each type if needed outside the combinations
            # plot_rep(adata, use_rep=pca_key, **plot_params)

            # Skip 'test' dataset unless eval_test is True
            if key == 'test' and not eval_test:
                break
    total_time = time.time() - start_time
    print(f"Total PCA computation time: {total_time} seconds")
    return adata_dict



def get_encoder_latentandscores(adata_dict, model_encoder, model_params, batch_col, bio_col, plot_params, save_model=False, return_scores=False, z_ohe_dict=None, model_type="ae_da", other_inputs=None, shape_color_dict={"celltype_vs_donor": {"shape_col": "celltype", "color_col": "donor"}},sample_size=None):
    """
    Processes latent representations and calculates clustering scores for provided datasets using the given model encoder.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects for 'train', 'val', and 'test' datasets. Each adataset contains an annData object.
    - model_encoder: Encoder method of the trained model to compute latent representations.
    - model_params: Object containing model parameters and paths for saving data.
    - batch_col (str): Batch column name used for calculating clustering scores.
    - bio_col (str): Biological column name used for calculating clustering scores.
    - plot_params (dict): Parameters for plotting functions.
    - save_model (bool): Flag indicating whether to save latent representations.
    - return_scores (bool): Flag indicating whether to return clustering scores.
    - z_ohe_dict (dict, optional): One-hot encoded dictionary for different data types.
    - model_type (str): Type of model (default "ae_da").
    - other_inputs (dict, optional): Inputs for the mec model, nested dictionary in which each train, val and test dataset is a dictionary with keys = ['fe_latent','re_latent']
    - shape_color_dict (dict, optional): Dictionary with shape_col and color_col combinations for plotting.

    Returns:
    - (dict): Updated AnnData objects with latent representations.
    - (dict): DataFrames containing clustering scores for each dataset (if return_scores is True).
    - sample_size (int): sample size used to calculate clustering scores on a random subset of the latent space
    """
    def get_latent_list(base_latent_name, include_pca, include_baseline):
        latent_list = [base_latent_name]
        if include_pca:
            latent_list = ['X_pca_' + base_latent_name.split('_')[-1]] + latent_list
        if include_baseline:
            latent_list = ['X'] + latent_list
        return latent_list

    df_scores_dict = {}
    valid_data_types = ['train', 'val', 'test']

    print("\n\nGetting encoder latent space and clustering scores")

    for data_type, adata in adata_dict.items():
        # Proceed only if data_type is one of the valid types
        if data_type in valid_data_types:
            print("Subset:",data_type)
            # Determine input structure based on model_type and data format
            if model_type == "scmedalre" and isinstance(adata, AnnData) and z_ohe_dict is not None:
                # AE_RE.encoder inputs are = (x,z).
                inputs = (adata.X, z_ohe_dict[data_type].values)
                print(data_type,"inputs x,z shape:",np.shape(inputs[0]),np.shape(inputs[1]))
                print("\nx:",inputs[0])
                print("\nz:",inputs[1])
                print("\n unique values z:",np.unique(inputs[1]))
            elif model_type == "mec" and isinstance(other_inputs, dict):
                # Use other_inputs for 'mec' model. or most models the encoder inputs are x. 
                inputs = other_inputs[data_type]  
            elif isinstance(adata, AnnData):
                # For most models, we use  we use  x = adata.X for the encoder inputs. 
                # Note: Even for ae_da, the encoder is only applied to x.
                # The encoder weights were already penalized by the adversarial during training
                inputs = adata.X
                #convert to tensor
                inputs = tf.convert_to_tensor(inputs, dtype=tf.float32) 
                print("Type of inputs:", type(inputs))
            else:
                print(f"Unexpected data format for inputs from data_type '{data_type}'.")
                continue

                

            latent_key = f"{model_params.encoder_latent_name}_{data_type}"


            # Check if the model encoder has return_layer_activations attribute set to True
            use_layer_activations = hasattr(model_encoder, 'return_layer_activations') and model_encoder.return_layer_activations

            print("use_layer_activations:",use_layer_activations)


            # Get latent representation using model_encoder.predict(inputs) to ensure inference mode and simplify code (New version)
            outputs = model_encoder.predict(inputs, batch_size=model_params.batch_size)
            adata.obsm[latent_key] = outputs[-1] if use_layer_activations else outputs

            print("\nLatent_representation retrived using the encoder:",latent_key)

            if save_model:
                save_latent_representation(model_params.latent_path, latent_key, adata)

            latent_list = get_latent_list(latent_key, model_params.get_pca, model_params.get_baseline)

            print("\nPlotting latent space")
            if shape_color_dict:
                for combo_name, combo_params in shape_color_dict.items():
                    shape_col = combo_params["shape_col"]
                    color_col = combo_params["color_col"]
                    file_name = f"{shape_col}-{color_col}_{latent_key}"
                    specific_plot_params = {**plot_params, "file_name": file_name}
                    plot_rep(adata, use_rep=latent_key, shape_col=shape_col, color_col=color_col, **specific_plot_params)
            print("Finished plots")
            
            if return_scores:
                print("\nCalculating scores ..")
                df_scores = calculate_merge_scores(latent_list, adata, labels=[batch_col, bio_col],sample_size=sample_size)
                df_scores_dict[data_type] = df_scores

                df_scores.to_csv(os.path.join(model_params.latent_path, f"scores_{data_type}_samplesize-{sample_size}.csv"))
                #plot_table(df=np.round(df_scores, 4), out_name=data_type, model_path=model_params.latent_path)
                print("\nscores retrived")

    return (adata_dict, df_scores_dict) if return_scores else adata_dict





def save_latent_representation(latent_path, latent_key, adata):
    """
    Saves the latent representation of the given AnnData object.

    Parameters:
    - latent_path (str): The path to save the latent data.
    - latent_key (str): Name of the latent key.
    - adata (AnnData): AnnData object containing the data.
    """
    np.save(os.path.join(latent_path, f"{latent_key}.npy"), adata.obsm[latent_key])



def run_model_pipeline(Model, input_path_dict, build_model_dict, compile_dict, model_params, save_model, batch_col, bio_col, batch_col_categories=None, bio_col_categories=None, return_scores=False, return_adata_dict=False, return_trained_model=False,model_type="ae_da",issparse=False, load_dense=True,shape_color_dict={"celltype_vs_donor": {"shape_col": "celltype", "color_col": "donor"}},sample_size=None,return_history=False):
    """
    Runs the full pipeline from loading data to training the model and processing latent representations. This is useful for AE,AEC,AE_DA,AE_RE,AE_conv
    Parameters:
    - Model: The model class to be instantiated and trained.
    - input_path_dict (dict): Dictionary containing paths to the training, validation, and optionally test datasets.
    - build_model_dict (dict): Dictionary of model building parameters.
    - compile_dict (dict): Dictionary of model compiling parameters.
    - model_params: Parameters and configurations for the model.
    - save_model (bool): Flag to determine whether to save the model.
    - batch_col (str): Name of the batch column in the dataset.
    - bio_col (str): Name of the biological column in the dataset.
    - batch_col_categories (list): List of categories for one-hot encoding of batch column.
    - bio_col_categories (list): List of categories for one-hot encoding of biological column.
    - return_scores (bool): Flag to return clustering scores.
    - return_adata_dict (bool): Flag to return the updated AnnData dictionary.
    - return_trained_model (bool): Flag to return the trained model.
    - model_type (str): For plotting loss. Options: ["ae_da","ae"]
    - issparse (bool): data is saved as sparse npy array
    - load_dense (bool): If True, forces conversion of sparse arrays to dense format.
    - shape_color_dict (dict, optional): Dictionary with shape_col and color_col combinations for plotting.
    - sample_size (int): sample size used to calculate clustering scores on a random subset of the latent space
    - return history (bool): Flag to return the history dataframe


    Returns:
    - dict: A dictionary of results, which may include the trained model, clustering scores, AnnData dictionary and history dataframe.
    """
    import gc
    print("input path dict",input_path_dict)
    # 1. Load data in dense format
    adata_dict = load_data(input_path_dict, eval_test=model_params.eval_test, scaling=model_params.scaling,issparse=issparse, load_dense=load_dense)
    print("loaded adata, adata_dict keys check:",adata_dict.keys())

    print("Batches available: ", np.unique(adata_dict["train"].obs[batch_col]))

    # 2. Prepare input and output data for training
    data_dict = get_train_val_data(adata_dict, batch_col=batch_col, bio_col=bio_col, get_pred=model_params.get_pred, use_z=model_params.use_z, batch_col_categories=batch_col_categories, bio_col_categories=bio_col_categories,eval_test=model_params.eval_test)
    train_in, train_out = data_dict['train_in'], data_dict['train_out']
    val_in, val_out = data_dict['val_in'], data_dict['val_out']
    if model_params.eval_test:
        test_in = data_dict['test_in']

    # Check the shape of 'train_in'
    # If 'train_in' is a tuple and has more than one element, print the shape of each element
    if isinstance(train_in, tuple) and len(train_in) > 1:
        print("train_in shapes (x, z):", train_in[0].shape, ",", train_in[1].shape)
    else:
        print("train_in shape x:", train_in.shape)

    # Check the shape of 'train_out'
    # If 'train_out' is a tuple and has more than one element, print the shape of each element
    if isinstance(train_out, tuple) and len(train_out) > 1:
        print("train_out shapes (x, y):", train_out[0].shape, ",", train_out[1].shape)
    else:
        print("train_out shape x:", train_out.shape)


    if model_type =="aec": #because AEC call method returns a dict not a tuple
        train_out_dict = {
        'reconstruction_output': train_out[0],
        'classification_output': train_out[1]}
        val_out_dict = {
            'reconstruction_output': val_out[0],
            'classification_output': val_out[1]}

    # Get metadata for callbacks
    metadata_dict = None
    if model_params.compute_latents_callback:
        metadata_dict={}
        metadata_dict["train"] = adata_dict["train"].obs 
        metadata_dict["val"] = adata_dict["val"].obs 


    # 3. Build and train model
    model = Model(in_shape=adata_dict["train"].shape, **build_model_dict)
    model.compile(**compile_dict)
    # Train the model with appropriate data format
    if model_type == "aec":
        trained_model, history = train_and_save_model(model, train_in, train_out_dict, val_in, val_out_dict, model_params, save_model,metadata_dict)
    else:
        trained_model, history = train_and_save_model(model, train_in, train_out, val_in, val_out, model_params, save_model,metadata_dict)
    print(trained_model.summary())
    # if model_type == "ae_re":
    #     print(trained_model.re_encoder.summary())
    #     print(trained_model.re_decoder.summary())
    if model_type == "scmedalre":
        print("Available keys in re_layers:", trained_model.re_decoder.re_layers.keys())
        for key,layer in trained_model.re_decoder.re_layers.items():
            print(key," re layer summary\n",layer.summary())
    else:
        print(trained_model.encoder.summary())
        print(trained_model.decoder.summary())



    # 4. Plot Loss graph
    plot_params = {"outpath": model_params.plots_path}
    print("Starting plots")

    PlotLoss(history, model_params, save_model=save_model, model_type=model_type)

    # Ensure 'history' is in the correct format
    if isinstance(history, tf.keras.callbacks.History):
        history = history.history

    # Create a DataFrame from the history dictionary and save it to a CSV file
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(model_params.latent_path, "history.csv")  # corrected path and added file extension
    
    history_df["epochs"]=history_df.index
    history_df.to_csv(history_csv_path)

    print(f"History saved to {history_csv_path}")

    # Before starting intensive computation or after completing a significant data processing step
    gc.collect()
    # 5. Perform PCA if requested
    if model_params.get_pca:
        print("\ngetting pca")
        adata_dict = get_pca_andplot(adata_dict, plot_params, eval_test=model_params.eval_test,shape_color_dict=shape_color_dict,n_components=model_params.n_components)

    # Initialize z_ohe_dict
    z_ohe_dict = None

    # If model_type is 'ae_re', get z_ohe_dict
    if model_type == "scmedalre":
        z_ohe_dict = get_z_ohe_dict(adata_dict, batch_col, batch_col_categories)
        print("z_ohe_dict keys check:",z_ohe_dict.keys())
    print("\ngetting encoder")
    # 6. Process latent representations and calculate clustering scores
    encoder_method = trained_model.re_encoder if model_type == "scmedalre" else trained_model.encoder
    
    encoder_out = get_encoder_latentandscores(
        adata_dict=adata_dict,
        model_encoder=encoder_method,
        model_params=model_params,
        batch_col=batch_col,
        bio_col=bio_col,
        plot_params=plot_params,
        save_model=save_model,
        return_scores=return_scores,
        z_ohe_dict=z_ohe_dict,
        model_type=model_type,
        shape_color_dict=shape_color_dict,
        sample_size=sample_size)
    if return_scores:
        adata_dict, df_scores_dict = encoder_out
    else: 
        adata_dict = encoder_out

    # 6.1. Reconstruct input
    outputs = {
        "train": trained_model.predict(train_in, batch_size=model_params.batch_size),
        "val": trained_model.predict(val_in, batch_size=model_params.batch_size)
    }
    if model_params.eval_test:
        outputs["test"] = trained_model.predict(test_in, batch_size=model_params.batch_size)
    # ToDo: Here I need to add trained_model.predict(test_in) to outputs      

    # Check the model type and compute reconstructions accordingly
    for dataset_type in outputs:
        print("\nComputing reconstruction for ", dataset_type)
        output = outputs[dataset_type]

        if model_type == "ae":
            recon = output
        elif model_type == "aec":
            recon = output['reconstruction_output']
        elif model_type in ["scmedalfe","scmedalfec"]:
            if model_params.get_pred:
                recon, pred_class, pred_cluster = output
            else:
                recon, pred_cluster = output
        elif model_type == "scmedalre":
            recon = output['recon']

        # Ensure recon is a numpy array
        if not isinstance(recon, np.ndarray):
            recon = np.array(recon)

        # Save recon outputs to a file at the specified latent path
        np.save(os.path.join(model_params.latent_path, "recon_" + dataset_type), recon)
        
        # Calculate and print standard deviations of the original and reconstructed data
        std_X = np.std(adata_dict[dataset_type].X, axis=1, ddof=1)  # Std of original data
        print(f"{dataset_type} std of X:", std_X)
        print(f"{dataset_type} Mean of std of X:", np.mean(std_X))
        
        std_recon = np.std(recon, axis=1, ddof=1)  # Std of reconstructed data
        print(f"{dataset_type} std of recon(X):", std_recon)
        print(f"Mean of std of recon(X):", np.mean(std_recon))



    gc.collect()

    # 7. Collect results based on flags
    results = {}
    if return_trained_model:
        results["model"] = trained_model
    if return_scores:
        results["scores"] = df_scores_dict
    if return_adata_dict:
        results["adata"] = adata_dict
    if return_history:
        results["history"]=history_df

    #
    if model_type == "scmedalre" and getattr(model_params, 'get_cf_batch', False):
        print("\nComputing counterfactual for all batches")
        inputs_dict = {"train": train_in, "val": val_in}
        if model_params.eval_test:
            inputs_dict["test"] = test_in
        # Here: add test_in to inputs_dict
        for dataset, inputs in inputs_dict.items():
            x, z = inputs
            cols = z.shape[1]
            rows = z.shape[0]

            for i, z_batch in zip(range(cols), batch_col_categories):
                print("Getting recon for batch: ", z_batch)
                z_batch_matrix = np.zeros((rows, cols))
                z_batch_matrix[:, i] = 1

                # Assuming re_ae_model.predict takes a tuple (x, z_batch_matrix) and returns the reconstruction
                output_batch = trained_model.predict((x, z_batch_matrix))
                z_batch_recon = output_batch['recon']

                # Save the reconstructed batch matrix to a .npy file
                np.save(os.path.join(model_params.latent_path, f'recon_batch_{dataset}_{z_batch}.npy'), z_batch_recon)

                # Collect garbage to free up memory
                gc.collect()


    return results


def run_all_folds(Model, input_base_path, out_base_paths_dict, folds_list, run_name, model_params_dict, build_model_dict, compile_dict, save_model, batch_col, bio_col, batch_col_categories, bio_col_categories,model_type="ae_da",issparse=False, load_dense=True,shape_color_dict={"celltype_vs_donor": {"shape_col": "celltype", "color_col": "donor"}},sample_size=None):
    
    """
    Executes a model training pipeline across multiple folds, typically for cross-validation.

    Parameters:
    - Model: The model class to be instantiated and trained.
    - input_base_path: Base path for input data (train, validation, and optionally test are within base_path).
    - out_base_paths_dict (dict): Output paths for saving models, figures, and latent spaces.
    - folds_list: List of integers representing the fold numbers for cross-validation.
    - run_name: Name for the model run, used for saving outputs.
    - model_params_dict: Dictionary containing model parameters.
    - build_model_dict: Dictionary of parameters for building the model.
    - compile_dict: Dictionary of parameters for compiling the model.
    - save_model: Boolean indicating whether to save the model outputs.
    - batch_col: Name of the batch column in the dataset.
    - bio_col: Name of the biological column in the dataset.
    - batch_col_categories: Categories for the batch column for one-hot encoding.
    - bio_col_categories: Categories for the biological column for one-hot encoding.
    - model_type (str): For plotting loss. Options: ["ae_da","ae"]
    - issparse (bool): data is saved as sparse npy array
    - shape_color_dict (dict, optional): Dictionary with shape_col and color_col combinations for plotting.
    - sample_size (int): sample size used to calculate clustering scores on a random subset of the latent space

    Returns:
    - Dict of DataFrames: A dictionary containing the mean scores aggregated from all folds for each dataset type (train, val, test).
    """
    import time
    import gc
    # Initialize dictionaries to hold results for each dataset type
    all_scores:Dict[str,List[Any]] = {
        'train': [],
        'val': [],
        'test': []
    }
    #initialize empty dictionaries to collect results from all folds
    all_folds_adata = {}
    all_folds_model_params = {}
    all_history_df = pd.DataFrame()

    # Set return_scores_temp to True if you want to calculate scores within the fold loop
    return_scores_temp = False
    start_time_train = time.time()
    
    for intFold in folds_list:
        print(f"\n\nRunning Fold {intFold}\n\n")

        # Update model parameters for the current fold
        model_manager = ModelManager(params_dict=model_params_dict, base_paths_dict=out_base_paths_dict, run_name=run_name, save_model=save_model, use_kfolds=True, kfold=intFold)
        model_params = model_manager.params
        model_manager.print_params()

        # Get paths_dict with train, test and val paths
        input_path_dict = get_split_paths(base_path=input_base_path, fold=intFold)
        print("\ninput_path_dict:\n", input_path_dict)
        # print("input_path_dict keys:", input_path_dict.keys())


        # Run pipeline for the current fold
        fold_results = run_model_pipeline(Model=Model,
                                          input_path_dict=input_path_dict,
                                          build_model_dict=build_model_dict,
                                          compile_dict=compile_dict,
                                          model_params=model_params,
                                          save_model=save_model,
                                          batch_col=batch_col,
                                          bio_col=bio_col,
                                          batch_col_categories=batch_col_categories,
                                          bio_col_categories=bio_col_categories,
                                          return_scores=return_scores_temp, 
                                          return_adata_dict=True, 
                                          return_trained_model=False,
                                          model_type=model_type,
                                          issparse=issparse,
                                          load_dense=load_dense,
                                          shape_color_dict=shape_color_dict,
                                          sample_size = sample_size,
                                          return_history=True)
        
        # add adata dictionaries per fold to all_folds_adata dict
        print("Output fold adata",fold_results["adata"])
        all_folds_adata[intFold] = fold_results["adata"]
        # add model params to all_folds_model_params dict
        all_folds_model_params[intFold] = model_params


        # Process history if available in fold_results
        if "history" in fold_results:
            history_df = pd.DataFrame(fold_results["history"])
            history_df['fold'] = intFold
            all_history_df = pd.concat([all_history_df, history_df])
            



        # Reminder: Set return_scores_temp to True if you want to calculate scores within the fold loop
        if return_scores_temp:
            # Append the scores for each dataset type to the respective list in all_scores
            for dataset_type in all_scores.keys():
                if dataset_type in fold_results['scores']:
                    scores_df = fold_results['scores'][dataset_type]
                    scores_df['fold'] = intFold
                    scores_df['dataset_type'] = scores_df.index
                    all_scores[dataset_type].append(scores_df)
    total_time_train = time.time() - start_time_train

    print(f"\n\nTotal time for training all folds and getting latent spaces: {total_time_train} seconds")
        

    # save history in a single df
    all_history_df.to_csv(os.path.join(model_params.latent_path_main, f"history_allfolds.csv"))
    average_history_df = all_history_df.groupby('epochs').mean() 
    average_history_df.to_csv(os.path.join(model_params.latent_path_main, f"mean_history_allfolds.csv"))
    PlotLoss(average_history_df, model_params, save_model=save_model, model_type=model_type,average_curve=True)
    print("Pipeline finished running for all folds")
    # clean trash
    gc.collect()
    print("\n\nComputing scores for different sampel sizes")
 
    print("sample size",sample_size)
    # if return_scores_temp==False the scores are calculated after training all models
    if return_scores_temp ==False:
        print("\nStarted iteration through the folds to calculate scores..")
        start_time_scores = time.time()
        # Loop through each fold and dataset type to calculate scores for each latent space representation
        for intFold in folds_list:
            for dataset_type, adata_subset in all_folds_adata[intFold].items():
                latent_list = list(adata_subset.obsm.keys())
                # for latent_name in adata_subset.obsm.keys():
                print(f"\n\nProcessing clustering scores {latent_list} for dataset {dataset_type} in fold {intFold}..")
                scores_df = calculate_merge_scores(latent_list=latent_list, 
                                                        adata=adata_subset, 
                                                        labels=[batch_col, bio_col], 
                                                        sample_size=sample_size)
                scores_df.to_csv(os.path.join(all_folds_model_params[intFold].latent_path, f"scores_{dataset_type}_samplesize-{sample_size}.csv"))
                print(f"Scores calculated for {latent_list} on {dataset_type} dataset in fold {intFold}")
                scores_df['fold'] = intFold
                scores_df['dataset_type'] = scores_df.index
                all_scores[dataset_type].append(scores_df)
        #count time
        total_time_scores = time.time() - start_time_scores
        print("\nScores obtained for all folds")
        print(f"\n\nTotal computation time for clustering scores: {total_time_scores} seconds")
            


        # Process all scores for each dataset type and save the results
        mean_scores_dict = {}
        for dataset_type, scores_list in all_scores.items():
            if scores_list:  # Check if there are scores to process
                print("Averaging scores for ",dataset_type)
                # Concatenate all results for the dataset type into a single DataFrame
                df_all_results = pd.concat(scores_list, ignore_index=True)
                # Calculate the mean across all rows (folds)
    #            mean_scores = df_all_results.mean()
                print("\n\n mean scores",df_all_results)
                if not (model_params.get_pca or model_params.get_baseline):
                    # calculate mean
                    mean_scores = df_all_results.mean(numeric_only=True).to_frame('mean')
                    print("\n\n mean scores",mean_scores)
                    # fill the dict
                    mean_scores_dict[dataset_type] = mean_scores

                    # Calculate sample standard deviation scores
                    std_scores = df_all_results.std(numeric_only=True,ddof=1).to_frame('std')  # Using sample standard deviation
                    print("\n\n std scores",std_scores)
                    # Calculate standard error of the mean (SEM)
                    se_scores = std_scores / (len(folds_list) ** 0.5)  # SEM calculation

                    # Combine mean, std, and se into a single DataFrame
                    summary_df = pd.concat([mean_scores, std_scores, se_scores], axis=1)
                    summary_df.columns = ['mean', 'std', 'sem']


                else:

                    grouped = df_all_results.groupby('dataset_type')
                    print(grouped)
                    mean_scores = grouped.mean()
                    print(mean_scores)
                    std_scores = grouped.std(ddof=1)
                    print(std_scores)
                    # Calculate SEM correctly aligning DataFrame and Series indices
                    sem_scores = std_scores.div(np.sqrt(grouped.size()), axis='index').rename(columns=lambda x: 'sem_' + x)

                    # Combine mean, std, and sem into a single DataFrame, preserving multi-level column structure
                    summary_df = pd.concat({'mean': mean_scores, 'std': std_scores, 'sem': sem_scores}, axis=1)

                # Display the final DataFrame
                print("\nSummary scores for all 5 folds:\n",summary_df)


                # Save results if required
                if save_model:
                    all_scores_path = os.path.join(model_params.latent_path_main, f'all_scores_{dataset_type}_samplesize-{sample_size}.csv')
                    
                    df_all_results.to_csv(all_scores_path)
    #                if not (model_params.get_pca or model_params.get_baseline):
                    mean_scores_path = os.path.join(model_params.latent_path_main, f'mean_scores_{dataset_type}_samplesize-{sample_size}.csv')
                    summary_df.to_csv(mean_scores_path)
    if not (model_params.get_pca or model_params.get_baseline):
        return mean_scores_dict

def get_scores_all_folds(input_base_path, out_base_paths_dict, folds_list, run_name, model_params_dict, save_model, batch_col, bio_col, issparse=False, load_dense=True, sample_size=None):
    # Initialize dictionaries to hold results for each dataset type

    """
    Calculate clustering scores for all folds and datasets, and summarize the results.

    Parameters:
    - input_base_path (str): Base path for input data.
    - out_base_paths_dict (dict): Dictionary containing paths for output data.
    - folds_list (list): List of fold numbers to process.
    - run_name (str): Name of the run for identification purposes.
    - model_params_dict (dict): Dictionary of model parameters.
    - save_model (bool): Flag to determine whether to save the model.
    - batch_col (str): Column name for batch labels.
    - bio_col (str): Column name for biological labels.
    - issparse (bool): Flag indicating whether the input data is sparse. Default is False.
    - load_dense (bool): Flag indicating whether to load dense data. Default is True.
    - sample_size (int, optional): Sample size for score calculations. Default is None.

    Returns:
    - all_scores (dict): Dictionary containing scores for each dataset type ('train', 'val', 'test') and fold.
    - combined_summary_stats (pd.DataFrame): DataFrame summarizing the scores with mean, standard deviation, SEM, and 95% confidence interval.
    """
    all_scores = {
        'train': [],
        'val': [],
        'test': []
    }

    
    for intFold in folds_list:
        print(f"\n\nRunning Fold {intFold}\n\n")

        # Update model parameters for the current fold
        model_manager = ModelManager(params_dict=model_params_dict,
                                        base_paths_dict=out_base_paths_dict,
                                        run_name=run_name,
                                        save_model=save_model,
                                        use_kfolds=True,
                                        kfold=None,
                                        run_model=False,
                                        get_baseline_scores=True)
        model_params = model_manager.params
        model_manager.print_params()

        # Get paths_dict with train, test and val paths
        input_path_dict = get_split_paths(base_path=input_base_path, fold=intFold)
        print("\ninput_path_dict:\n", input_path_dict)

        # 1. Load data in dense format
        adata_dict = load_data(input_path_dict, eval_test=model_params.eval_test, scaling=model_params.scaling, issparse=issparse, load_dense=load_dense)
        for dataset_type, adata in adata_dict.items():
            df_scores = get_clustering_scores_optimized(adata, use_rep="X", labels=[model_params.batch_col, model_params.bio_col], sample_size=sample_size)
            df_scores['fold'] = intFold
            df_scores['dataset_type'] = dataset_type
            all_scores[dataset_type].append(df_scores)


    # Combine all scores into a single DataFrame for each dataset type
    combined_scores = {k: pd.concat(v) for k, v in all_scores.items()}

    # Debugging: Print combined scores to check structure
    for dataset_type, df in combined_scores.items():
        print(f"\nCombined scores for {dataset_type}:\n", df.head())
        df.to_csv(f"{model_params.Baseline_scores_path}/all_scores_{dataset_type}.csv")

    summary_stats = []
    for dataset_type, df in combined_scores.items():
        print(f"Processing dataset_type: {dataset_type}")
        print(f"Available columns in df: {df.columns}")
        label_df = df[df['dataset_type'] == dataset_type]

        for score in ['db', '1/db', 'ch', 'silhouette']:
            if score not in label_df.index:
                print(f"Score '{score}' not found in DataFrame for dataset_type: {dataset_type}")
                continue

            score_series = label_df.loc[score]
            score_mean = score_series.mean()
            score_std = score_series.std(ddof=1)
            score_sem = score_series.sem(ddof=1)
            score_95ci = score_sem * 1.96  # For 95% CI

            summary_stats.append({
                'dataset_type': dataset_type,
                'score': score,
                f'mean_{batch_col}': score_mean[batch_col],
                f'std_{batch_col}': score_std[batch_col],
                f'sem_{batch_col}': score_sem[batch_col],
                f'95ci_lower_{batch_col}': score_mean[batch_col] - score_95ci[batch_col],
                f'95ci_upper_{batch_col}': score_mean[batch_col] + score_95ci[batch_col],
                f'mean_{bio_col}': score_mean[bio_col],
                f'std_{bio_col}': score_std[bio_col],
                f'sem_{bio_col}': score_sem[bio_col],
                f'95ci_lower_{bio_col}': score_mean[bio_col] - score_95ci[bio_col],
                f'95ci_upper_{bio_col}': score_mean[bio_col] + score_95ci[bio_col],
            })
    combined_summary_stats = pd.DataFrame(summary_stats)
    combined_summary_stats.to_csv(f"{model_params.Baseline_scores_path}/summary_stats_all.csv", index=False)

    return all_scores, combined_summary_stats


def get_metric2optimizemodel(mean_scores, subset='val', metric='silhouette', batch_col='donor', bio_col='celltype'):
    """
    Calculates a metric to optimize a model by maximizing biological clustering and minimizing batch clustering.

    Parameters:
    - mean_scores (DataFrame): DataFrame containing mean scores (mean across folds).
    - subset (str): Subset of data to consider (e.g., 'train', 'val', 'test').
    - metric (str): Metric to be used for optimization, typically 'silhouette'.
    - batch_col (str): Column name for batch data.
    - bio_col (str): Column name for biological data.

    Returns:
    - float: Calculated metric value to optimize.
    """
    batch_mean = mean_scores[subset].loc[(batch_col, metric), 'mean']
    bio_mean = mean_scores[subset].loc[(bio_col, metric), 'mean']

    # Aim to maximize biological mean and minimize batch mean
    metric2optimize = bio_mean - batch_mean
    return metric2optimize

def get_metric2optimize_re(mean_scores, subset='val', metric='silhouette', batch_col='donor'):
    """
    Calculates a metric to optimize a model by maximizing biological clustering and minimizing batch clustering.

    Parameters:
    - mean_scores (DataFrame): DataFrame containing mean scores (mean across folds).
    - subset (str): Subset of data to consider (e.g., 'train', 'val', 'test').
    - metric (str): Metric to be used for optimization, typically 'silhouette'.
    - batch_col (str): Column name for batch data.
    - bio_col (str): Column name for biological data.

    Returns:
    - float: Calculated metric value to optimize.
    """
    batch_mean = mean_scores[subset].loc[(batch_col, metric), 'mean']
    # bio_mean = mean_scores[subset].loc[(bio_col, metric), 'mean']

    # Aim to maximize biological mean and minimize batch mean
    metric2optimize = batch_mean
    return metric2optimize





def analyze_ray_tune_results(file_path, metric):

    from ray.tune.analysis import ExperimentAnalysis
    """
    Analyzes Ray Tune experiment results, prints the best configuration, 
    and plots a histogram of the specified metric.

    Parameters:
    - file_path (str): Path to the Ray Tune experiment results.
    - metric (str): The metric to analyze and plot.

    Returns:
    - None
    """
    # Load the experiment analysis
    analysis = ExperimentAnalysis(file_path)

    # Get the best trial information
    best_trial = analysis.get_best_trial(metric, mode="max", scope="all")
    best_config = best_trial.config
    best_scores = best_trial.last_result[metric]
    print(f"Best configuration: {best_config}")
    print(f"The {metric} for the best trial is: {best_scores}")

    # Generate a dataframe for the specified metric
    df = analysis.dataframe(metric=metric, mode="max")
    mean_scores = df[metric]

    # Plot histogram
    plt.hist(mean_scores, bins=10, edgecolor='black')  # Adjust the number of bins as needed
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {metric}')
    plt.savefig(file_path + "/HPO_scores.png")
    plt.show()


def get_latent_spaces_paths(models_list, common_params_dict, outputs_path, function_type="1DGMM", folder_name="20ct-20donor/1-12-1", model_params=None):
    """
    Gets latent space paths for specified models and datasets based on common parameters.
    This is useful for retriving the latent space paths after applying: AE_DA,AEC_DA,AE_RE,AE_conv

    Parameters:
    - models_list (list): List of model names. For example: ["AE_RE", "AE_DA", "AEC_DA", "AEC", "AE_conv"]
    - common_params_dict (dict): Dictionary mapping models to their common parameters. Note: map the models according to the name of the latent space stored, which indicates parameters from model_params_dict
    - outputs_path (str): Base path to the outputs directory.
    - function_type (str): Type of function or analysis (default: "1DGMM").
    - folder_name (str): Name of the folder containing analysis or results (default: "20ct-20donor/1-12-1").
    - model_params (Namespace): Generate using ModelManager.params (default: None).

    Returns:
    - latent_path_dict (dict): Dictionary of latent paths for each model, fold, and dataset.
    """
    
    # Initialize the dictionary to store latent paths
    latent_path_dict = {}
    dataset_list = ["train", "val"]

    # Check if evaluation on the 'test' dataset is enabled in the model parameters
    # if model_params_dict and model_params_dict.get("eval_test", False):
    if model_params.eval_test:
        # If evaluation on 'test' is enabled, add 'test' to the list of datasets
        dataset_list.append("test")

    # Iterate through each model
    for model in models_list:
        # Construct the base path for latent space using function_type and folder_name
        base_latent_path = os.path.join(outputs_path, "latent_space", function_type, folder_name, model)
        # Construct the full pattern for the latent space paths
        latent_path_pattern = f"run_crossval_2X_*{common_params_dict[model]}*"
        full_latent_path = os.path.join(base_latent_path, latent_path_pattern)
        
        latent_path_dict[model] = {}
        
        # Iterate through folds within the model's latent path
        for fold in glob.glob(full_latent_path + "/splits*"):
            fold_name = fold.split("/")[-1]
            latent_path_dict[model][fold_name] = {}
            
            # Iterate through datasets: train, val, test
            for dataset in dataset_list:
                # Store the paths for each dataset
                dataset_path_pattern = os.path.join(fold, f"*{dataset}.npy")
                latent_path_dict[model][fold_name][dataset] = glob.glob(dataset_path_pattern)

    # Check the number of paths in each model and fold
    print("\nLatent space paths check:")
    for model, fold_dict in latent_path_dict.items():
        print(f"Model: {model}")
        for fold_name, dataset_dict in fold_dict.items():
            for dataset, paths in dataset_dict.items():
                count = len(paths)
                print(f"Fold: {fold_name}, Dataset: {dataset.capitalize()}, Paths: {count}")
        print()
    return latent_path_dict

def load_latent_spaces(base_path, fold, models_list, latent_path_dict, model_params, batch_col, bio_col, batch_col_categories, bio_col_categories,issparse=False, load_dense=False):
    """
    Loads and stores latent spaces for specified models and datasets, and retrieves 'y' and 'z' components. This is useful for retriving the save latent space after applying: AE_DA,AEC_DA,AE_RE,AE_conv
    Parameters:
    - base_path (str): Base path to the data.
    - fold (int): The specific fold of the data.
    - models_list (list): List of model names.
    - latent_path_dict (dict): Dictionary mapping models and datasets to their latent space paths. You can obtain it using the function get_latent_spaces_paths.
    - model_params (Namespace object): The parameters of the model, used for scaling loss values. Created using ModelManager class.
    - batch_col (str): The batch column name used for retrieving 'z' component.
    - bio_col (str): The biological column name used for retrieving 'y' component.
    - batch_col_categories (list): Categories for the batch column to be used in one-hot encoding.
    - bio_col_categories (list): Categories for the biological column to be used in one-hot encoding.
    - issparse(bool): True if X is in sparse array, False if its dense
    - load_dense (bool): If True, forces conversion of sparse arrays to dense format.



    Returns:
    - adata_dict (dict): Dictionary of AnnData objects updated with latent spaces and 'y', 'z' components.
    """
    
    # Load initial dataset paths and data
    input_path_dict = get_split_paths(base_path=base_path, fold=fold)
    adata_dict = load_data(input_path_dict, eval_test = model_params.eval_test, scaling = model_params.scaling,issparse=issparse, load_dense=load_dense)

    # Initialize dataset list and add 'test' dataset if evaluation on 'test' is enabled
    dataset_list = ["train", "val"]
    if model_params.eval_test:
        dataset_list.append("test")
        
    # Iterate through each model and dataset
    for model in models_list:
        for dataset in dataset_list:
            # Load the latent space for the current model and dataset
            latent_space_path = latent_path_dict[model]["splits_" + str(fold)][dataset]
            if isinstance(latent_space_path, (np.ndarray,list)):
                latent_space_path = latent_space_path[0]
            latent_space = np.load(latent_space_path)

            latent_key = f"{model}_latent_{dataset}"
            # save latent space in .obsm as latent_key
            adata_dict[dataset].obsm[latent_key] = latent_space
  

    # Iterate through datasets to retrieve 'y' and 'z' components
    for dataset in dataset_list:
        x_y_z_dict = get_x_y_z(adata_dict[dataset], batch_col, bio_col, 
                               batch_col_categories, bio_col_categories, use_rep="X")
        adata_dict[dataset+'_y'] = x_y_z_dict['y']
        adata_dict[dataset+'_z'] = x_y_z_dict['z']

    return adata_dict

def prepare_latent_space_inputs(adata_dict, latent_keys_config, eval_test=False):
    """
    Prepares the latent space inputs for training, validation, and optionally testing based on the provided configuration.
    This would be useful for the mixed ffect classifier.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects for 'train', 'val', and optionally 'test' datasets.
    - latent_keys_config (dict): Configuration dictionary indicating which latent space to use for fixed effects (fe_latent)
                                 and random effects (re_latent) from all the models' latent spaces.
    - eval_test (bool): Flag indicating whether to include the 'test' dataset.

    Returns:
    - inputs (dict): Dictionary of inputs with latent space data for training, validation, and optionally testing.
    """
    # Initialize the dictionary to store inputs for training and validation, include 'test' if eval_test is True
    inputs = {'train': {}, 'val': {}}
    if eval_test:
        if 'test' in adata_dict:
            inputs['test'] = {}
        else:
            print("Warning: 'test' is set to True in eval_test, but 'test' data is not in adata_dict.")


    # Iterate through the configuration of latent keys (fixed and random effects)
    for latent_type, base_latent_key in latent_keys_config.items():
        # Iterate through the datasets in the inputs dictionary (train, val, and optionally test)
        for dataset_type in inputs.keys():
            # Construct the full latent key by appending the dataset type (train, val, or test)
            full_latent_key = f"{base_latent_key}_{dataset_type}"

            # Check if the full latent key exists in the obsm attribute of adata_dict
            if full_latent_key in adata_dict[dataset_type].obsm:
                # Store the latent space data in the inputs dictionary
                inputs[dataset_type][latent_type] = adata_dict[dataset_type].obsm[full_latent_key]
            else:
                # Handle the case where the full latent key is not found
                print(f"Key not found: {full_latent_key} in adata_dict['{dataset_type}'].obsm")
    
    return inputs

def evaluate_model(trained_model, inputs, adata_dict, model_params,metric_name ="CategoricalAccuracy"):
    """
    Evaluates the trained model on train, validation, and optionally test datasets, and returns results in a DataFrame.

    Parameters:
    - trained_model: The trained model to be evaluated.
    - inputs (dict): Dictionary containing input data for each dataset type.
    - adata_dict (dict): Dictionary of AnnData objects or y values for 'train', 'val', and 'test' datasets.
    - model_params: Object containing model parameters including batch_size and eval_test flag.

    Returns:
    - results_df (DataFrame): DataFrame containing the loss and metric for each evaluated dataset.
    """
    
    # Define the datasets to evaluate
    datasets_to_evaluate = ['train', 'val']
    if getattr(model_params, 'eval_test', False):  # Use getattr to avoid AttributeError if eval_test is not set
        datasets_to_evaluate.append('test')

    # Create an empty DataFrame to store the results
    metrics_df = pd.DataFrame(columns=['Dataset', 'Loss', metric_name])

    # Evaluate the model on each dataset
    for dataset_type in datasets_to_evaluate:
        # Extract the inputs and outputs based on dataset_type
        inputs_data = inputs[dataset_type]
        outputs_data = adata_dict[f'{dataset_type}_y']  # Assuming your outputs are stored like this

        # Evaluate the model on the current dataset
        loss, metric = trained_model.evaluate(inputs_data, outputs_data, batch_size=model_params.batch_size)  # Default batch_size to 32 if not set

        currdf = pd.DataFrame().from_dict({
            'Dataset': [dataset_type],
            'Loss': [loss],
            metric_name: [metric]
        })
        # Append the results to the DataFrame
        metrics_df = pd.concat([metrics_df, currdf], ignore_index=True)

    metrics_df.to_csv(os.path.join(model_params.latent_path, "metrics.csv"))

    return metrics_df


# def run_model_pipeline_LatentClassifier(Model, latent_path_dict, build_model_dict, compile_dict, model_params, save_model, 
#                                         batch_col, bio_col, base_path, fold, models_list, latent_keys_config,
#                                         batch_col_categories=None, bio_col_categories=None, return_scores=False, 
#                                         return_adata_dict=False, return_trained_model=False, model_type="mec",
#                                         issparse=False, load_dense=False):
#     """
#     Runs the complete model pipeline, including loading data, training, evaluation, and obtaining scores.

#     Parameters:
#     - Model: The model class to be used.
#     - latent_path_dict: Dictionary containing the paths to the latent spaces.
#     - build_model_dict: Dictionary containing the parameters for model building.
#     - compile_dict: Dictionary containing the parameters for model compilation.
#     - model_params: Object containing model parameters and configurations.
#     - save_model: Flag indicating whether to save the trained model.
#     - batch_col: Name of the batch column.
#     - bio_col: Name of the biological column.
#     - base_path: The base path for the dataset and model.
#     - fold: The specific fold of the data being used.
#     - models_list: List of models being used.
#     - latent_keys_config: Configuration for the latent keys.
#     - batch_col_categories: Categories for the batch column.
#     - bio_col_categories: Categories for the biological column.
#     - return_scores: Flag indicating whether to return clustering scores.
#     - return_adata_dict: Flag indicating whether to return the AnnData dictionary.
#     - return_trained_model: Flag indicating whether to return the trained model.
#     - model_type: Type of the model (default "mec").
#     - issparse(bool): True if X is in sparse array, False if its dense
#     - load_dense (bool): If True, forces conversion of sparse arrays to dense format.

#     Returns:
#     - results: Dictionary containing the trained model, metrics, scores, and/or adata_dict based on the provided flags.
#     """
#     # 1. Load data latent paths and adata_dict
#     adata_dict = load_latent_spaces(base_path, fold, models_list, latent_path_dict, model_params, batch_col, bio_col, batch_col_categories, bio_col_categories,issparse, load_dense)

#     print("Batches available: ", np.unique(adata_dict["train"].obs[batch_col]))

#     # 2. Prepare data for training
#     inputs = prepare_latent_space_inputs(adata_dict, latent_keys_config, eval_test=model_params.eval_test)

#     # 3. Build and train model
#     me_model = Model(**build_model_dict)
#     me_model.compile(**compile_dict)
#     trained_model, history = train_and_save_model(me_model, train_in=inputs['train'], train_out=adata_dict['train_y'], val_in=inputs['val'], val_out=adata_dict['val_y'], model_params=model_params, save_model=save_model)

#     # 4. Plot Loss graph
#     plot_params = {"outpath": model_params.plots_path}
#     PlotLoss(history, model_params, save_model=save_model, model_type=model_type)

#     # 5. Evaluate the model and get metrics
#     metrics_df = evaluate_model(trained_model, inputs, adata_dict, model_params)

#     # 6. Get latent scores using the model's encoder
#     encoder_method = trained_model.encoder
#     adata_dict, df_scores_dict = get_encoder_latentandscores(
#         adata_dict=adata_dict,
#         model_encoder=encoder_method,
#         model_params=model_params,
#         batch_col=batch_col,
#         bio_col=bio_col,
#         plot_params=plot_params,
#         save_model=save_model,
#         return_scores=return_scores,
#         z_ohe_dict=None,
#         model_type=model_type,
#         other_inputs=inputs
#     )

#     # 7. Collect results based on flags
#     results = {}
#     if return_trained_model:
#         results["model"] = trained_model
#     if return_scores:
#         results["metrics"] = metrics_df
#         results["scores"] = df_scores_dict
#     if return_adata_dict:
#         results["adata"] = adata_dict

#     return results






def evaluate_model_v2(trained_model, inputs, adata_dict, model_params, metric_name="CategoricalAccuracy", **kwargs):
    from sklearn.metrics import balanced_accuracy_score
    # Define the datasets to evaluate
    datasets_to_evaluate = ['train', 'val']
    if getattr(model_params, 'eval_test', False):
        datasets_to_evaluate.append('test')

    # Create an empty DataFrame to store the results
    metrics_df = pd.DataFrame(columns=['Dataset', 'Loss', f"{metric_name}_dffn", "Balanced_Accuracy_dffn"])

    # Evaluate the model on each dataset
    for dataset_type in datasets_to_evaluate:
        inputs_data = inputs[dataset_type]
        outputs_data = adata_dict[f'{dataset_type}_y']

        # Evaluate the model on the current dataset
        loss, metric = trained_model.evaluate(inputs_data, outputs_data, batch_size=model_params.batch_size)

        # Predict labels
        y_pred = trained_model.predict(inputs_data, batch_size=model_params.batch_size)
        predicted_classes = y_pred.argmax(axis=1)
        true_classes = outputs_data.values.argmax(axis=1)

        # Calculate balanced accuracy
        balanced_acc = balanced_accuracy_score(true_classes, predicted_classes)

        # Save predictions to adata_dict
        num_classes = y_pred.shape[1]
        y_pred_ohe = np.eye(num_classes)[predicted_classes]
        y_pred_df = pd.DataFrame(y_pred_ohe, columns=outputs_data.columns)
        adata_dict[f'{dataset_type}'].obs["true_labels"] = [outputs_data.columns[ind] for ind in true_classes]
        adata_dict[f'{dataset_type}'].obs["pred_labels"] = [y_pred_df.columns[ind] for ind in predicted_classes]
        adata_dict[f'{dataset_type}'].obs.to_csv(os.path.join(model_params.latent_path, f"y_pred_{dataset_type}.csv"))

        currdf = pd.DataFrame().from_dict({
            'Dataset': [dataset_type],
            'Loss':[loss],
            f'{metric_name}_dffn': [metric],
            'Balanced_Accuracy_dffn': [balanced_acc]  # Add balanced accuracy here
        })
        # Append the results to the DataFrame
        metrics_df = pd.concat([metrics_df, currdf], ignore_index=True)

    return {
        "metrics": metrics_df,
        "adata_dict": adata_dict
    }



def build_train_evaluate_model(Model,build_model_dict, compile_dict, inputs, adata_dict, model_params, save_model, model_type, *args,**kwargs):
    """
    Builds, trains, and evaluates a machine learning model, with options to save the model and plot training history.

    Parameters:
    - Model: The model class to instantiate for training.
    - build_model_dict: Dictionary containing the parameters for building the model instance.
    - compile_dict: Dictionary containing the parameters for compiling the model (e.g., optimizer, loss function).
    - inputs: Dictionary containing the input data for training and validation, with keys 'train' and 'val'.
    - adata_dict: Dictionary containing the AnnData objects, including training and validation labels ('train_y' and 'val_y').
    - model_params: Object containing additional parameters for training, such as epochs, batch size, and paths.
    - save_model: Boolean flag indicating whether to save the trained model to disk.
    - model_type: String specifying the type of model (used for saving or plotting).

    Returns:
    - dict: A dictionary containing:
        - "model": The trained model instance.
        - "history": The training history, which includes loss and metric values across epochs.
        - "metrics": A DataFrame containing evaluation metrics for the model.
        - "adata_dict": Updated AnnData dictionary with evaluation results.
    """
    # 3. Build and train model
    print(build_model_dict)
    print(vars(model_params))

    me_model = Model(**build_model_dict)
    me_model.compile(**compile_dict)
    trained_model, history = train_and_save_model(
        me_model,
        train_in=inputs['train'],
        train_out=adata_dict['train_y'],
        val_in=inputs['val'],
        val_out=adata_dict['val_y'],
        model_params=model_params,
        save_model=save_model
    )

    # 4. Plot Loss graph
    plot_params = {"outpath": model_params.plots_path}
    PlotLoss(history, model_params, save_model=save_model, model_type=model_type)

    # 5. Evaluate the model and get metrics
    results = evaluate_model_v2(trained_model, inputs, adata_dict, model_params)
    metrics_df = results["metrics"]

    return {"model":trained_model,"history":history,"metrics":metrics_df,"adata_dict":results["adata_dict"]}



def svm_accuracy_and_predictions(inputs, adata_dict, model_params, eval_test=False):
    """
    Trains an SVM classifier using concatenated latent space features, evaluates it on train, validation, 
    and optionally the test datasets, and returns accuracy metrics along with updated AnnData objects 
    containing SVM predictions.

    Parameters:
    - inputs (dict): Dictionary containing input data for each dataset type ('train', 'val', 'test'). 
                     It includes keys 'fe_latent' and optionally 're_latent' for feature and regularization latent spaces.
    - adata_dict (dict): Dictionary containing AnnData objects or ground truth y values for 'train', 'val', and 'test' datasets.
    - model_params: Object containing model parameters, including the path to save predictions.
    - eval_test (bool): Boolean flag indicating whether to evaluate the model on the test set (default: False).

    Returns:
    - dict: A dictionary containing:
        - "metrics": DataFrame with accuracy and balanced accuracy metrics for the SVM classifier across train, validation, and optionally test datasets.
        - "adata_dict": Updated AnnData dictionary with true and predicted labels for each dataset, labeled with SVM predictions.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.svm import SVC


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
        adata_dict[dataset_type].obs["true_labels_svm"] = outputs_data.columns[outputs_data.values.argmax(axis=1)]
        adata_dict[dataset_type].obs["pred_labels_svm"] = y_pred_df.columns[y_pred_df.values.argmax(axis=1)]
        adata_dict[dataset_type].obs.to_csv(os.path.join(model_params.latent_path, f"y_pred_{dataset_type}_svm.csv"))

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

    # Train the SVM classifier
    clf = SVC(kernel='rbf', gamma='scale', C=1.0)
    clf.fit(X_train, y_train)
    num_classes = adata_dict["train_y"].shape[1]

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
        "SVMAccuracy": [train_results["accuracy"], val_results["accuracy"]],
        "SVMBalancedAccuracy": [train_results["balanced_accuracy"], val_results["balanced_accuracy"]]
    })

    # Evaluate on the test set if eval_test is True
    if eval_test:
        X_test = concatenate_features("test")
        X_test = scaler.transform(X_test)
        y_test = label_encoder.transform(adata_dict["test_y"].values.argmax(axis=1))
        test_results = process_dataset("test", X_test, y_test)
        test_df = pd.DataFrame().from_dict({
            "Dataset": ["test"],
            "SVMAccuracy": [test_results["accuracy"]],
            "SVMBalancedAccuracy": [test_results["balanced_accuracy"]]
        })

        metrics_df = pd.concat([metrics_df, test_df], ignore_index=True)
        adata_dict["test"] = test_results["adata_dict"]["test"]

    adata_dict["train"] = train_results["adata_dict"]["train"]
    adata_dict["val"] = val_results["adata_dict"]["val"]

    return {"metrics": metrics_df, "adata_dict": adata_dict}







def random_forest_accuracy_and_predictions(inputs, adata_dict, model_params, eval_test=False):
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
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    num_classes = adata_dict["train_y"].shape[1]

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

        currdf = pd.DataFrame().from_dict({
            "Dataset": ["test"],
            "RFAccuracy": [test_results["accuracy"]],
            "RFBalancedAccuracy": [test_results["balanced_accuracy"]]
        })
        # Append the results to the DataFrame
        metrics_df = pd.concat([metrics_df, currdf], ignore_index=True)

        adata_dict["test"] = test_results["adata_dict"]["test"]

    adata_dict["train"] = train_results["adata_dict"]["train"]
    adata_dict["val"] = val_results["adata_dict"]["val"]

    return {"metrics": metrics_df, "adata_dict": adata_dict}




def dummy_classifier_chance_accuracy(inputs, adata_dict, model_params, eval_test=False,seed = 42):
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

        currdf = pd.DataFrame().from_dict({
            "Dataset": ["test"],
            "ChanceAccuracy": [test_results["chance_accuracy"]],
        })
        # Append the results to the DataFrame
        metrics_df = pd.concat([metrics_df, currdf])
        adata_dict["test"] = test_results["adata_dict"]["test"]

    adata_dict["train"] = train_results["adata_dict"]["train"]
    adata_dict["val"] = val_results["adata_dict"]["val"]

    return {"metrics": metrics_df, "adata_dict": adata_dict}




# def run_model_pipeline_LatentClassifier_v2(Model, latent_path_dict, build_model_dict, compile_dict, model_params, save_model, 
#                                            batch_col, bio_col, base_path, fold, models_list, latent_keys_config,
#                                            batch_col_categories=None, bio_col_categories=None, return_metrics=True, 
#                                            return_adata_dict=False, return_trained_model=False, model_type="mec",
#                                            issparse=False, load_dense=False,seed=None):
#     """
#     Runs the complete model pipeline, including data loading, model training, evaluation, and metric collection.

#     Parameters:
#     - Model: The model class to be instantiated and trained.
#     - latent_path_dict: Dictionary containing paths to latent space data for each model.
#     - build_model_dict: Dictionary of parameters for building the model.
#     - compile_dict: Dictionary of parameters for compiling the model.
#     - model_params: Object containing additional model parameters and configurations.
#     - save_model: Boolean flag indicating whether to save the trained model to disk.
#     - batch_col: Name of the column representing batch information in the data.
#     - bio_col: Name of the column representing biological information in the data.
#     - base_path: Base directory path for datasets and model output.
#     - fold: The specific fold identifier for cross-validation or data splitting.
#     - models_list: List of models to be used in the pipeline.
#     - latent_keys_config: Configuration dictionary for the latent keys used in model input.
#     - batch_col_categories: List or array of categories for the batch column (optional).
#     - bio_col_categories: List or array of categories for the biological column (optional).
#     - return_metrics: Boolean flag indicating whether to return performance metrics (default: True).
#     - return_adata_dict: Boolean flag indicating whether to return the AnnData dictionary (default: False).
#     - return_trained_model: Boolean flag indicating whether to return the trained model (default: False).
#     - model_type: String specifying the type of model being used (default: "mec").
#     - issparse(bool): True if X is in sparse array, False if its dense
#     - load_dense (bool): If True, forces conversion of sparse arrays to dense format.
#     - seed (int) : seed set for repreducible results of dummy classifier with strategy: stratified

#     Returns:
#     - results: Dictionary containing the results based on the specified flags. Possible keys include:
#         - "dffn_model": The trained deep feedforward network model (if return_trained_model is True).
#         - "metrics": DataFrame of performance metrics for the trained model and SVM.
#         - "adata": The AnnData dictionary containing processed data (if return_adata_dict is True). Default: None
#     """

#     # 1. Load data latent paths and adata_dict
#     adata_dict = load_latent_spaces(base_path, fold, models_list, latent_path_dict, model_params, batch_col, bio_col, batch_col_categories, bio_col_categories,issparse, load_dense)

#     print("Batches available: ", np.unique(adata_dict["train"].obs[batch_col]))

#     # 2. Prepare data for training
#     inputs = prepare_latent_space_inputs(adata_dict, latent_keys_config, eval_test=model_params.eval_test)

#     # 3. Build and train model, plott loss and evaluate dffn model
#     dffn_results = build_train_evaluate_model(Model,build_model_dict, compile_dict, inputs, adata_dict, model_params, save_model, model_type)

#     adata_dict = dffn_results["adata_dict"]
#     dffn_metrics = dffn_results["metrics"]

#     svm_results = svm_accuracy_and_predictions(inputs, adata_dict, model_params,eval_test=model_params.eval_test)
#     adata_dict = svm_results["adata_dict"]
#     svm_metrics = svm_results["metrics"]
#     #metrics_df = pd.merge(dffn_metrics,svm_metrics)


#     # Evaluate using RandomForest
#     rf_results = random_forest_accuracy_and_predictions(inputs, adata_dict, model_params, eval_test=model_params.eval_test)
#     adata_dict = rf_results["adata_dict"]
#     rf_metrics = rf_results["metrics"]

#     # Calculate chance accuracy
#     chance_results = dummy_classifier_chance_accuracy(inputs, adata_dict, model_params, eval_test=model_params.eval_test,seed = seed)
#     adata_dict = chance_results["adata_dict"]
#     chance_metrics = chance_results["metrics"]

#     # Merge the metrics from DFFN, SVM, and RandomForest
#     metrics_df = pd.merge(dffn_metrics, svm_metrics, on="Dataset")
#     metrics_df = pd.merge(metrics_df, rf_metrics, on="Dataset")
#     metrics_df = pd.merge(metrics_df, chance_metrics, on="Dataset")

#     metrics_df.to_csv(os.path.join(model_params.latent_path, "metrics.csv"))

    


#     # 7. Collect results based on flags
#     results = {}
#     if return_trained_model:
#         results["dffn_model"] = dffn_results["model"]
#     if return_metrics:
#         results["metrics"] = metrics_df
#     if return_adata_dict:
#         results["adata"] = adata_dict

#     return results



def run_model_pipeline_LatentClassifier_v2_PCA(Model, latent_path_dict, build_model_dict, compile_dict, model_params, save_model, 
                                           batch_col, bio_col, base_path, fold, models_list, latent_keys_config,
                                           batch_col_categories=None, bio_col_categories=None, return_metrics=True, 
                                           return_adata_dict=False, return_trained_model=False, model_type="mec",
                                           issparse=False, load_dense=False,seed=None, get_pca:bool=True, get_svc:bool=False, **kwargs):
    """
    Runs the complete model pipeline, including data loading, model training, evaluation, and metric collection.

    Parameters:
    - Model: The model class to be instantiated and trained.
    - latent_path_dict: Dictionary containing paths to latent space data for each model.
    - build_model_dict: Dictionary of parameters for building the model.
    - compile_dict: Dictionary of parameters for compiling the model.
    - model_params: Object containing additional model parameters and configurations.
    - save_model: Boolean flag indicating whether to save the trained model to disk.
    - batch_col: Name of the column representing batch information in the data.
    - bio_col: Name of the column representing biological information in the data.
    - base_path: Base directory path for datasets and model output.
    - fold: The specific fold identifier for cross-validation or data splitting.
    - models_list: List of models to be used in the pipeline.
    - latent_keys_config: Configuration dictionary for the latent keys used in model input.
    - batch_col_categories: List or array of categories for the batch column (optional).
    - bio_col_categories: List or array of categories for the biological column (optional).
    - return_metrics: Boolean flag indicating whether to return performance metrics (default: True).
    - return_adata_dict: Boolean flag indicating whether to return the AnnData dictionary (default: False).
    - return_trained_model: Boolean flag indicating whether to return the trained model (default: False).
    - model_type: String specifying the type of model being used (default: "mec").
    - issparse(bool): True if X is in sparse array, False if its dense
    - load_dense (bool): If True, forces conversion of sparse arrays to dense format.
    - seed (int) : seed set for repreducible results of dummy classifier with strategy: stratified. Default: None

    Returns:
    - results: Dictionary containing the results based on the specified flags. Possible keys include:
        - "dffn_model": The trained deep feedforward network model (if return_trained_model is True).
        - "metrics": DataFrame of performance metrics for the trained model and SVM.
        - "adata": The AnnData dictionary containing processed data (if return_adata_dict is True).
    """

    # 1. Load data latent paths and adata_dict
    adata_dict = load_latent_spaces(base_path, fold, models_list, latent_path_dict, model_params, batch_col, bio_col, batch_col_categories, bio_col_categories,issparse, load_dense)
    
    # Calculate PCA
    if get_pca:
        adata_dict = get_pca_andplot(adata_dict, plot_params=None, eval_test=model_params.eval_test,n_components=model_params.n_components,shape_color_dict = None)


    print("Batches available: ", np.unique(adata_dict["train"].obs[batch_col]))

    # 2. Prepare data for training
    inputs = prepare_latent_space_inputs(adata_dict, latent_keys_config, eval_test=model_params.eval_test)
    print(f"inputs: {inputs}")
    print(f"model params {model_params}")

    # 3. Build and train model, plott loss and evaluate dffn model
    dffn_results = build_train_evaluate_model(Model, build_model_dict, compile_dict, inputs, adata_dict, model_params, save_model, model_type)

    adata_dict = dffn_results["adata_dict"]
    dffn_metrics = dffn_results["metrics"]
    metrics_df = None
    if get_svc:
        print("Training svc classifier")
        svm_results = svm_accuracy_and_predictions(inputs, adata_dict, model_params,eval_test=model_params.eval_test)
        adata_dict = svm_results["adata_dict"]
        svm_metrics = svm_results["metrics"]
        #metrics_df = pd.merge(dffn_metrics,svm_metrics)
         # Merge the metrics from DFFN, SVM, and RandomForest
        metrics_df = pd.merge(dffn_metrics, svm_metrics, on="Dataset")


    # Evaluate using RandomForest
    print("Training random forest classifier")

    rf_results = random_forest_accuracy_and_predictions(inputs, adata_dict, model_params, eval_test=model_params.eval_test)
    adata_dict = rf_results["adata_dict"]
    rf_metrics = rf_results["metrics"]

    # Calculate chance accuracy
    chance_results = dummy_classifier_chance_accuracy(inputs, adata_dict, model_params, eval_test=model_params.eval_test,seed=seed)
    adata_dict = chance_results["adata_dict"]
    chance_metrics = chance_results["metrics"]

    if metrics_df is None:
        metrics_df = dffn_metrics
    metrics_df = pd.merge(metrics_df, rf_metrics, on="Dataset")
    metrics_df = pd.merge(metrics_df, chance_metrics, on="Dataset")

    metrics_df.to_csv(os.path.join(model_params.latent_path, "metrics.csv"))

    


    # 7. Collect results based on flags
    results = {}
    if return_trained_model:
        results["dffn_model"] = dffn_results["model"]
    if return_metrics:
        results["metrics"] = metrics_df
    if return_adata_dict:
        results["adata"] = adata_dict

    return results


def calculate_metrics_with_ci(df, group_col="Dataset"):
    """
    Calculate the mean and 95% confidence intervals for numeric columns in a DataFrame, grouped by a specified column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - group_col (str): The column to group by before calculating metrics. Default is "Dataset".

    Returns:
    - pd.DataFrame: A DataFrame containing the mean, lower 95% CI, and upper 95% CI for each numeric column.
    """
    from scipy import stats
    
    def calculate_95ci(series):
        """
        Calculate the 95% confidence interval for a given pandas Series.

        Parameters:
        - series (pd.Series): The data series to calculate the confidence interval for.

        Returns:
        - tuple: A tuple containing the lower and upper bounds of the 95% confidence interval.
        """
        n = series.count()
        mean = series.mean()
        sem = series.sem()  # Standard error of the mean
        margin_of_error = sem * stats.t.ppf((1 + 0.95) / 2., n-1)  # t-distribution critical value for 95% CI
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        return lower_bound, upper_bound

    # Select only numeric columns for the operations
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Calculate the mean for each numeric column, grouped by the specified column
    mean_df = df.groupby(group_col)[numeric_columns].mean()

    # Rename the columns in mean_df to indicate they are means
    mean_df.columns = [f"{col}_mean" for col in mean_df.columns]

    # Apply the function to each numeric column in the DataFrame, grouped by the specified column
    ci_df = df.groupby(group_col)[numeric_columns].agg(calculate_95ci)

    # Separate the CI into lower and upper bounds
    ci_lower_df = ci_df.applymap(lambda x: x[0])
    ci_upper_df = ci_df.applymap(lambda x: x[1])

    # Rename the columns to indicate they are lower and upper bounds
    ci_lower_df.columns = [f"{col}_95CI_lower" for col in ci_lower_df.columns]
    ci_upper_df.columns = [f"{col}_95CI_upper" for col in ci_upper_df.columns]

    # Combine the mean, lower CI, and upper CI into one DataFrame
    results_df = pd.concat([mean_df, ci_lower_df, ci_upper_df], axis=1)

    return results_df
