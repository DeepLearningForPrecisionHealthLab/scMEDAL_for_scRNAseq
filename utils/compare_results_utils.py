
import os
import re
from utils import get_split_paths,read_adata,min_max_scaling,plot_rep
import pandas as pd
from utils_load_model import get_latest_checkpoint
# Glob does not work well with strange characters


# def glob_like(directory, pattern):
#     """Mimic glob.glob() using os and re modules."""
#     # Compile a regular expression from the glob-like pattern
#     # Convert glob pattern (simple cases) to regex
#     regex = re.compile(re.escape(pattern).replace(r'\*', '.*').replace(r'\?', '.'))
#     # List all files in the directory
#     files = os.listdir(directory)
#     # Filter files based on the regex pattern
#     matched_files = [file for file in files if regex.match(file)]
#     # Create full paths
#     return [os.path.join(directory, file) for file in matched_files]


def glob_like(directory, pattern):
    """Mimic glob.glob() using os and re modules."""
    regex_pattern = re.compile(pattern)
    files = os.listdir(directory)
    matched_files = []
    for file in files:
        if regex_pattern.search(file):
            matched_files.append(file)
    return [os.path.join(directory, file) for file in matched_files]



def get_recon_paths_df(results_path_dict,get_batch_recon_paths = False,k_folds = 5):
# Initialize an empty list to store data
    data = []

    # Patterns to search for with corresponding type
    patterns = [('recon_train*', 'train'), ('recon_val*', 'val')]
    if get_batch_recon_paths:
        patterns += [('recon_batch_train*', 'train'), ('recon_batch_val*', 'val')]

    # Iterate over all the keys and splits
    for key, path in results_path_dict.items():
        for split in range(1, k_folds+1):
            # Directory to search in
            directory = f"{path}/splits_{split}"
            if os.path.exists(directory):
                for pattern, data_type in patterns:
                    # Use glob_like to find files matching the pattern
                    files = glob_like(directory, pattern)
                    
                    # Append the results to the data list
                    for file in files:
                        data.append({'Key': key, 'Split': split, 'ReconPath': file, 'Type': data_type})

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)
    return df

def get_latent_paths_df(results_path_dict,get_batch_recon_paths = False,k_folds = 5):
# Initialize an empty list to store data
    data = []

    # Patterns to search for with corresponding type
    patterns = [('latent*train*', 'train'), ('latent*val*', 'val')]


    # Iterate over all the keys and splits
    for key, path in results_path_dict.items():
        for split in range(1, k_folds+1):
            # Directory to search in
            directory = f"{path}/splits_{split}"
            if os.path.exists(directory):
                for pattern, data_type in patterns:
                    # Use glob_like to find files matching the pattern
                    files = glob_like(directory, pattern)
                    
                    # Append the results to the data list
                    for file in files:
                        data.append({'Key': key, 'Split': split, 'LatentPath': file, 'Type': data_type})

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)
    return df

import os
import re
import pandas as pd

def get_latent_paths_df(results_path_dict, k_folds=5):
    data = []

    # Simplified patterns to search for
    patterns = [('latent.*train', 'train'), ('latent.*val', 'val')]

    for key, path in results_path_dict.items():
        for split in range(1, k_folds+1):
            directory = f"{path}/splits_{split}"
            #print(f"Checking directory: {directory}")
            if os.path.exists(directory):
                for pattern, data_type in patterns:
                    files = glob_like(directory, pattern)
                    #print(f"Pattern: {pattern}, Files: {files}")
                    
                    for file in files:
                        data.append({'Key': key, 'Split': split, 'LatentPath': file, 'Type': data_type})
            else:
                print(f"Directory does not exist: {directory}")

    df = pd.DataFrame(data)
    return df







def get_input_paths_df(input_base_path):
#    Initialize an empty list to store data
    data = []

    # Iterate over all folds
    for fold in range(1, 6):
        # Get input paths for the current fold
        input_path_dict = get_split_paths(base_path=input_base_path, fold=fold)
        # print(f"Folder split: {input_path_dict}")
        
        for data_type, directory in input_path_dict.items():
            if data_type != 'test' and os.path.exists(directory):

                #for file in files:
                data.append({'Split': fold, 'Type': data_type, 'InputsPath': directory})

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)


    return df




def get_model_paths_df(results_path_dict):
    """
    Creates a DataFrame containing the paths to the latest checkpoint and model parameters for each key and split.

    Args:
        results_path_dict (dict): A dictionary where keys are identifiers and values are base directories to search for checkpoints.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'Key', 'Split', 'last_checkpoint', and 'model_params' containing the relevant paths.
    
    Notes:
        The function searches for checkpoint files and model parameter files in subdirectories named 'splits_<split>'
        under each path provided in results_path_dict. It considers up to 5 splits.
    """
    # Initialize an empty list to store data
    data = []

    # Iterate over all the keys and splits
    for key, path in results_path_dict.items():
        for split in range(1, 6):
            # Directory to search in
            directory = f"{path}/splits_{split}"
            if os.path.exists(directory):

                file = get_latest_checkpoint(directory)
                
                model_params_file = os.path.join(directory, 'model_params.yaml')
                config_file = os.path.join(directory, 'model_config.py')

                data.append({'Key': key, 'Split': split, 'last_checkpoint': file,'model_params':model_params_file,'config':config_file})

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)
    return df


import numpy as np
import anndata
import pandas as pd

def get_hvg(adata, num_genes_to_retain=None):
    """
    Computes the variance of each gene, stores the variances in the AnnData object as a DataFrame,
    and flags the top high variable genes with their rank, while preserving the original gene order.

    Parameters:
    adata (anndata.AnnData): The input AnnData object where rows are cells and columns are genes.
    num_genes_to_retain (int, optional): The number of top high variable genes to retain and flag. If None, no flagging is done.

    Returns:
    pd.DataFrame: DataFrame with gene names, variances, rank, and a flag indicating if they are in the top high variable genes.
    """
    # Compute the variance of each gene (column) across all cells (rows)
    gene_variances = np.var(adata.X, axis=0)

    # Create a DataFrame with gene names and their variances
    variances_df = pd.DataFrame({
        'gene': adata.var_names,
        'variance': gene_variances
    })

    # Sort the DataFrame by variance in descending order to determine ranks
    variances_df_sorted = variances_df.sort_values(by='variance', ascending=False).reset_index(drop=True)

    # Add rank column based on sorted variances
    variances_df_sorted['rank'] = np.arange(1, len(variances_df_sorted) + 1)

    # Merge back to original DataFrame to preserve original gene order
    variances_df = variances_df.merge(variances_df_sorted[['gene', 'rank']], on='gene', how='left')

    if num_genes_to_retain is not None:
        # Flag the top high variable genes
        variances_df['high_variable'] = variances_df['rank'] <= num_genes_to_retain
    else:
        # Add the column without flagging any genes
        variances_df['high_variable'] = False

    # Store the DataFrame in the AnnData object
    adata.uns['gene_variances'] = variances_df

    return adata


# Define some functions to preprocess df and get the same format for all of them
# functions to homogenize the results

def preprocess_results_model_pca_format(file_path, columns_to_drop):
    """
    Load data from a CSV file, clean and transpose the DataFrame. This is for a experiment results format where get_pca=True

    Parameters:
    - file_path: str, the path to the CSV file.
    - columns_to_drop: list of str, the names of columns to drop from level 1 of the MultiIndex.


    Returns:
    - pd.DataFrame, the transposed and cleaned DataFrame.
    """
    print("reading file:",file_path)
    # Load the CSV file into a DataFrame with specified MultiIndex for rows and columns
    df = pd.read_csv(file_path, header=[0, 1, 2], index_col=[0])

    # Drop specified columns from level 1 of the MultiIndex in the DataFrame
    for column in columns_to_drop:
        df = df.drop(labels=column, axis=1, level=1)

    # Remove 'sem_' prefix from levels 1 and 2 of the column names
    new_columns = [(level0, level1.replace('sem_', ''), level2.replace('sem_', '')) 
                   for level0, level1, level2 in df.columns]
    df.columns = pd.MultiIndex.from_tuples(new_columns)

    # Transpose the DataFrame and adjust the MultiIndex accordingly
    df_transposed = df.T
    new_index = [(level1, level2, level0) for level0, level1, level2 in df_transposed.index]
    df_transposed.index = pd.MultiIndex.from_tuples(new_index)

    return df_transposed



def process_single_model_format(file_path,model_name):
    """
    Process the dataframe to:
    1. Remove columns where the category level is 'fold'.
    2. Convert the dataframe back and forth to a dictionary to reformat into multi-level columns.
    """
    # Assuming 'fold' is at a specific level, let's say level 1 for category in a MultiIndex column
    #if 'fold' in [col[1] for col in df.columns]:  # Check if 'fold' exists in the second position of any column tuple
    #df = df.drop(columns=[col for col in df.columns if col[1] == 'fold'], level=1)
    
    # Convert DataFrame to dictionary and back to DataFrame with multi-level columns
    print("reading file:",file_path)

    df = pd.read_csv(file_path, header=[0],index_col=[0,1]).T
    #df = df1.copy()
    del df['fold']
    data = df.to_dict()
    
    # Create a DataFrame with multi-level columns
    columns = pd.MultiIndex.from_tuples([(key[0], key[1], stat) for key in data for stat in data[key]])#, names=['Category', 'Metric', 'Statistic'])
    values = [data[key][stat] for key in data for stat in data[key]]

    
    # Recreate DataFrame with multi-level columns
    new_df = pd.DataFrame([values], columns=columns)

    new_df["dataset_type"]=model_name
    new_df.set_index("dataset_type",inplace=True) 
    return new_df

def calculate_and_append_ci(df, dof):
    import scipy.stats as stats
    """
        Calculates 95% confidence intervals and adds it to df
        Args:
            df with clustering scores and 3 level columns: 1st level: batch,celltype, second level: 1/db, ch, silhouette, third level: mean, sem, std
            dof(int): degrees of freedom for confidence intervals
        Return:
            df
    """
    # Function to calculate the 95% confidence interval
    def get_CI(df, dof, mean_col, sem_col):
        # Critical t-value for 95% confidence interval
        t_critical = stats.t.ppf(1 - 0.025, dof)

        # Calculate the margin of error
        df['margin_of_error'] = df[sem_col] * t_critical

        # Calculate the confidence intervals
        df[(mean_col[0], mean_col[1], 'CI_lower')] = df[mean_col] - df['margin_of_error']
        df[(mean_col[0], mean_col[1], 'CI_upper')] = df[mean_col] + df['margin_of_error']
        return df[[(mean_col[0], mean_col[1], 'CI_lower'), (mean_col[0], mean_col[1], 'CI_upper')]]

    # Prepare to store CI columns
    ci_columns = []

    # Iterate through each level of the MultiIndex to calculate confidence intervals
    for level_0 in df.columns.levels[0]:
        for level_1 in df.columns.levels[1]:
            try:
                # Filter the DataFrame for the given index level combination
                mean_col = (level_0, level_1, 'mean')
                sem_col = (level_0, level_1, 'sem')

                if mean_col in df.columns and sem_col in df.columns:
                    df_subset = df[[mean_col, sem_col]]
                    ci_df = get_CI(df_subset.copy(), dof, mean_col, sem_col)
                    ci_columns.append(ci_df)
            except KeyError:
                # Skip if the combination of levels does not exist in the data
                continue

    # Concatenate the original DataFrame with the new CI columns
    ci_df = pd.concat(ci_columns, axis=1)
    final_df = pd.concat([df, ci_df], axis=1)

    return final_df


def get_umap_plot(df, umap_path, plot_params,sample_size=10000,n_neighbors=15):
    """
    Read adata and plot umap
    """
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    import os
    import scanpy as sc
    import gc

    # Get input paths
    print("\n Computing umap for the following files")
    input_prefixes = np.unique(df["input_prefix"])
    for input_prefix in input_prefixes:
        # Its the same input for the same split, same model
        input_path = df.loc[df["input_prefix"]==input_prefix,"InputsPath"].values[0]
        print(input_prefix,input_path)
   
        # Get umaps for input paths
        # Read input
        X, var, obs = read_adata(input_path, issparse=True)
        X = X.toarray()
        # Input needs to be scaled because reconstructions are already scaled
        X = min_max_scaling(X)
        
        # Subset obs to a random sample size (e.g., 1000)
        sampled_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X = X[sampled_indices,:]
        obs = obs.iloc[sampled_indices]
        
        # Compute umap
        adata_input = AnnData(X, obs=obs, var=var)
        sc.pp.neighbors(adata_input, n_neighbors=n_neighbors)
        sc.tl.umap(adata_input)
        plot_rep(adata_input, outpath=umap_path, file_name=f"input_{input_prefix}_{sample_size}cells", **plot_params)
        np.save(os.path.join(umap_path, f"umap_input_{input_prefix}_{sample_size}cells.npy"),adata_input.obsm["X_umap"])

        latent_prefixes = df.loc[df["input_prefix"]==input_prefix,"latent_prefix"].values
        latent_paths = df.loc[df["input_prefix"]==input_prefix,"LatentPath"].values

        for latent_pref, latent_path in zip(latent_prefixes, latent_paths):
            print(latent_pref,latent_paths)

            X = np.load(latent_path)
            X = X[sampled_indices,:]  # Subset to the same indices as above
            adata = AnnData(X, obs=obs)

            # Calculate UMAP 
            sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_neighbors)
            sc.tl.umap(adata)
            plot_rep(adata, outpath=umap_path, file_name=f"{latent_pref}_{sample_size}cells", **plot_params)
            np.save(os.path.join(umap_path, f"umap_{latent_pref}_{sample_size}cells.npy"),adata.obsm["X_umap"])
            gc.collect()