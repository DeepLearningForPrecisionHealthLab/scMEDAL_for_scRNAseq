
import os
import re
from .utils import get_split_paths,read_adata,min_max_scaling,plot_rep,calculate_zscores
import pandas as pd
from .utils_load_model import get_latest_checkpoint
from anndata import AnnData
import scanpy as sc
import gc
from typing import Optional, Tuple, List
import os
import re
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Glob does not work well with strange characters

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
    patterns = [('recon_train*', 'train'), ('recon_val*', 'val'),('recon_test*', 'test')]
    if get_batch_recon_paths:
        patterns += [('recon_batch_train*', 'train'), ('recon_batch_val*', 'val'),('recon_batch_test*', 'test')]

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




def get_latent_paths_df(results_path_dict, k_folds=5):
    data = []
    print(results_path_dict)

    # Simplified patterns to search for
    patterns = [('latent.*train', 'train'), ('latent.*val', 'val'),('latent.*test', 'test')]

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


def create_latent_dict_from_df(df_latent,model_col="Key",fold_col="Split", dataset_col = "Type",path_col="LatentPath"):
    """
    Create a nested dictionary from a DataFrame to structure latent paths.

    Parameters:
    -----------
    df_latent : pd.DataFrame
        A DataFrame containing columns that specify the model, split (or fold), dataset type, 
        and the corresponding latent path.

    model_col : str, optional (default="Key")
        The column name in df_latent representing the model identifiers.

    fold_col : str, optional (default="Split")
        The column name in df_latent representing the split or fold identifiers.

    dataset_col : str, optional (default="Type")
        The column name in df_latent representing the dataset type (e.g., 'train', 'test').

    path_col : str, optional (default="LatentPath")
        The column name in df_latent representing the paths or values to be structured in the dictionary.

    Returns:
    --------
    dict
        A dictionary where the keys are model names, and the values are nested dictionaries structured as:
        latent_dict[model]["splits_" + str(fold)][dataset] = path or value.

    Example:
    --------
    >>> df_latent = pd.DataFrame({
            'Key': ['model1', 'model1', 'model2'],
            'Split': [1, 2, 1],
            'Type': ['train', 'test', 'train'],
            'LatentPath': ['/path/to/model1_fold1_train', '/path/to/model1_fold2_test', '/path/to/model2_fold1_train']
        })
    >>> latent_dict = create_latent_dict_from_df(df_latent)
    >>> print(latent_dict)
    {
        'model1': {
            'splits_1': {'train': '/path/to/model1_fold1_train'},
            'splits_2': {'test': '/path/to/model1_fold2_test'}
        },
        'model2': {
            'splits_1': {'train': '/path/to/model2_fold1_train'}
        }
    }
    """
    latent_dict = {}

    for _, row in df_latent.iterrows():
        model = row[model_col]
        fold = row[fold_col]
        dataset = row[dataset_col]
        value = row[path_col]  # Assuming there is a column 'value' in df_latent

        # Initialize nested dictionaries if not already present
        if model not in latent_dict:
            latent_dict[model] = {}
        if "splits_" + str(fold) not in latent_dict[model]:
            latent_dict[model]["splits_" + str(fold)] = {}

        # Assign the value to the correct place in the dictionary
        latent_dict[model]["splits_" + str(fold)][dataset] = value

    return latent_dict



def get_input_paths_df(input_base_path,k_folds = 5,eval_test=False):
    """
    Generate a DataFrame containing the input paths for each fold in a k-fold cross-validation setup.

    Parameters:
    -----------
    input_base_path : str
        The base directory path where the input data is stored.

    k_folds : int, optional
        The number of folds in the k-fold cross-validation (default is 5).

    eval_test : bool, optional
        If True, includes all data types (including 'test') in the DataFrame. If False, excludes 'test' data type from the DataFrame (default is False).
        When training the model you don't use test. This is why the default is False.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns 'Split', 'Type', and 'InputsPath', where:
            - 'Split' indicates the fold number.
            - 'Type' indicates the type of data (e.g., 'train', 'validation', 'test').
            - 'InputsPath' contains the directory path for the corresponding fold and data type.

    Example:
    --------
    >>> df = get_input_paths_df("/path/to/data", k_folds=5, eval_test=True)
    >>> print(df.head())
       Split       Type                    InputsPath
    0      1      train  /path/to/data/fold_1/train
    1      1  validation  /path/to/data/fold_1/validation
    2      1       test  /path/to/data/fold_1/test
    3      2      train  /path/to/data/fold_2/train
    4      2  validation  /path/to/data/fold_2/validation

    Notes:
    ------
    - The function assumes that the directory structure for each fold and data type is already created and accessible.
    - If `eval_test` is set to False, the 'test' data type will be excluded from the resulting DataFrame.
    """
    #   Initialize an empty list to store data
    data = []

    # Iterate over all folds
    for fold in range(1, k_folds+1):
        # Get input paths for the current fold
        input_path_dict = get_split_paths(base_path=input_base_path, fold=fold)
        # print(f"Folder split: {input_path_dict}")
        
        for data_type, directory in input_path_dict.items():

            # if data_type != 'test' and os.path.exists(directory):

            # If eval_test is True, append all data types
            # If eval_test is False, append only non-'test' data types
            if eval_test or (data_type != 'test' and os.path.exists(directory)):
                data.append({'Split': fold, 'Type': data_type, 'InputsPath': directory})

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)


    return df


def get_model_paths_df(results_path_dict,k_folds = 5):
    """
    Creates a DataFrame containing the paths to the latest checkpoint and model parameters for each key and split.

    Args:
        results_path_dict (dict): A dictionary where keys are identifiers and values are base directories to search for checkpoints.
        kfolds(int): number of k folds. This is equal to number of saved models.

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
        for split in range(1, k_folds+1):
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

def filter_models_by_type_and_split(df, models_dict, Type='train'):
    """
    Filters the DataFrame for the specified models, type 'train', and given splits.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the following columns:
        - 'Key': The name of the model.
        - 'Type': The type of data (e.g., 'train', 'val', 'test').
        - 'Split': The split identifier.
    models_dict (dict): Dictionary specifying models and the corresponding splits to filter.
        The keys are the model names (values in the 'Key' column), and the values are the split numbers (values in the 'Split' column).
    Type (str): The type of data to filter on (default is 'train').

    Returns:
    pd.DataFrame: The filtered DataFrame containing only the rows that match the specified model names,
                  have 'Type' equal to 'train', and match the corresponding split number.
    """
    filtered_df = pd.DataFrame()

    for model, split in models_dict.items():
        temp_df = df[(df['Key'] == model) & (df['Type'] == Type) & (df['Split'] == split)]
        filtered_df = pd.concat([filtered_df, temp_df], ignore_index=True)

    return filtered_df



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


def aggregate_paths(results_path_dict, pattern = f'mean_scores_test_samplesize'):
    # Initialize a dictionary to store paths
    paths_dict = {}

    # Populate paths_dict with model names and corresponding file paths
    for model_name, path in results_path_dict.items():
        print(model_name, path)
        paths_dict[model_name] = glob_like(path, pattern)

    # Initialize an empty DataFrame to store all the data
    df_all_paths = pd.DataFrame()

    # Iterate over the dictionary items
    for model_name, files in paths_dict.items():
        # Extract the sample size from each file path
        sample_size = [file.split("samplesize-")[-1].split(".csv")[0] for file in files]
        # Create a DataFrame for the current batch of files
        df = pd.DataFrame({"sample_size": sample_size, "path": files})
        df['model_name'] = model_name  # Add a column for the model name
        # Append the current DataFrame to the aggregated DataFrame
        df_all_paths = pd.concat([df_all_paths, df])  # Ensure to reassign and use ignore_index=True

    # Now df_all_paths contains all the data combined from the different files and models
    return df_all_paths

def read_and_aggregate_scores(df_all_paths):
    """
    Reads CSV files from the paths listed in the provided DataFrame and aggregates the data into a single DataFrame.

    Parameters:
    df_all_paths (pd.DataFrame): DataFrame containing columns 'sample_size', 'path', and 'model_name'.

    Returns:
    pd.DataFrame: A DataFrame containing all the data from the CSV files specified in the 'path' column.
    """
    # Initialize an empty DataFrame to store all scores
    df_all_paths_allscores = pd.DataFrame()

    # Iterate over each row in the DataFrame
    for index, row in df_all_paths.iterrows():
        print(row)
        # Read the CSV file
        df = pd.read_csv(row['path'], header=[0, 1],index_col=0)
                
        # Rename 'Unnamed: 7_level_1' and 'Unnamed: 8_level_1' columns to empty strings
        df.rename(columns=lambda x: '' if 'Unnamed' in x else x, level=1, inplace=True)
        # Reset the index to avoid issues when appending
        df.reset_index(inplace=True)
        # Add additional information from the original DataFrame
        df['sample_size'] = row['sample_size']
        # df['model_name'] = row['model_name']
        # Append the current DataFrame to the aggregated DataFrame
        df_all_paths_allscores = pd.concat([df_all_paths_allscores, df], ignore_index=True)
        df_all_paths_allscores["latent_name"] = [s.split("_latent_")[0] for s in df_all_paths_allscores["dataset_type"]]


    return df_all_paths_allscores




def filter_min_max_silhouette_scores(df,batch_col='batch'):
    """
    Filters rows with the minimum and maximum silhouette scores for each dataset type,
    considering the column MultiIndex structure.

    Parameters:
    df (pd.DataFrame): DataFrame containing all scores with columns including 'silhouette' under 'batch' and 'celltype' levels,
                       as well as other relevant columns.
    batch_col (str): String with name of the columns that containes the batch scores. Default:'batch'

    Returns:
    tuple: A tuple containing two DataFrames: one for minimum silhouette scores and one for maximum silhouette scores.
    """
    # Initialize DataFrames for storing the min and max silhouette scores
    df_min_silhouette = pd.DataFrame(columns=df.columns)
    df_max_silhouette = pd.DataFrame(columns=df.columns)

    # Iterate over each unique dataset type
    for dataset_type in df[('dataset_type', '')].unique():
        # Filter the DataFrame for the current dataset type
        df_filtered = df[df[('dataset_type', '')] == dataset_type]

        if not df_filtered.empty:
            # Find the row with the minimum silhouette score
            min_row = df_filtered.loc[df_filtered[(batch_col, 'silhouette')].idxmin()].reset_index(drop=True)
            df_min_silhouette = pd.concat([df_min_silhouette, min_row], ignore_index=True)
            
            # Find the row with the maximum silhouette score
            max_row = df_filtered.loc[df_filtered[(batch_col, 'silhouette')].idxmax()].reset_index(drop=True)
            df_max_silhouette = pd.concat([df_max_silhouette, max_row], ignore_index=True)

    return df_min_silhouette, df_max_silhouette


def process_all_results(df_all_paths, models2process_dict, out_name, dataset_type):
    """
    Process and merge results from different models and sample sizes into a single DataFrame, and save to CSV.

    Parameters:
    df_all_paths (pd.DataFrame): DataFrame containing columns 'sample_size', 'model_name', and 'path'.
    models2process_dict (dict): Dictionary indicating the processing type for each model.
    out_name (str): Directory path where the processed results will be saved.
    dataset_type (str): Type of dataset being processed.

    Returns:
    dict: Dictionary with sample sizes as keys and their corresponding processed DataFrames as values.
    """
    df_sample_size = {}
    print(df_all_paths)
    # Loop over each unique sample size
    for sample_size in np.unique(df_all_paths["sample_size"]):
        print("sample size", sample_size)
        model_path_dict = {}
        # DataFrames for single model format and PCA format
        df_smf = pd.DataFrame()
        df_pcaf = pd.DataFrame()
        count_i = 0

        for model_name in np.unique(df_all_paths["model_name"]):
            # Get the results file path
            print(model_name)
            file_paths = df_all_paths.loc[(df_all_paths["sample_size"] == sample_size) & (df_all_paths["model_name"] == model_name), "path"].values
            if len(file_paths) > 0:
                file_path = file_paths[0]
                print(model_name, "file path", file_path)
                model_path_dict[model_name] = file_path
            
            # Process models without PCA results
            if models2process_dict[model_name] == "process_single_model_format":
                df_smf = pd.concat([df_smf, process_single_model_format(file_path=model_path_dict[model_name], model_name=model_name) ])
                print("smf", df_smf)
            
            # Process models with PCA results
            elif models2process_dict[model_name] == "preprocess_results_model_pca_format":
                df_pcaf_i = preprocess_results_model_pca_format(model_path_dict[model_name], columns_to_drop=['fold', 'sem_fold'])
                df_pcaf_i = df_pcaf_i.rename(columns={"X_pca_val": "X_pca_val_" + str(count_i)})
                df_pcaf = pd.concat([df_pcaf, df_pcaf_i], axis=1)
                count_i += 1

            else:
                print(model_name, "not processed")
        
        # Merge PCA format and single model format DataFrames
        df_all = pd.concat([df_pcaf, df_smf.T], axis=1).T
        # Save the DataFrame to CSV
        df_all.to_csv(os.path.join(out_name, f"{dataset_type}_scores_{sample_size}.csv"))
        print(df_all)
        df_sample_size[sample_size] = df_all

    return df_sample_size



def process_confidence_intervals(df_all, out_name, dataset_type, sample_size, dof=4):
    """
    Calculate 95% CI for the DataFrame, intercalate mean and CI columns, and save to a CSV file.

    Parameters:
    df_all (pd.DataFrame): DataFrame containing all scores with columns including 'mean' and 'CI' values.
    out_name (str): Directory path where the processed results will be saved.
    dataset_type (str): Type of dataset being processed.
    sample_size (int): Sample size for the current processing batch.
    dof (int): Degrees of freedom for calculating the CI. Default is 4 (n=5, dof=n-1=4).

    Returns:
    pd.DataFrame: DataFrame with intercalated mean and CI columns.
    """
    # Calculate and append 95% CI
    df_ci = calculate_and_append_ci(df_all, dof=dof)

    # Extract only mean and CI columns
    mean_ci_columns = [col for col in df_ci.columns if 'mean' in col[2] or 'CI' in col[2]]
    df_mean_ci = df_ci[mean_ci_columns]

    # Intercalate the mean and CI columns
    intercalated_columns = []
    for level_0 in df_ci.columns.levels[0]:
        for level_1 in df_ci.columns.levels[1]:
            mean_col = (level_0, level_1, 'mean')
            ci_lower_col = (level_0, level_1, 'CI_lower')
            ci_upper_col = (level_0, level_1, 'CI_upper')
            if mean_col in df_mean_ci.columns:
                intercalated_columns.append(mean_col)
            if ci_lower_col in df_mean_ci.columns:
                intercalated_columns.append(ci_lower_col)
            if ci_upper_col in df_mean_ci.columns:
                intercalated_columns.append(ci_upper_col)

    # Reorder the DataFrame columns based on intercalated columns
    df_mean_ci = df_mean_ci[intercalated_columns]

    # Save the DataFrame with 95% CI to a CSV file
    output_path = os.path.join(out_name, f"{dataset_type}_scores_{sample_size}_95CI.csv")
    df_mean_ci.to_csv(output_path)

    return df_mean_ci

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
    print(df.columns)
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








def filter_adata_by_batch(adata_input, n_batches_sample, batch_col="batch"):
    """
    Filter an AnnData object by a list of batch values.

    Parameters:
    adata_input (anndata.AnnData): The input AnnData object.
    n_batches_sample (list): A list of batch values to filter by.
    batch_col (str): The column name in obs to filter by.

    Returns:
    anndata.AnnData: A new AnnData object with filtered obs and X.
    """
    # Filter the obs DataFrame based on the specified batch values
    filtered_obs = adata_input.obs[adata_input.obs[batch_col].isin(n_batches_sample)]
    
    # Get the indices of the filtered rows
    filtered_indices = filtered_obs.index.astype(int)
    
    # Filter the X matrix using the filtered indices
    filtered_X = adata_input.X[filtered_indices, :]

    # Create a new AnnData object with the filtered data
    filtered_adata = AnnData(X=filtered_X, obs=filtered_obs, var=adata_input.var)
    return filtered_adata



def get_umap_plot(df, umap_path, plot_params, sample_size=10000, n_neighbors=15, scaling="min_max", n_batches_sample=20, batch_col="batch", issparse=True, seed:int=42):
    """
    Reads a DataFrame with input paths, loads input and latent spaces, applies scaling, computes UMAP, and plots the results.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing the input and latent paths along with prefixes.
    umap_path : str
        Path where UMAP plots and data will be saved.
    plot_params : dict
        Dictionary of parameters to customize the plot appearance.
    sample_size : int, optional
        Number of samples to subset for UMAP computation, default is 10,000.
    n_neighbors : int, optional
        Number of neighbors to use in UMAP computation, default is 15.
    scaling : str, optional
        Type of scaling to apply to the data, either "min_max" or "z_scores", default is "min_max".
    n_batches_sample : int, optional
        Number of batches to sample for plotting, default is 20.
    batch_col : str, optional
        Column name in obs to filter by, default is "batch".

    issparse: bool, optional
        Flag if the inputs are sparse to load tehm and convert them to numpy array.

    Returns:
    None

    The function performs the following steps:
    1. Iterates through unique input prefixes in the DataFrame.
    2. Reads input data and applies scaling based on the provided scaling parameter.
    3. Subsets the data to a specified sample size.
    4. Computes UMAP for the input data and saves the results.
    5. Iterates through latent data paths corresponding to each input prefix.
    6. Computes UMAP for each latent data and saves the results.
    7. Plots and saves the UMAP representations.

    Notes:
    - The `read_adata`, `min_max_scaling`, `calculate_zscores`, and `plot_rep` functions are defined in utils.
    - UMAP results are saved in the specified `umap_path` directory.
    """
    
    print("\nComputing UMAP for the following files:")
    
    # Get unique input prefixes
    input_prefixes = np.unique(df["input_prefix"])
    batches_sample = []
    # Set the random seed for reproducibility
    np.random.seed(seed)
    for i, input_prefix in enumerate(input_prefixes):
        # Get the path for the current input prefix
        input_path = df.loc[df["input_prefix"] == input_prefix, "InputsPath"].values[0]
        print(input_prefix, input_path)

        # Read input data
        
        X, var, obs = read_adata(input_path, issparse = issparse)
        if issparse:
            X = X.toarray()

        # Apply scaling to X based on the scaling parameter
        if scaling == "min_max":
            X = min_max_scaling(X)
        elif scaling == "z_scores":
            X = calculate_zscores(X)

        # Subset obs to a random sample size
        if sample_size is not None:
            sampled_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X = X[sampled_indices, :]
            obs = obs.iloc[sampled_indices].reset_index(drop=True)

        # Compute UMAP for the input data
        adata_input = AnnData(X, obs=obs, var=var)
        sc.pp.neighbors(adata_input, n_neighbors=n_neighbors)
        sc.tl.umap(adata_input)
        
        # Save UMAP results to CSV
        umap_df = pd.DataFrame(adata_input.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
        umap_df = pd.concat([umap_df, obs], axis=1)
        umap_df.to_csv(os.path.join(umap_path, f"umap_input_{input_prefix}_{sample_size}cells.csv"), index=False)
        
        # Plot UMAP results for input data
        plot_rep(adata_input, outpath=umap_path, file_name=f"input_{input_prefix}_{sample_size}cells", **plot_params)

        # Plot batch subsample
        # If first iteration, determine batches to sample
        if i == 0 and n_batches_sample is not None:
            batches = np.unique(obs[batch_col])
            batches = batches.tolist()
            batches_sample = np.random.choice(batches, size=n_batches_sample, replace=False)
            print("batches sample for plotting",batches_sample)
        # Modify plot parameters to plot by batch
            plot_params_batch_col = plot_params.copy()
            plot_params_batch_col.update({
                "shape_col": batch_col,
                "color_col": batch_col
            })

        if n_batches_sample is not None:
            # Filter data by batches and calculate UMAP
            adata_input = filter_adata_by_batch(adata_input, batches_sample, batch_col=batch_col)
            sc.pp.neighbors(adata_input, use_rep="X", n_neighbors=n_neighbors)
            sc.tl.umap(adata_input)
                
            # Save UMAP results for batch-filtered data to CSV
            umap_df = pd.DataFrame(adata_input.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
            umap_df = pd.concat([umap_df, adata_input.obs.reset_index(drop=True)], axis=1)
            umap_df.to_csv(os.path.join(umap_path, f"umap_input_{sample_size}cells_{n_batches_sample}{batch_col}.csv"), index=False)
                
            # Plot UMAP results for batch-filtered data
            plot_rep(adata_input, outpath=umap_path, file_name=f"input_{input_prefix}_{sample_size}cells_{n_batches_sample}{batch_col}_biocolors", **plot_params)
            plot_rep(adata_input, outpath=umap_path, file_name=f"input_{input_prefix}_{sample_size}cells_{n_batches_sample}{batch_col}_batchcolors", **plot_params_batch_col)


        # Get latent prefixes and paths for the current input prefix
        latent_prefixes = df.loc[df["input_prefix"] == input_prefix, "latent_prefix"].values
        latent_paths = df.loc[df["input_prefix"] == input_prefix, "LatentPath"].values


        for latent_pref, latent_path in zip(latent_prefixes, latent_paths):
            print(latent_pref, latent_path)

            # Load latent data
            X = np.load(latent_path)
            if sample_size is not None:
                X = X[sampled_indices, :]
            adata = AnnData(X, obs=obs)

            # Calculate UMAP for latent data
            sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_neighbors)
            sc.tl.umap(adata)
            
            # Save UMAP results to CSV
            umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
            umap_df = pd.concat([umap_df, obs], axis=1)
            umap_df.to_csv(os.path.join(umap_path, f"umap_{latent_pref}_{sample_size}cells.csv"), index=False)
            
            # Plot UMAP results for latent data
            plot_rep(adata, outpath=umap_path, file_name=f"{latent_pref}_{sample_size}cells", **plot_params)
            try:
                plot_rep(adata, outpath=umap_path, file_name=f"{latent_pref}_{sample_size}cells_{batch_col}", **plot_params_batch_col)
            except Exception as e:
                print("Check the error but probably there are too many batches")
                print(e)

            if n_batches_sample is not None:
                # Filter data by batches and calculate UMAP
                adata = filter_adata_by_batch(adata, batches_sample, batch_col=batch_col)
                sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_neighbors)
                sc.tl.umap(adata)
                
                # Save UMAP results for batch-filtered data to CSV
                umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
                umap_df = pd.concat([umap_df, adata.obs.reset_index(drop=True)], axis=1)
                umap_df.to_csv(os.path.join(umap_path, f"umap_{latent_pref}_{sample_size}cells_{n_batches_sample}{batch_col}.csv"), index=False)
                
                # Plot UMAP results for batch-filtered data
                plot_rep(adata, outpath=umap_path, file_name=f"{latent_pref}_{sample_size}cells_{n_batches_sample}{batch_col}_celltypecolors", **plot_params)
                plot_rep(adata, outpath=umap_path, file_name=f"{latent_pref}_{sample_size}cells_{n_batches_sample}{batch_col}_batchcolors", **plot_params_batch_col)

            gc.collect()
class DimensionalityReductionProcessor:
<<<<<<< HEAD:utils/compare_results_utils.py
=======
    """Compute UMAPs, save CSV/NPY, and plot latent + UMAP reps"""
    from typing import Optional, Tuple, List

    #  INIT 
>>>>>>> e362fe11d74fc7a997deee93612524376f027bf1:scMEDAL/utils/compare_results_utils.py
    def __init__(
        self,
        df: pd.DataFrame,
        output_path: str,
        plot_params: dict,
        sample_size: Optional[int] = None,
        n_neighbors: int = 15,
        scaling: str = "min_max",
        n_batches_sample: Optional[int] = 20,
        batch_col: str = "batch",
        plot_tsne: bool = False,        # placeholder for future use
        n_pca_components: int = 50,
        min_dist: float = 0.5,
        rng_seed: int = 5,
        extra_color_cols: Optional[List[str]] = None,
    ):
        self.df = df
        self.output_path = output_path
        self.plot_params = plot_params
        self.sample_size = sample_size
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.scaling = scaling
        self.n_batches_sample = n_batches_sample
        self.batch_col = batch_col
        self.n_pca_components = n_pca_components
<<<<<<< HEAD:utils/compare_results_utils.py
        self.batches_sample = None
=======
>>>>>>> e362fe11d74fc7a997deee93612524376f027bf1:scMEDAL/utils/compare_results_utils.py
        self.extra_color_cols = extra_color_cols or []
        self.rng = np.random.RandomState(rng_seed)

        os.makedirs(self.output_path, exist_ok=True)
        self.batches_sample: Optional[np.ndarray] = None
        self.sampled_indices: Optional[np.ndarray] = None
        self._umap_written: set[str] = set()

<<<<<<< HEAD:utils/compare_results_utils.py
=======
    #  HELPERS 
>>>>>>> e362fe11d74fc7a997deee93612524376f027bf1:scMEDAL/utils/compare_results_utils.py
    def _scale(self, X):
        if self.scaling == "min_max":
            return min_max_scaling(X)
        if self.scaling == "z_scores":
            return calculate_zscores(X)
        return X

    def _subset(self, X, obs):
        if self.sample_size:
            idx = self.rng.choice(X.shape[0], self.sample_size, replace=False)
            self.sampled_indices = idx
            return X[idx, :], obs.iloc[idx].reset_index(drop=True)
        self.sampled_indices = np.arange(X.shape[0])
        return X, obs

    def _pca(self, X):
        return PCA(self.n_pca_components, random_state=self.rng).fit_transform(X)

    def _embed(self, adata, use_rep):
        sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=use_rep, random_state=self.rng)
        sc.tl.umap(adata, min_dist=self.min_dist, random_state=self.rng)

    #  central plot wrapper 
    def _plot_all(self, adata, stem, shape_col, color_col):
        """Generate regular & axis?free plots while avoiding duplicate kwargs."""
        pp = self.plot_params.copy()
        for k in ("shape_col", "color_col", "markers", "outpath", "file_name"):
            pp.pop(k, None)
        mk = self.plot_params.get("markers", None)

        for suf, axes, box in [("", True, True), ("_noaxis", False, False)]:
            plot_rep(
                adata,
                shape_col=shape_col,
                color_col=color_col,
                markers=mk,
                axes=axes,
                legend_box=box,
                outpath=self.output_path,
                file_name=f"{stem}{suf}",
                **pp
            )

    #   SAVE HELPERS 
    def _save_csv(self, adata, fname, rep_key):
        """Write embedding coords to CSV without mutating original obs.
<<<<<<< HEAD:utils/compare_results_utils.py
=======

>>>>>>> e362fe11d74fc7a997deee93612524376f027bf1:scMEDAL/utils/compare_results_utils.py
        * `X_umap` ? CSV only (UMAP¹/² columns not added to `obs`).
        * `X_pca`  ? CSV + full 50?PC `.npy` dump.
        * others   ? CSV only.
        Column names include the `fname` suffix, so collisions are impossible.
        """
        if rep_key not in adata.obsm:
            return

        arr = adata.obsm[rep_key]
        label = {"X_umap": "UMAP", "X_tsne": "TSNE", "X_pca": "PC"}.get(
            rep_key, rep_key.upper().lstrip("X_"))
        col1, col2 = f"{label}1_{fname}", f"{label}2_{fname}"
        coords_df = pd.DataFrame(arr[:, :2], columns=[col1, col2])
        df_out = coords_df.join(adata.obs.copy().reset_index(drop=True))
        df_out.to_csv(os.path.join(self.output_path, f"{rep_key}_{fname}.csv"), index=False)

        if rep_key == "X_pca":
            np.save(os.path.join(self.output_path, f"{rep_key}_{fname}.npy"), arr)

    def _save_umap_once(self, adata, fname):
        """Ensure each global UMAP embedding is saved exactly once."""
        if fname not in self._umap_written:
            self._umap_written.add(fname)
            self._save_csv(adata, fname, "X_umap")
<<<<<<< HEAD:utils/compare_results_utils.py
       
=======

    # MAIN DRIVER 
>>>>>>> e362fe11d74fc7a997deee93612524376f027bf1:scMEDAL/utils/compare_results_utils.py
    def get_dimensionality_reduction_plots(self, process_allbatches=True, issparse=True):

        # sanity: you can?t request ?subset only? without a subset size
        if self.n_batches_sample is None and not process_allbatches:
            raise ValueError("n_batches_sample=None but process_allbatches=False")

        for i, ipref in enumerate(np.unique(self.df["input_prefix"])):

            #  1. load & preprocess the INPUT matrix 
            in_path = self.df.loc[self.df["input_prefix"] == ipref, "InputsPath"].values[0]
            X, var, obs = read_adata(in_path, issparse=issparse)
            X = X.toarray() if issparse else X
            X, obs = self._subset(self._scale(X), obs)          # optional cell-subsample

            # choose the batch subset once and reuse for every prefix / latent
            if i == 0:
                all_batches = np.unique(obs[self.batch_col])
                self.batches_sample = (
                    all_batches if self.n_batches_sample is None else
                    self.rng.choice(all_batches, size=self.n_batches_sample, replace=False)
                )
<<<<<<< HEAD:utils/compare_results_utils.py
            
            # ?? 2. build an AnnData with PCA on the full matrix (always) 
            ad_full = AnnData(X, obs=obs, var=var)
            ad_full.obsm["X_pca"] = self._pca(X)                 # PCA always computed
=======

            # ?? 2. build an AnnData with PCA on the full matrix (always) 
            ad_full = AnnData(X, obs=obs, var=var)
            ad_full.obsm["X_pca"] = self._pca(X)                 # PCA always computed

            # 2A. FULL-data UMAP / plots / CSV  ? only when you ask for them
            if process_allbatches:
                self._embed(ad_full, "X_pca")                    # neighbour graph + UMAP
                self._plot_all(ad_full, f"input_{ipref}",         "celltype", "celltype")
                self._plot_all(ad_full, f"input_{ipref}_batch",   self.batch_col, self.batch_col)
                for c in self.extra_color_cols:
                    if c in obs:
                        self._plot_all(ad_full, f"input_{ipref}_{c}", c, c)
                self._save_umap_once(ad_full, f"input_{ipref}")   # CSV once per stem

            # always save the PCA latent (needed downstream, cheap)
            self._save_csv(ad_full, f"input_{ipref}_modellatent", "X_pca")

            # 2B. SUBSET input (only if n_batches_sample specified)
            if self.n_batches_sample is not None:
                ad_sub = filter_adata_by_batch(ad_full.copy(), self.batches_sample, self.batch_col)
                ad_sub.obsm["X_pca"] = self._pca(ad_sub.X)
                self._embed(ad_sub, "X_pca")
                stub = f"input_{ipref}_{len(self.batches_sample)}{self.batch_col}"
                self._plot_all(ad_sub, stub,          "celltype", "celltype")
                self._plot_all(ad_sub, f"{stub}_batch", self.batch_col, self.batch_col)
                for c in self.extra_color_cols:
                    if c in ad_sub.obs:
                        self._plot_all(ad_sub, f"{stub}_{c}", c, c)
                self._save_csv(ad_sub, stub, "X_umap")            # subset UMAP CSV

            # ?? 3. LATENT files 
            lat_df = self.df.loc[self.df["input_prefix"] == ipref, ["latent_prefix", "LatentPath"]]

            # (a) FULL-data latent outputs only when BOTH:
            #     process_allbatches is True  AND  we did NOT request a subset
            if process_allbatches and self.n_batches_sample is None:
                for lp, lpath in lat_df.values:
                    ad_lat = AnnData(np.load(lpath)[self.sampled_indices], obs=obs)
                    self._embed(ad_lat, "X")
                    self._plot_all(ad_lat, lp, "celltype", "celltype")
                    self._plot_all(ad_lat, f"{lp}_batch", self.batch_col, self.batch_col)
                    for c in self.extra_color_cols:
                        if c in obs:
                            self._plot_all(ad_lat, f"{lp}_{c}", c, c)
                    self._save_umap_once(ad_lat, lp)

            # (b) SUBSET latent outputs when we have a batch subset
            if self.n_batches_sample is not None:
                for lp, lpath in lat_df.values:
                    ad_lat_sub = AnnData(np.load(lpath)[self.sampled_indices], obs=obs)
                    ad_lat_sub = filter_adata_by_batch(ad_lat_sub, self.batches_sample, self.batch_col)
                    self._embed(ad_lat_sub, "X")
                    stub_lat = f"{lp}_{len(self.batches_sample)}{self.batch_col}"
                    self._plot_all(ad_lat_sub, stub_lat,          "celltype", "celltype")
                    self._plot_all(ad_lat_sub, f"{stub_lat}_batch", self.batch_col, self.batch_col)
                    for c in self.extra_color_cols:
                        if c in ad_lat_sub.obs:
                            self._plot_all(ad_lat_sub, f"{stub_lat}_{c}", c, c)
                    self._save_csv(ad_lat_sub, stub_lat, "X_umap")

            gc.collect()
>>>>>>> e362fe11d74fc7a997deee93612524376f027bf1:scMEDAL/utils/compare_results_utils.py

            try:
                # 2A. FULL-data UMAP / plots / CSV  ? only when you ask for them
                self._embed(ad_full, "X_pca")                    # neighbour graph + UMAP
                self._plot_all(ad_full, f"input_{ipref}",         "celltype", "celltype")
                self._plot_all(ad_full, f"input_{ipref}_batch",   self.batch_col, self.batch_col)
                for c in self.extra_color_cols:
                    if c in obs:
                        self._plot_all(ad_full, f"input_{ipref}_{c}", c, c)
                self._save_umap_once(ad_full, f"input_{ipref}")   # CSV once per stem

                # always save the PCA latent (needed downstream, cheap)
                self._save_csv(ad_full, f"input_{ipref}_modellatent", "X_pca")
            except:
                print("I broke")
                pass

            # 2B. SUBSET input (only if n_batches_sample specified)
            if self.n_batches_sample is not None:
                ad_sub = filter_adata_by_batch(ad_full.copy(), self.batches_sample, self.batch_col)
                ad_sub.obsm["X_pca"] = self._pca(ad_sub.X)
                self._embed(ad_sub, "X_pca")
                stub = f"input_{ipref}_{len(self.batches_sample)}{self.batch_col}"
                self._plot_all(ad_sub, stub,          "celltype", "celltype")
                self._plot_all(ad_sub, f"{stub}_batch", self.batch_col, self.batch_col)
                for c in self.extra_color_cols:
                    if c in ad_sub.obs:
                        self._plot_all(ad_sub, f"{stub}_{c}", c, c)
                self._save_csv(ad_sub, stub, "X_umap")            # subset UMAP CSV

            # ?? 3. LATENT files 
            lat_df = self.df.loc[self.df["input_prefix"] == ipref, ["latent_prefix", "LatentPath"]]

            # (a) FULL-data latent outputs only when BOTH:
            #     process_allbatches is True  AND  we did NOT request a subset
            if process_allbatches and self.n_batches_sample is None:
                for lp, lpath in lat_df.values:
                    ad_lat = AnnData(np.load(lpath)[self.sampled_indices], obs=obs)
                    self._embed(ad_lat, "X")
                    self._plot_all(ad_lat, lp, "celltype", "celltype")
                    self._plot_all(ad_lat, f"{lp}_batch", self.batch_col, self.batch_col)
                    for c in self.extra_color_cols:
                        if c in obs:
                            self._plot_all(ad_lat, f"{lp}_{c}", c, c)
                    self._save_umap_once(ad_lat, lp)

            # (b) SUBSET latent outputs when we have a batch subset
            if self.n_batches_sample is not None:
                for lp, lpath in lat_df.values:
                    ad_lat_sub = AnnData(np.load(lpath)[self.sampled_indices], obs=obs)
                    ad_lat_sub = filter_adata_by_batch(ad_lat_sub, self.batches_sample, self.batch_col)
                    self._embed(ad_lat_sub, "X")
                    stub_lat = f"{lp}_{len(self.batches_sample)}{self.batch_col}"
                    self._plot_all(ad_lat_sub, stub_lat,          "celltype", "celltype")
                    self._plot_all(ad_lat_sub, f"{stub_lat}_batch", self.batch_col, self.batch_col)
                    for c in self.extra_color_cols:
                        if c in ad_lat_sub.obs:
                            self._plot_all(ad_lat_sub, f"{stub_lat}_{c}", c, c)
                    self._save_csv(ad_lat_sub, stub_lat, "X_umap")

            gc.collect()
