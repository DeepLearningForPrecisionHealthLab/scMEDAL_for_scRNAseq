
from anndata import AnnData
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
# Set the Matplotlib backend to 'Agg'
matplotlib.use('Agg')
import numpy as np
import scipy.sparse as sp
import sklearn.metrics as mpd
import gc
import os
import glob
import re

from scipy.spatial.distance import pdist, squareform
from utils.utils import read_adata, min_max_scaling,save_adata,calculate_zscores
# I run it with Aixa_genomap
from genomap.genomapOPT import create_space_distributions, gromov_wasserstein_adjusted_norm
from genomap.genomap import createMeshDistance,createInteractionMatrix


"""
    This file includes one function adapted from Genomap:
    `construct_genomap`

    Genomap repository:
    https://github.com/xinglab-ai/genomap

    Original genomap's source:
    https://github.com/xinglab-ai/genomap/blob/main/genomap/genomap.py

    Genomaps license:
    CC BY-NC-ND 2.0
    https://creativecommons.org/licenses/by-nc-nd/2.0/

    Original genomaps' license link:
    https://github.com/xinglab-ai/genomap/blob/main/LICENSE.txt

    Genomaps Reference: Islam, M.T., Xing, L. Cartography of Genomic Interactions Enables Deep Analysis of Single-Cell Expression Data. Nat Commun 14, 679 (2023). https://doi.org/10.1038/s41467-023-36383-6

    Only the function `construct_genomap` was adapted from Genomap.
    The rest of the code in this file was developed in this project.

    Changes made to `construct_genomap`:
    - added NaN handling for the interaction matrix
    - modified the returned outputs

    Original source and license are acknowledged here.
"""


def construct_genomap(data,rowNum,colNum,epsilon=0,num_iter=1000):

    """
    Adapted function from genomap repository: https://github.com/xinglab-ai/genomap/blob/main/genomap/genomap.py

    Constructs 2D "genomaps" by coupling a gene-gene interaction matrix with
    a grid distance matrix using Gromov-Wasserstein. Note that GW transport
    yields fractional assignments, so forcibly rounding positions to integer
    grid coordinates can cause collisions (duplicate integer indices) and
    result in missing or overwritten gene indices.

    Returns the constructed genomaps
    I added code to avoid nan in interactions matrix


    Parameters
    ----------
    data : ndarray, shape (cellNum, geneNum)
         gene expression data in cell X gene format. Each row corresponds
         to one cell, whereas each column represents one gene
    rowNum : int, 
         number of rows in a genomap
    colNum : int,
         number of columns in a genomap

    Returns
    -------
    genomaps : ndarray, shape (rowNum, colNum, zAxisDimension, cell number)
           genomaps are formed for each cell. zAxisDimension is more than
           1 when 3D genomaps are created. 
    """

    sizeData=data.shape
    numCell=sizeData[0]
    numGene=sizeData[1]
    # distance matrix of 2D genomap grid
    distMat = createMeshDistance(rowNum,colNum)
    # gene-gene interaction matrix 
    interactMat = createInteractionMatrix(data, metric='correlation')
    # I added the following line to avoid nan in interactions matrix
    interactMat = np.nan_to_num(interactMat)

    totalGridPoint=rowNum*colNum
    
    if (numGene<totalGridPoint):
        totalGridPointEff=numGene
    else:
        totalGridPointEff=totalGridPoint
    
    M = np.zeros((totalGridPointEff, totalGridPointEff))
    p, q = create_space_distributions(totalGridPointEff, totalGridPointEff)

   # Coupling matrix 
    T = gromov_wasserstein_adjusted_norm(
    M, interactMat, distMat[:totalGridPointEff,:totalGridPointEff], p, q, loss_fun='kl_loss', epsilon=epsilon,max_iter=num_iter)
 
    projMat = T*totalGridPoint
    # Data projected onto the couping matrix
    projM = np.matmul(data, projMat)

    genomaps = np.zeros((numCell,rowNum, colNum, 1))

    px = np.asmatrix(projM)

    # Formation of genomaps from the projected data
    for i in range(0, numCell):
        dx = px[i, :]
        fullVec = np.zeros((1,rowNum*colNum))
        fullVec[:dx.shape[0],:dx.shape[1]] = dx
        #ex = np.reshape(fullVec, (rowNum, colNum), order='F').copy()
        ex = np.reshape(fullVec, (rowNum, colNum), order='C').copy()
        genomaps[i, :, :, 0] = ex
        
    geno_dict = {"genomaps":genomaps,"T":T,"totalGridPoint":totalGridPoint} 
    return geno_dict 
    #return genomaps

def create_gene_coordinates_mapping(projMat, gene_names, num_genes=2916, rowNum=54, colNum=54):
    """
    Applies a projection matrix to a diagnostic array of gene indices, reshapes the
    result into a 54×54 grid, and maps each gene to an (x,y) coordinate. Note that
    if 'projMat' contains fractional assignments (e.g., from Gromov-Wasserstein),
    rounding to int can cause collisions and potentially overwrite or miss certain
    gene indices. Any missing genes are reported in the console.
    """
    # Create and transform the diagnostic matrix
    diagnostic_matrix = np.arange(num_genes).reshape(1, -1)
    transformed_indices = np.matmul(diagnostic_matrix, projMat).flatten()
    px = np.round(transformed_indices, 2)
    
    # Reshape into a rowNum, colNum matrix
    # Order ='F' returns transposed indexes
    # genomaps_diagnostic = np.reshape(px, (rowNum, colNum), order='F')
    # Reshape into a rowNum, colNum matrix
    genomaps_diagnostic = np.reshape(px, (rowNum, colNum), order='C')  # Default is 'C' order (Not transposed)
    
    # Map genes to coordinates
    gene_to_coordinates = {}
    count = 0
    for x in range(rowNum):
        for y in range(colNum):
            gene_index = int(genomaps_diagnostic[x, y])
            if 0 <= gene_index < len(gene_names):
                gene_name = gene_names[gene_index]
                gene_to_coordinates[gene_name] = (x, y)
                count += 1
            else:
                print(f"Index {gene_index} not assigned to a gene name")

    expected_indices = set(range(len(gene_names)))
    found_indices = set(int(genomaps_diagnostic[i, j]) for i in range(rowNum) for j in range(colNum) if 0 <= int(genomaps_diagnostic[i, j]) < len(gene_names))
    missing_indices = expected_indices - found_indices
    print("Missing indices:", missing_indices)

    # Example usage:
    # projMat = np.random.rand(2916, 2916)  # Example initialization; replace with actual matrix
    # gene_names = ['Gene1', 'Gene2', ..., 'Gene2916']  # Define a list of gene names
    # mapping = create_gene_name_coordinates_mapping(projMat, gene_names)
    
    return gene_to_coordinates


def process_data_genomap(inputs_path, recon_path=None, ncells=50000, ngenes=2916, return_input_zscores=False,issparse=True):
    """
    Process input and recon data, calculate z-scores, and return a subset of the dataset.

    Args:
        inputs_path (str): Path to the input data file.
        recon_path (str,optional): Path to the reconstruction data file.
        ncells (int): Number of cells to subset. Default is 50000.
        ngenes (int): Number of genes to subset. Default is 2916.
        return_input_zscores (bool): Flag to return z-scores from input data. Default is False.
        issparse (bool): Flag to read adata as sparse matrix. Default is True.

    Returns:
        dict: Nested dictionary containing subsets of z-scores and AnnData objects for input and recon data.
    """
    # Load input data
    X_input, var, obs = read_adata(inputs_path, issparse=issparse)
    adata_input_train = AnnData(X_input, obs=obs, var=var)

    # Process input data: Get z-scores
    X_input_dense = adata_input_train.X.toarray() if sp.issparse(adata_input_train.X) else adata_input_train.X
    input_z_scores = calculate_zscores(X_input_dense) if return_input_zscores else None

    # Subset dataset to ncells, ngenes
    subset_input_z_scores = input_z_scores[:ncells, :ngenes] if input_z_scores is not None else None

    if subset_input_z_scores is not None:
        print("subset_input_z_scores shape:", subset_input_z_scores.shape)

    out_dict = {"input": {"z_scores": subset_input_z_scores,
            "adata": adata_input_train}}

    if recon_path is not None:
        # Load recon data
        loaded_data = np.load(recon_path, allow_pickle=False)
        X_recon = loaded_data.item() if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object else loaded_data

        # Load cell metadata
        meta_path = glob.glob(inputs_path+'/*meta*.csv')[0]
        obs = pd.read_csv(meta_path)
        adata_recon_train = AnnData(X_recon, obs=obs, var=var)


        # Process recon data: Get z-scores
        X_recon_dense = adata_recon_train.X.toarray() if sp.issparse(adata_recon_train.X) else adata_recon_train.X
        recon_z_scores = calculate_zscores(X_recon_dense)

        # Subset dataset to ncells, ngenes
        subset_recon_z_scores = recon_z_scores[:ncells, :ngenes] 

        print("subset_recon_z_scores shape:", subset_recon_z_scores.shape)
        out_dict["recon"] =  {"z_scores": subset_recon_z_scores,"adata": adata_recon_train}

    return out_dict


def plot_genomaps(genoMaps, labels, filename):
    """
    Plots genoMaps with a global color scale and includes a color bar.

    Args:
        genoMaps (numpy.ndarray): Array of genoMaps to plot, expected shape (num_cells, height, width, channels).
        labels (list): List of labels for each cell to be used in the plot titles.
        filename (str): The file path to save the figure.

    Returns:
        None
    """
    # Determine the global min and max for the color scale
    global_min = np.percentile(genoMaps, 1)
    global_max = np.percentile(genoMaps, 98)

    # Create a figure with 10x5 subplots
    fig, axes = plt.subplots(10, 5, figsize=(20, 40))

    # Iterate over the subplots and plot each cell
    for i, ax in enumerate(axes.flat):
        if i < len(genoMaps):
            findI = genoMaps[i, :, :, :]
            #for now i wont clip data
            #clipped_data = np.clip(findI, global_min, global_max)
            im = ax.imshow(findI, origin='lower', extent=[0, 10, 0, 10], aspect=1, vmin=global_min, vmax=global_max, cmap='viridis')
            ax.set_title(f'Genomap of cell {i+1}\n' + labels[i])
        else:
            ax.axis('off')

    # Add a color bar below the figure
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', shrink=0.8)

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make room for colorbar
    
    # Save the figure
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    gc.collect()




def process_and_plot_genomaps_singlepath(inputs_path, ncells, ngenes, rowNum, colNum, epsilon, num_iter, output_folder,genomap_name,gene_names=None):
    """
    Process the input data to generate z-scores and genoMaps, then plot and save the results.

    Args:
        inputs_path (str): Path to the input data file.
        ncells (int): Number of cells to subset.
        ngenes (int): Number of genes to subset.
        rowNum (int): Number of rows for the genoMap.
        colNum (int): Number of columns for the genoMap.
        epsilon (float): Epsilon value for the genoMap construction.
        num_iter (int): Number of iterations for the genoMap construction.
        output_folder (str): Path to the folder where outputs will be saved.
        genomap_name (str): Name identifier for the genoMap.
        gene_names (list,optional): List with the gene names of the genomap. The number of gene_names has to be equal to ngenes
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Process the data, returning input z-scores only for the first iteration
    result = process_data_genomap(inputs_path, recon_path=None, ncells=ncells, ngenes=ngenes, return_input_zscores=True)

    # Get input z-scores and construct genoMaps for the input data
    input_z_scores = result['input']['z_scores']
    try:
        genoMaps_input = construct_genomap(input_z_scores, rowNum, colNum, epsilon=epsilon, num_iter=num_iter)

        projMat = genoMaps_input["T"]*genoMaps_input["totalGridPoint"]

        # Save genoMaps and transition matrix (T) for input data
        np.save(os.path.join(output_folder, f'genomap_{genomap_name}.npy'), genoMaps_input["genomaps"])
        np.save(os.path.join(output_folder, f'T_input_{genomap_name}.npy'), genoMaps_input["T"])

        # Plot and save the genoMaps for input data
        input_ct_labels = result['input']['adata'].obs['celltype'].tolist()
        input_batch_labels = result['input']['adata'].obs['batch'].tolist()
        input_labels = [f'{ct}_{b}' for ct, b in zip(input_batch_labels, input_ct_labels)]
        
        plot_genomaps(genoMaps_input["genomaps"], input_labels, os.path.join(output_folder, f'first50genomaps_from_CMmultibatch_{genomap_name}.png'))
        # Map genes to genomap
        if gene_names is not None:
            gene_to_coordinates = create_gene_coordinates_mapping(projMat, gene_names, ngenes, rowNum, colNum)
            gene_to_coordinates_df = pd.DataFrame(gene_to_coordinates).T
            gene_to_coordinates_df.columns=["pixel_i","pixel_j"]
            gene_to_coordinates_df.to_csv(os.path.join(output_folder, f'gene_coordinates_{genomap_name}.csv'))
    except:
        print("failed to construct genomap...") 


def process_and_plot_genomaps(df, models, types, splits, rowNum, colNum, output_folder, return_input_zscores=True, ncells=50000, ngenes=2916, epsilon=0.0, num_iter=200):
    """
    Processes data, constructs genoMaps, and plots/saves the results for specified models, types, and splits.

    Args:
        df (DataFrame): DataFrame containing the paths to input and reconstruction data.
        models (list): List of model names.
        types (list): List of data types (e.g., 'train', 'test', validate').
        splits (list): List of split indices.
        rowNum (int): Number of rows for genoMap construction.
        colNum (int): Number of columns for genoMap construction.
        output_folder (str): Path to the folder where the results will be saved.
        return_input_zscores (bool): Flag to determine if input z-scores should be returned.

    Returns:
        dict: Dictionary containing the processed results.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize the results dictionary
    results = {}

    # Loop through each model
    for model in models:
        results[model] = {}
        
        # Loop through each data type (train, test, validate)
        for Type in types:
            results[model][Type] = {}
            
            # Loop through each split
            for Split in splits:
                print("\nProcessing:",model,Type,Split)
                # All the batches have the same input for the same type and split
                k = 0
                results[model][Type][Split] = {
                    'process_data_result': {},
                    'genoMaps_input_train': "",
                    'genoMaps_recon_train': {}
                }

                # Get the input and reconstruction paths from the DataFrame
                inputs_path = df.loc[
                    (df["Key"] == model) & 
                    (df["Type"] == Type) & 
                    (df["Split"] == Split), 
                    "InputsPath"
                ].values[0]
                
                recon_paths = df.loc[
                    (df["Key"] == model) & 
                    (df["Type"] == Type) & 
                    (df["Split"] == Split), 
                    "ReconPath"
                ].values
                
                # Process each reconstruction path
                print("\nrecon paths: ",recon_paths)
                for recon_path in recon_paths:
                    recon_prefix = recon_path.split("/")[-1].split(".npy")[0]
                    print("\nProcessing ",recon_prefix)
                    # Process the data, returning input z-scores only for the first iteration
                    result = process_data_genomap(
                        inputs_path, recon_path, 
                        ncells=ncells, ngenes=ngenes, 
                        return_input_zscores=(k == 0)
                    )
                    # Create genomaps if return_input_zscores=True and k==0, ie first recon for the same type and split
                    if (k == 0) & return_input_zscores:

                        # Get input z-scores and construct genoMaps for the input data
                        input_z_scores = result['input']['z_scores']
                        print("Computing genomaps for input..")

                        try:
                            genoMaps_input = construct_genomap(
                                input_z_scores, rowNum, colNum, 
                                epsilon=epsilon, num_iter=num_iter
                            ) if input_z_scores is not None else None
                            
                            results[model][Type][Split]['genoMaps_input_train'] = genoMaps_input["genomaps"]

                            if genoMaps_input["genomaps"] is not None:
                                # Save genoMaps and transition matrix (T) for input data
                                save_genomaps(genoMaps_input, output_folder, model, Type, Split, data_type="input")

                                # Plot and save the genoMaps for input data
                                input_ct_labels = result['input']['adata'].obs['celltype'].tolist()
                                input_batch_labels = result['input']['adata'].obs['batch'].tolist()

                                input_labels = [f'{ct}_{b}' for ct, b in zip(input_batch_labels, input_ct_labels)]
                                plot_genomaps(genoMaps_input["genomaps"], input_labels, os.path.join(output_folder, f'{model}_{Type}_{Split}_input.png'))
                        except:
                            print(f"Failed to construct genomap for {model} {Type} {Split} {recon_path}")

                    # Get reconstruction z-scores and construct genoMaps for the reconstruction data
                    recon_z_scores = result['recon']['z_scores']
                    print("Computing genomaps for recon..")
                    genoMaps_recon = construct_genomap(
                        recon_z_scores, rowNum, colNum, 
                        epsilon=epsilon, num_iter=num_iter
                    )

                    results[model][Type][Split]['process_data_result'][recon_prefix] = result
                    results[model][Type][Split]['genoMaps_recon_train'][recon_prefix] = genoMaps_recon["genomaps"]

                    if genoMaps_recon["genomaps"] is not None:
                        # Save genoMaps and transition matrix (T) for reconstruction data
                        save_genomaps(genoMaps_recon, output_folder, model,Type, Split=Split,data_type="recon",recon_prefix = recon_prefix)

                        # Plot and save the genoMaps for reconstruction data
                        recon_ct_labels = result['recon']['adata'].obs['celltype'].tolist()
                        n_labels = len(recon_ct_labels)
                        # recon original
                        if recon_prefix in ["recon_train", "recon_val"]:
                            recon_batch_labels = result['recon']['adata'].obs['batch'].tolist()
                        # cf recon 
                        else:
                            recon_batch_labels = [recon_prefix ]*n_labels
                            
                        
                        recon_labels = [f'{ct}_{b}' for ct, b in zip(recon_batch_labels, recon_ct_labels)]
                        plot_genomaps(genoMaps_recon["genomaps"], recon_labels, os.path.join(output_folder, f'genomap_{recon_prefix}_{model}_{Split}.png'))
                    
                    k += 1
                    gc.collect()

    return results

def save_genomaps(genoMaps, output_folder, model, Type, Split, data_type, recon_prefix=None):
    """
    Saves the genoMaps and transition matrices as .npy files.

    Args:
        genoMaps (dict): Dictionary containing the genoMaps and transition matrices.
        output_folder (str): Path to the folder where the results will be saved.
        model (str): Model name.
        Type (str): Data type (e.g., 'train', 'test', validate').
        Split (int): Split index.
        data_type (str): Type of data ('input' or 'recon').
        recon_prefix (str): Prefix for reconstruction files (optional).
    """
    if data_type == "input":
        np.save(os.path.join(output_folder, f'genomap_input_{model}_{Type}_{Split}.npy'), genoMaps["genomaps"])
        np.save(os.path.join(output_folder, f'T_input_{model}_{Type}_{Split}.npy'), genoMaps["T"])
    else:
        np.save(os.path.join(output_folder, f'genomap_{recon_prefix}_{model}_{Split}.npy'), genoMaps["genomaps"])
        np.save(os.path.join(output_folder, f'T_{recon_prefix}_{model}_{Split}.npy'), genoMaps["T"])



import numpy as np

def get_genomapfromT(t_files_path, inputs_path, recon_path,ncells=50000, ngenes=2916,colNum = 54 ,rowNum = 54,gene_names=None):
    """
    Generates genomaps from given input data and projection matrix.

    Parameters:
    - t_files_path (str): Path to the T matrix file.
    - inputs_path (str): Path to the input data.
    - recon_path (str): Path to the reconstructed data.
    - ncells (int, optional): Number of cells in the data. Default is 50000.
    - ngenes (int, optional): Number of genes in the data. Default is 2916.
    - colNum (int, optional): Number of columns in the genomap. Default is 54.
    - rowNum (int, optional): Number of rows in the genomap. Default is 54.
    - gene_names (list, optional): list of gene names for coordinates mapping

    Returns:
    dict: A dictionary containing:
        - 'genomaps': A 4D numpy array representing the genomaps.
        - 'genes_coor_map': A dictionary mapping gene names to their coordinates in the genomap.
    """
    # Load the T matrix
    T = np.load(t_files_path)
    
    # Process data and get z scores
    out_dict = process_data_genomap(inputs_path, recon_path, ncells, ngenes, return_input_zscores=False)
    
    # Get genomap without perturbation
    data = out_dict['recon']["z_scores"].copy()
    sizeData = data.shape
    print("sizeData", sizeData)
    
    # Define parameters for genomap construction
    numCell = sizeData[0]
    numGene = sizeData[1]
    totalGridPoint = ngenes
    
    # Project data onto the coupling matrix
    projMat = T * totalGridPoint
    projM = np.matmul(data, projMat)
    
    # Initialize genomaps
    genomaps = np.zeros((numCell, rowNum, colNum, 1))
    px = np.asmatrix(projM)
    
    # Formation of genomaps from the projected data
    for i in range(numCell):
        dx = px[i, :]
        fullVec = np.zeros((1, rowNum * colNum))
        fullVec[:dx.shape[0], :dx.shape[1]] = dx
        #ex = np.reshape(fullVec, (rowNum, colNum), order='F').copy()
        ex = np.reshape(fullVec, (rowNum, colNum), order='C').copy()
        genomaps[i, :, :, 0] = ex

    # Map genes to genomap
    if gene_names is not None:
        gene_to_coordinates = create_gene_coordinates_mapping(projMat, gene_names, ngenes, rowNum, colNum)
        return {"genomaps": genomaps, "genes_coor_map": gene_to_coordinates}
    else:
        return {"genomaps": genomaps}

import matplotlib.pyplot as plt

def plot_genomap_with_genes(cell_geno, top10hvg_genes, gene_to_coordinates):
    """
    Plots a genomap with identified genes highlighted in red.
    
    Parameters:
    - cell_geno: numpy array representing the genomap data matrix.
    - top10hvg_genes: list of top 10 highly variable genes.
    - gene_to_coordinates: dictionary mapping gene names to their coordinates.
    """
    # Set the size of the figure
    fig, ax = plt.subplots(figsize=(10, 10))  # Width and height in inches
    
    # Plot the genomap using an appropriate colormap
    cax = ax.imshow(cell_geno, cmap='viridis')
    
    # Coordinates for the first 20 genes, assuming you have a list of gene names in the correct order
    first_n_genes = top10hvg_genes
    coordinates = [gene_to_coordinates[gene] for gene in first_n_genes]
    
    # Highlight each gene coordinate
    for coord, gene in zip(coordinates, first_n_genes):
        x, y = coord
        ax.scatter(y, x, color='red', s=100, alpha=0.3)  # Adjust size and color as needed
        gene_str = gene.decode('utf-8') if isinstance(gene, bytes) else gene
        ax.text(y, x, gene_str, color='white', ha='right', va='top')
    
    # Add a colorbar to the plot
    fig.colorbar(cax, orientation='vertical', fraction=0.02, pad=0.04)
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()




def sample_cells(obs, n_cells, celltype=None, batch_col="batch", force_batches=None, seed=None):
    """
    Sample cells from a dataframe while ensuring representation from specified batches and cell types.

    Parameters:
    obs (pd.DataFrame): The dataframe containing cell data with columns including 'celltype' and 'batch'.
    n_cells (int): The total number of cells to sample.
    celltype (str or list, optional): The cell type(s) to be included in the sampling. Can be a single cell type (str) or a list of cell types (list).
    batch_col (str): The name of the column representing batches in the dataframe.
    force_batches (list, optional): A list of batch names to ensure at least one cell from each batch is included in the sample.
    seed (int, optional): Random seed for reproducibility.

    Returns:
    pd.DataFrame: A dataframe containing the sampled cells.

    Raises:
    ValueError: If a specified batch or cell type does not exist in the dataframe.
    """
    # Initialize selected_indices list to keep track of selected cells
    print("\nSampling cells.. ")
    selected_indices = []

    # Ensure at least one cell from each forced batch and specified celltype (if provided) is included
    if force_batches:
        for batch in force_batches:
            # Filter cells belonging to the current batch
            batch_cells = obs[obs[batch_col] == batch]
            if celltype is not None:
                if isinstance(celltype, str):
                    # Further filter cells of the specified celltype within the batch
                    batch_cells = batch_cells[batch_cells["celltype"] == celltype]
                    if batch_cells.empty:
                        print(f"No cells available for celltype {celltype} in batch {batch}. Skipping batch.")
                        continue
                        
                    # Randomly select one cell from the batch (and celltype if specified)
                    selected_index = batch_cells.sample(n=1, random_state=seed).index[0]
                    selected_indices.append(selected_index)
                    print(f"Selected 1 cell from batch {batch} with celltype {celltype}.")
                elif isinstance(celltype, list):
                    for ct in celltype:
                        ct_batch_cells = batch_cells[batch_cells["celltype"] == ct]
                        if ct_batch_cells.empty:

                            print(f"No cells available for celltype {celltype} in batch {batch}. Skipping batch.")
                            continue
                        # Randomly select one cell from the batch for each celltype
                        selected_index = ct_batch_cells.sample(n=1, random_state=seed).index[0]
                        selected_indices.append(selected_index)
                        print(f"Selected 1 cell from batch {batch} with celltype {ct}.")
            else:
                if batch_cells.empty:
                    print(f"No cells available in batch {batch}. Skipping batch.")
                    continue
                # Randomly select one cell from the batch without celltype constraint
                selected_index = batch_cells.sample(n=1, random_state=seed).index[0]
                selected_indices.append(selected_index)
                print(f"Selected 1 cell from batch {batch} without specific celltype constraint.")

    # Remove selected cells to avoid re-sampling them
    obs_remaining = obs.drop(selected_indices)

    # Sample remaining cells based on celltype
    if celltype is not None:
        if isinstance(celltype, str):
            # Filter cells of the specified celltype
            celltype_obs = obs_remaining[obs_remaining["celltype"] == celltype]
            # Randomly select the remaining required cells from the filtered cells
            additional_cells = celltype_obs.sample(n=n_cells - len(selected_indices), random_state=seed)
            print(f"Selected {n_cells - len(selected_indices)} additional cells of celltype {celltype}.")
        elif isinstance(celltype, list):
            additional_cells_list = []
            cells_needed_per_type = (n_cells - len(selected_indices)) // len(celltype)
            for ct in celltype:
                ct_cells = obs_remaining[obs_remaining["celltype"] == ct]
                additional_cells_list.append(ct_cells.sample(n=cells_needed_per_type, random_state=seed))
            additional_cells = pd.concat(additional_cells_list)
            print(f"Selected {n_cells - len(selected_indices)} additional cells from specified celltypes.")
    else:
        # Randomly select the remaining required cells from the entire dataset
        additional_cells = obs_remaining.sample(n=n_cells - len(selected_indices), random_state=seed)
        print(f"Selected {n_cells - len(selected_indices)} additional cells without specific celltype constraint.")

    # Combine the selected indices from forced batches and the additional sampled cells
    final_selected_indices = pd.Index(selected_indices).append(additional_cells.index)
    obs_final = obs.loc[final_selected_indices]

    return obs_final


def create_count_matrix_multibatch(recon_prefix, recon_paths, obs, var, n_genes, n_cells, n_batches, out_path, add_inputs_fe=True, n_inputs_fe=2, celltype=None, save_data=False, scaling="min_max", issparse=True,seed=42,force_batches=None):
    """
    Create a concatenated count matrix from cell reconstructions from multiple batches, optionally filtered by cell type.
    This concatenated matrix will be used to generate a genomap.

    Parameters:
    recon_prefix (list of str): List of prefixes indicating the type of data in recon_paths.
    recon_paths (list of str): List of paths to the reconstructed data files.
    obs (DataFrame): DataFrame containing observation (cell) metadata.
    var (DataFrame): DataFrame containing variable (gene) metadata.
    n_genes (int): Number of genes to include in the count matrix.
    n_cells (int): Number of cells to include from each batch.
    n_batches (int): Number of batches to process.
    out_path (str): Output directory path to save the concatenated AnnData object.
    add_inputs_fe (bool, optional): Whether to add cells from the original input data and from the fixed effects recon. Default is True.
    n_inputs_fe (int, optional): If add_inputs_fe = True. Specify how many reconstructions you want to add to the count matrix. Default is 2. 
        One for inputs and another one for fixed effects (fe). You could also add a fe classifier recon and a base autoencoder recon.
    celltype (str or list, optional): Specific cell type(s) to filter for. If None, include all cell types. Default is None.
    save_data (bool, optional): Whether to save the resulting AnnData object to disk. Default is False.
    issparse (bool, optional): Flag to read sparse count matrix. Default is True.
    seed (int,optional): seed for reproducibility. Default = 42.
    force_batches (list, optional): A list of batch names to ensure at least one cell from each batch is included in the sample.

    Returns:
    None
    """
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    import os

    # Initialize empty arrays and DataFrame
    x = np.empty((0, n_genes))
    obs_combined = pd.DataFrame()


    # Use sample_cells function to select the appropriate cells
    if celltype is not None:
        obs = sample_cells(obs, n_cells, celltype=celltype, batch_col="batch", force_batches=force_batches, seed=seed)
        selected_indices = obs.index

    if add_inputs_fe:
        n_recon_prefix = n_batches + n_inputs_fe
    else:
        n_recon_prefix = n_batches

    for recon_pref, recon_path in zip(recon_prefix[0:n_recon_prefix], recon_paths[0:n_recon_prefix]):
        print("Creating count matrix multibatch using reconstructions from following batches: ",recon_pref)
        if "input" in recon_pref:
            X, _, _ = read_adata(recon_path, issparse=issparse)
            if issparse:
                X = X.toarray()
            # Input needs to be scaled because reconstructions are in the scaled space
            # Apply scaling to X based on the scaling parameter.
            if scaling == "min_max":
                X = min_max_scaling(X)
            elif scaling == "z_scores":
                X = calculate_zscores(X)
        else:
            X = np.load(recon_path)

        if celltype is None:
            x_i = X[:n_cells, :n_genes]
            obs_i = obs.iloc[:n_cells].copy()
        else:
            x_i = X[selected_indices, :n_genes]
            obs_i = obs.copy()

        # Add a new column that indicates the recon prefix
        obs_i["recon_prefix"] = recon_pref

        x = np.concatenate((x, x_i), axis=0)
        obs_combined = pd.concat([obs_combined, obs_i], axis=0)

    obs_combined.reset_index(drop=True, inplace=True)
    print("Shape of concatenated x:", x.shape)
    print("Shape of concatenated obs:", obs_combined.shape)

    if isinstance(celltype, str):
        celltype_name = celltype.replace("/", "")
    elif isinstance(celltype, list):
        celltype_name = "_".join([ct.replace("/", "") for ct in celltype])
    else:
        celltype_name = None

    folder_name = f"CMmultibatch_{n_cells}_cells_per_batch_{n_batches}batches" if not celltype_name else f"CMmultibatch_{n_cells}_cells_per_batch_{n_batches}batches_{celltype_name}"
    if add_inputs_fe:
        folder_name += f"_with_{n_inputs_fe}fe_input"

    folder_path = os.path.join(out_path, folder_name)

    if not os.path.exists(folder_path):
        print("created folder path:",folder_path)
        os.makedirs(folder_path)

    adata_multibatch = AnnData(x, obs=obs_combined, var=var.iloc[:n_genes, :])

    if save_data:
        save_adata(adata_multibatch, folder_path)
    print("Completed subsampling of adata_multibatch")

    return adata_multibatch


def find_intersection_batches(obs_combined, celltypes):
    """
    Find the intersection of batches that are present in at least one cell from each of the selected celltypes.

    Parameters:
    obs_combined (DataFrame): DataFrame containing observation (cell) metadata with a 'batch' column.
    celltypes (list): List of cell types to check for overlapping batch values.

    Returns:
    intersection_batches (set): Set of batches that are common across all specified cell types.
    """
    # Initialize the set with batches from the first cell type
    intersection_batches = set(obs_combined[obs_combined["celltype"] == celltypes[0]]["batch"].unique())

    # Iterate through the remaining cell types and find the intersection
    for celltype in celltypes[1:]:
        batches = set(obs_combined[obs_combined["celltype"] == celltype]["batch"].unique())
        intersection_batches.intersection_update(batches)

    return intersection_batches


def select_cells_from_batches(obs_combined, celltypes, batches_to_select_from,seed = 42,cell_id_col="_index"):
    """
    Select one cell from each cell type within the specified batches.

    Parameters:
    obs_combined (DataFrame): DataFrame containing observation (cell) metadata.
    celltypes (str or list): A single cell type (str) or list of cell types to select cells from.
    batches_to_select_from (list): List of batches to select cells from.
    cell_id_col(str): column with the cell ids

    Returns:
    cell_ids_2plot (list): List of selected cell IDs.
    """
    import random
    random.seed(seed)
    cell_ids_2plot = []
    if isinstance(celltypes, str):
        celltypes = [celltypes]  # Convert to list for uniform processing

    for batch in batches_to_select_from:
        for celltype in celltypes:
            cells_in_batch = obs_combined[(obs_combined['batch'] == batch) & (obs_combined['celltype'] == celltype)][cell_id_col].values
            if len(cells_in_batch) > 0:
                selected_cell = random.choice(cells_in_batch)
                cell_ids_2plot.append(selected_cell)

    return cell_ids_2plot


def compute_cell_stats_acrossbatchrecon(genomap, cell_indexes_batch_cf, genomap_coordinates, statistic='std', n_top_genes=10, order='C',path_2_genomap='',file_name='cell_id'):
    """
    Calculate standard deviation or variance for genomaps of a single cell across batch reconstructions. 
    Update genomap_coordinates DataFrame.
    
    Args:
        genomap: 4D numpy array with genomap data. Axis = 0 indicates individual cells.
        cell_indexes_batch_cf: List or array of cell indexes for  batch reconstructions of single cell. The same cell was reconstructed for multiple batches. 
        genomap_coordinates: DataFrame containing gene names and pixel coordinates.
        statistic: 'std' for standard deviation, 'var' for variance.
        n_top_genes: Number of top genes to identify.
        order: 'C' for default coordinate order, 'F' for transposed coordinates (i.e., pixel_i and pixel_j are swapped).
        path_2_genomap (str): path to genomap directory. To save stats df.
        file_name (str): Name to save the file. Default:"cell_id".
        
    Returns:
        Updated genomap_coordinates DataFrame with standard deviation/variance and rank.
        
    Explanation:
        - Statistics (standard deviation or variance) are calculated based on the genomap data indexed by cell_indexes_batch_cf.
        - The cell_indexes parameter includes the cell_indexes_batch_cf (they are a supersubset) but they can also be equal.
        - If order is 'C', the function uses pixel_i and pixel_j as they are to index the standard deviation/variance array.
        - If order is 'F', the function transposes pixel_i and pixel_j when indexing the standard deviation/variance array, effectively swapping the coordinates.
    """

    # Calculate the standard deviation or variance along axis 0 (batches)
    if statistic == 'std':
        stat_across_batches = np.std(genomap[cell_indexes_batch_cf, :, :, 0], axis=0, ddof=1)
    elif statistic == 'var':
        stat_across_batches = np.var(genomap[cell_indexes_batch_cf, :, :, 0], axis=0, ddof=1)
    else:
        raise ValueError("Statistic must be 'std' or 'var'")

    # print(f"Shape of {statistic}_across_batches:", stat_across_batches.shape)

    # Add the standard deviation/variance to the genomap_coordinates DataFrame
    genomap_coordinates[statistic] = np.nan
    for idx, row in genomap_coordinates.iterrows():
        pixel_i, pixel_j = int(row['pixel_i']), int(row['pixel_j'])
        if order == 'C':
            genomap_coordinates.at[idx, statistic] = stat_across_batches[pixel_i, pixel_j]
        elif order == 'F':
            genomap_coordinates.at[idx, statistic] = stat_across_batches[pixel_j, pixel_i]

    # Rank the pixels based on the absolute value of standard deviation/variance
    genomap_coordinates['Rank'] = genomap_coordinates[statistic].abs().rank(ascending=False)

    # Convert gene_names from bytes to strings
    #genomap_coordinates['gene_names'] = genomap_coordinates['gene_names'].apply(lambda x: x.strip('b').strip("'"))
    genomap_coordinates["gene_names"] = (
        genomap_coordinates["gene_names"]
            .apply(lambda x: x.split("|")[-1] if isinstance(x, str) else x)
            .apply(lambda x: re.sub(r"^b[\"']?", "", x).rstrip("'").strip() if isinstance(x, str) else x)
            )

    # Add a "Top N" column with True/False
    genomap_coordinates['Top_N'] = False
    top_n_indices = genomap_coordinates.nsmallest(n_top_genes, 'Rank').index
    genomap_coordinates.loc[top_n_indices, 'Top_N'] = True
    # Save

    outfile = os.path.join(
        path_2_genomap,
        f"genomap_{file_name}_{statistic}_{n_top_genes}topvariablegenesacrossbatches.csv"
    )
    genomap_coordinates.to_csv(outfile, index=False)
    print(f"Saved gene {statistic} across batches  to: {outfile}")
    return genomap_coordinates





def adjust_text_positions(x, y, threshold=0.5, offset=0.2):
    """
    Adjust text positions to avoid overlap.

    Args:
        x: List or array of x coordinates.
        y: List or array of y coordinates.
        threshold: Minimum distance to maintain between points.
        offset: Distance to shift overlapping points.
    
    Returns:
        List of adjusted (x, y) coordinates.
    """
    # Calculate pairwise distances
    points = np.array(list(zip(x, y)))
    dists = squareform(pdist(points))

    # Adjust positions to avoid overlap
    adjusted_positions = []
    for i, (x_i, y_i) in enumerate(points):
        shift_x, shift_y = 0, 0
        for j, (x_j, y_j) in enumerate(points):
            if i != j and dists[i, j] < threshold:
                shift_x += offset if x_i <= x_j else -offset
                shift_y += offset if y_i <= y_j else -offset
        adjusted_positions.append((x_i + shift_x, y_i + shift_y))
    
    return adjusted_positions



def plot_cell_recon_genomap(genomap, cell_indexes, genomap_coordinates, obs, original_batch=None, n_top_genes=10, min_val=-5, max_val=10, n_cols = 3,order='C',path_2_genomap='',file_name="cell_id",remove_ticks=False,extra_label_cols=None):
    """
    Plot genomap slices for the given cells and optionally highlight top genes.

    The function displays each selected cell?s 2D genomap (first channel of the
    4D input array) and, if provided, overlays red markers and labels for genes
    flagged as `Top_N == True` in `genomap_coordinates`. The number `n_top_genes`
    is only used in the output filename; actual selection is controlled by the
    `Top_N` column.
    
    Args:
        genomap: 4D numpy array with genomap data.
        cell_indexes: List or array of cell indexes.
        genomap_coordinates: DataFrame containing gene names and pixel coordinates.
        obs: DataFrame containing cell metadata including the column 'recon_prefix' that indicates the name of the reconstruction for a cell id.
        original_batch: Identifier for the original batch, if any. Default is None.
        n_top_genes: Number of top genes to highlight.
        min_val: Minimum value for color scale.
        max_val: Maximum value for color scale.
        n_cols (int): Number of columns for the subplot. Default is 3.
        order: 'C' for default coordinate order, 'F' for transposed coordinates (i.e., pixel_i and pixel_j are swapped).
        path_2_genomap (str): path to genomap directory. To save plot.
        file_name (str): Name to save the file. Default:"cell_id"
        remove_ticks (bool,optional): Flag to remove ticks from the plot. Default=False.
    """
    geno_slices_cell_id = genomap[cell_indexes, :, :, 0]

    n_images = len(cell_indexes)
    
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols+1, 5 * n_rows))
    
    # If axes is 1D, convert to 2D
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)
    if genomap_coordinates is not None:
        top_n_coordinates = genomap_coordinates[genomap_coordinates['Top_N']]

    for i, (cell_index, cell_geno) in enumerate(zip(cell_indexes, geno_slices_cell_id)):
        ax = axes[i // n_cols, i % n_cols]
        im = ax.imshow(cell_geno, cmap='viridis', vmin=min_val, vmax=max_val)

        # Remove ticks
        if remove_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        # # Add gene labels if genomap_coordinates is provided
        if genomap_coordinates is not None:
            # Get coordinates
            if order == 'C':
                x, y = top_n_coordinates['pixel_i'], top_n_coordinates['pixel_j']
            elif order == 'F':
                x, y = top_n_coordinates['pixel_j'], top_n_coordinates['pixel_i']

            # Adjust text positions
            adjusted_positions = adjust_text_positions(x, y, threshold=3, offset=6)

            for (adj_x, adj_y), (pixel_i, pixel_j), gene in zip(adjusted_positions, zip(x, y), top_n_coordinates['gene_names']):
                ax.plot(pixel_i, pixel_j, 'o', markerfacecolor='none', markeredgecolor='red', markersize=6, markeredgewidth=2)
                ax.text(adj_x, adj_y, gene, color='black', ha='left', va='center', fontweight='bold', fontsize=12, clip_on=False)

        
        # print("cell_index",cell_index)
        # print("obs_index",obs.index)
        # recon_prefix = obs.loc[cell_index, "recon_prefix"]
        # ax.set_title(recon_prefix)
        # if 'input' in recon_prefix:
        #     ax.set_title(f'{recon_prefix}\noriginal batch: {original_batch}')
        # elif original_batch in recon_prefix:
        #     ax.set_title(f'{recon_prefix}\n(original batch)', color='red')
        # else:
        #     ax.set_title(f'{recon_prefix}')
        
        # Build the title
        recon_prefix = obs.loc[cell_index, "recon_prefix"]
        title_parts  = [recon_prefix]

        if 'input' in recon_prefix:
            title_parts.append(f'original batch: {original_batch}')
        elif original_batch in recon_prefix:
            title_parts[-1] = f'{recon_prefix} (original batch)'  # red below
        else:
            title_parts[-1] = f'cf batch: {recon_prefix}' 


        # Extra labels
        if extra_label_cols:
            extra_values = [str(obs.loc[cell_index, col]) for col in extra_label_cols]
            title_parts.append('sc origin:' + ''.join(extra_values)) 

        title_text = '\n'.join(title_parts)
        ax.set_title(title_text,
                     color='red' if original_batch and
                                    original_batch in recon_prefix else 'black')






    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    cbar_ax = fig.add_axes([0.94, 0.15, 0.005, 0.7])  # Adjusted for vertical color bar
    fig.colorbar(im, cax=cbar_ax, orientation='vertical')

    fig.suptitle(file_name, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.97])
    file_name = file_name.replace("/", "")
    if genomap_coordinates is not None:
        final_file_name = f"genomap_{file_name}_{n_top_genes}topvariablegenesacrossbatches"
    else:
        final_file_name = f"genomap_{file_name}" 

    fig.savefig(os.path.join(path_2_genomap, final_file_name))





