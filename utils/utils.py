import anndata
import pandas as pd
import os
import glob
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
import h5py
# Some docstring are written using chatgpt4
# If we want to re run the whole clustering analysis, we have to delete the results
def del_clustering_analysis(adata):
    """
    Delete results from PCA+neighbors+tsne+louvain+plots
    Args:
        adata: ann object
    """
    #get plot_keys
    uns_keys=list(adata.uns.keys())
    plot_keys=[k for k in uns_keys if k.split("_")[-1]=="colors" ]
    #delete results
    for key in ['pca', 'neighbors', 'louvain']+plot_keys:
        del adata.uns[key]
    del adata.obsp['distances']
    del adata.obsp['connectivities']
    del adata.varm['PCs']
    del adata.obsm['X_pca']
    del adata.obsm['X_tsne']
    del adata.obs['louvain']
    return adata

#Compute DB and CH
def get_clustering_scores(adata,use_rep, labels):
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score,silhouette_score
    """ Get CH, DB and silhouette scores from adata object
    Args:
        adata: anndata object
        use rep: Use indicated representation. Any key for .obsm is valid. If use_rep =="X", then it will call adata.X
        labels: Any key for .obs is valid
    Return:
        df_scores: pd.DataFrame with CH and DB scores for every object
    
    """
    dict_scores={}
    
    print("Computing scores \nDB: lower values --> better clustering")
    print("1/DB, CH, Silhouette: higher values --> better clustering")

    df_scores = pd.DataFrame()   

    for label in labels:
        if use_rep =="X":
            db = davies_bouldin_score(adata.X,adata.obs[label])
            inv_db = 1/db
            ch = calinski_harabasz_score(adata.X, adata.obs[label])
            s = silhouette_score(adata.X, adata.obs[label])

        else:
            #calculate scores
            db = davies_bouldin_score(adata.obsm[use_rep],adata.obs[label])
            inv_db = 1/db
            ch = calinski_harabasz_score(adata.obsm[use_rep], adata.obs[label])
            s = silhouette_score(adata.obsm[use_rep], adata.obs[label])
        
        dict_scores[label] = [db,inv_db,ch,s]
        df_scores = pd.DataFrame(dict_scores, index=['db','1/db','ch','silhouette'])

    if len(labels) > 1:
        #add ratio col

        for i, label in enumerate(labels[0:-1]): 
            #ratio: label i/ label i+1
            ratio_col = "ratio_"+labels[i]+"-"+labels[i+1]

            df_scores.loc[:,ratio_col] = df_scores.loc[:,labels[i]] / df_scores.loc[:,labels[i+1]] 
    
    return df_scores


def get_clustering_scores_optimized(adata, use_rep, labels):
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
    """
    Optimized version of getting clustering scores from an AnnData object.

    Args:
    - adata: An AnnData object.
    - use_rep: Representation to use. Any key for .obsm is valid. If 'X', then adata.X is used.
    - labels: Labels for clustering. Any key for .obs is valid.

    Returns:
    - A pandas DataFrame with Davies-Bouldin (DB), inverse DB, Calinski-Harabasz (CH), and Silhouette scores for each label.
    """
    
    print("Computing scores \nDB: lower values --> better clustering")
    print("1/DB, CH, Silhouette: higher values --> better clustering")

    # Determine the data representation
    data_rep = adata.X if use_rep == "X" else adata.obsm[use_rep]

    dict_scores = {}

    # Compute scores for each label and store in dict
    for label in labels:
        db_score = davies_bouldin_score(data_rep, adata.obs[label])
        ch_score = calinski_harabasz_score(data_rep, adata.obs[label])
        # silhouette = silhouette_score(data_rep, adata.obs[label], sample_size=10000) # sample_size parameter added for speed
        silhouette = silhouette_score(data_rep, adata.obs[label]) 

        dict_scores[label] = [db_score, 1/db_score, ch_score, silhouette]

    # Create DataFrame from dict
    df_scores = pd.DataFrame(dict_scores, index=['DB', '1/DB', 'CH', 'Silhouette'])

    return df_scores


# def create_folder(folder_path):
#     pathExist = os.path.exists(folder_path)
#     #if path does not exist create a folder 
#     if not pathExist:
#         print("creating folder:",folder_path)
#         os.makedirs(folder_path)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print("Created folder:", folder_path)
        except Exception as e:
            print(f"Error creating folder {folder_path}: {e}")
    else:
        print("Folder already exists:", folder_path)



def min_max_scaling(x):
    x_max=np.max(x)
    x_min=np.min(x)
    y=(x-x_min)/(x_max-x_min)
    return y

# def read_adata(folder_path):
#     import glob
#     exprMatrix_path = glob.glob(folder_path+'/exprMatrix*.npy')[0]
#     gene_ids_path = glob.glob(folder_path+'/geneids*.csv')[0]
#     meta_path = glob.glob(folder_path+'/meta*.csv')[0]

#     X=np.load(exprMatrix_path)#cells * genes matrix
#     var=pd.read_csv(gene_ids_path)#genes metadata
#     obs=pd.read_csv(meta_path)#cell metadata
#     return X,var,obs




def read_adata(folder_path, issparse=False):
    """
    Reads data from a folder and loads it into memory.

    Parameters:
    folder_path (str): The path to the folder containing the data.
    isdense (bool): A flag indicating if the data is stored in dense format. If True, allows pickling.

    Returns:
    tuple: A tuple containing the expression matrix (X), variable annotations (var), and observation annotations (obs).
    """
    exprMatrix_path = glob.glob(folder_path+'/exprMatrix*.npy')[0]
    gene_ids_path = glob.glob(folder_path+'/geneids*.csv')[0]
    meta_path = glob.glob(folder_path+'/meta*.csv')[0]

    # Load the expression matrix with or without allowing pickle based on isdense flag
    loaded_data = np.load(exprMatrix_path, allow_pickle=issparse)
    
    # If the loaded data is wrapped in a numpy array, extract the sparse matrix
    if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
        X = loaded_data.item()
    else:
        X = loaded_data

    var = pd.read_csv(gene_ids_path)  # genes metadata
    obs = pd.read_csv(meta_path)  # cell metadata
    
    return X, var, obs

def save_adata(adata,output_path):
    create_folder(output_path)   
    np.save(output_path+'/exprMatrix.npy',adata.X)
    adata.var.to_csv(output_path+'/geneids.csv')
    adata.obs.to_csv(output_path+'/meta.csv')
#subset cells
def subset_adata(adata,index_array):
    try:
        obs=copy.deepcopy(adata.obs.loc[index_array].reset_index())
    except:#sometimes it throws an error when level_0 name for index is already in use
        obs=copy.deepcopy(adata.obs.loc[index_array].reset_index(drop=True))

    var=copy.deepcopy(adata.var)
    X=copy.deepcopy(adata[index_array].X)
    adata_sub = anndata.AnnData(X,obs,var)
    return adata_sub
#subset genes
def subset_adata_genes(adata,genes):
    obs=copy.deepcopy(adata.obs)
    var=copy.deepcopy(adata.var.loc[genes,:])
    X=copy.deepcopy(adata[:,genes].X)
    adata_sub = anndata.AnnData(X,obs,var)
    return adata_sub

def get_OHE(adata, categories, col):
    """
    Generates one-hot encoded data for a specified column in an AnnData object.
    If categories are provided, the one-hot encoding will be ordered by these categories.
    If categories is None, the categories are inferred from the data.

    Parameters:
    - adata: AnnData object containing the data.
    - categories (list or None): A list of categories to be used for one-hot encoding, 
                                 ordered as they should appear, or None to infer categories.
    - col (str): The name of the column in `adata.obs` from which to generate 
                 the one-hot encoding.

    Returns:
    - DataFrame: A pandas DataFrame containing the one-hot encoded values.
    """
    import pandas as pd

    data = adata.obs[col].values

    if categories is not None:
        # Convert the data to a categorical type with the defined categories
        data_categorical = pd.Categorical(data, categories=categories, ordered=True)
        one_hot_encoded_data = pd.get_dummies(data_categorical)
    else:
        # Use get_dummies directly if no categories are specified
        one_hot_encoded_data = pd.get_dummies(data)

    return one_hot_encoded_data




def create_splits(data_path,data2split_foldername,savesplits_foldername,save_data = True,random_state_list=[3,4]):
    """ function that creates train, test and val splits and saves them
        Args: 
            data_path (str): path to data parent folder
            data2split_foldername (str): folder with adata object components to split:  '/exprMatrix.npy','/geneids.csv','/meta.csv'
            savesplits_foldername (str): folder to save train, test and val splits
            save_data (bool): if True data will be saved
            random_state_list (list): list of random states (int values)
    """
    #Define if you want to save the data

    #read seen data
    adata2split_path = data_path + "/" + data2split_foldername
    print(adata2split_path)
    X,var,obs = read_adata(adata2split_path)
    adata = anndata.AnnData(X,obs=obs,var=var)
    print("adata shape",adata.shape)

    #split dataset in train, val and test
    index_all = adata.obs.index
    index_train,index_test = train_test_split(index_all, test_size=0.2, random_state=random_state_list[0])
    index_train,index_val = train_test_split(index_train, test_size=0.1, random_state=random_state_list[1])

    #subset data using the train,val and test indexes
    adata_train = subset_adata(adata,index_train)
    adata_val = subset_adata(adata,index_val)
    adata_test = subset_adata(adata,index_test)

    #print the shapes to confirm that the splits were performed correctly
    print("adata_train shape",adata_train.shape)
    print("adata_val shape",adata_val.shape)
    print("adata_test shape",adata_test.shape)

    #change splits out folder name to not rewrite results

    if save_data==True:
        create_folder(data_path+"/"+savesplits_foldername)
        save_adata(adata_train,data_path+"/"+savesplits_foldername+"/train")
        save_adata(adata_val,data_path+"/"+savesplits_foldername+"/val")
        save_adata(adata_test,data_path+"/"+savesplits_foldername+"/test")

def rename_obsm_col(adata,obsm_col,obsm_col_newname):
    """Rename obsm_col from adata object
    Args:
        adata: adata object
        obsm_col (str): name of the obsm col
        obsm_col_new_name (str): new name of the obsm col
    return:
        adata: adata object with the new obs_col_newname

    """
    adata.obsm[obsm_col+'_'+obsm_col_newname] = adata.obsm[obsm_col]
    del adata.obsm[obsm_col]
    return adata

def rename_uns_col(adata,uns_col,uns_col_newname):
    """Rename uns_col from adata object
    Args:
        adata: adata object
        uns_col (str): name of the uns col
        uns_col_new_name (str): new name of the uns col
    return:
        adata: adata object with the new uns_col_newname

    """
    adata.uns[uns_col+'_'+uns_col_newname] = adata.uns[uns_col]
    del adata.uns[uns_col]
    return adata

def rename_varm_col(adata,varm_col,varm_col_newname):
    """Rename varm_col from adata object
    Args:
        adata: adata object
        varm_col (str): name of the varm col
        varm_col_new_name (str): new name of the varm col
    return:
        adata: adata object with the new varm_col_newname

    """
    adata.varm[varm_col+'_'+varm_col_newname] = adata.varm[varm_col]
    del adata.varm[varm_col]
    return adata

def del_pca(adata):
    """Delete pca latent representation from adata object
    Args: 
        adata (anndata object): adata with adata.obsm["X_pca"],adata.uns["pca"],adata.varm["PCs"]
    Returns:
        adata (anndata object): adata without adata.obsm["X_pca"],adata.uns["pca"],adata.varm["PCs"]

    """
    del adata.obsm["X_pca"]
    del adata.uns["pca"]
    del adata.varm["PCs"]
    return adata

def get_diff_exp_genes(adata,clustering = "cluster", alpha = 0.05, output_folder="",save_data = False):
    """ This function was adapted from Genevieve Konopka Genomics course imparted in UTSW in Feb 2023
        This function can only be applied after generating adata.uns.rank_genes_groups with sc.tl.rank_genes_groups(adata_train, groupby="cluster", method="wilcoxon")
        It is useful to build a df with differentially expressed genes per cluster label clust
        Args: 
            adata (anndata object)
            clustering (str): name of the cluster labels column
            alpha (float): alpha criterion for statistical test
        Return:
            df (pandas df): df with differentially expressed genes per cluster
    """
    #create empty df with columns to extract
    column_names = ["genes","pval","scores",clustering]
    df = pd.DataFrame(columns = column_names)
    #for each cluster value in the clustering labels
    for clust in np.unique(adata.obs[clustering]):
        #get indexes for adjusted p vals < alpha
        filtered_indexes = adata.uns['rank_genes_groups']['pvals_adj'][clust]< alpha
        
        #get genes, p vals and scores for p vals < alpha
        genes = adata.uns["rank_genes_groups"]["names"][clust][filtered_indexes]
        pvals = adata.uns["rank_genes_groups"]["pvals"][clust][filtered_indexes]
        scores = adata.uns["rank_genes_groups"]["scores"][clust][filtered_indexes]
        
        #create df with significantly different genes, p val, scores
        n = len(genes)
        df_clust = pd.DataFrame({"genes":genes,"pval":pvals,"scores":scores,clustering:[clust]*n})
        df = df.append(df_clust)
    #save df
    if save_data == True:
        df.to_csv(output_folder+"/diff_exp_genes_"+clustering+".csv")
    return df


def create_figure_folders(plot_type,out_subfolder_name,obs_col=""):
    """ function to create folders to store scanpy.pl figures in a subfolder of the form /figures/plot_type/outsubfolder_name
    Args:
        plot_type (str): type pf plot. Options: tsne, pca, rank_genes_groups, matrixplot_, stacked_violin
        out_subfolder_name (str): Any name you want to assign to the out folder
        obs_col (str): any columsn from adata.obs.obs_col that your are interested to plot
    """
    #create figures folder
    create_folder("figures")
    #creates /figures/plot_type folder
    if len(obs_col)>0: #if the obs_col option different from default(default = "")
        create_folder("figures/"+plot_type+"_"+obs_col)
        figures_folder = "figures/"+plot_type+"_"+obs_col+"/"+out_subfolder_name
    else:
        create_folder("figures/"+plot_type)        
        figures_folder = "figures/"+plot_type+"/"+out_subfolder_name
    # creates /figures/plot_type/outsubfolder_name
    create_folder(figures_folder)

def get_colors_dict(celltype, donor,colors_list=['olive','darkolivegreen','springgreen','lightseagreen'],color_map ="combined"):
        """Creates a dict that assigns a celltype-donor key to a color value
        Args:
            colors_list (list): list of color names
            color_map (std): string that defines if the colormap is defined by "donor","celltype" or "combined"
        Returns:
            colors_dict (dict): dictionary con keys with a celltype-donor label and color values 
        """
        colors_dict = {}
        if color_map == "donor":
            # Create a dictionary that maps each donor to a color
            color_map = {d: color for d, color in zip(np.unique(donor), colors_list)}
            
            # Create a colors_dict using celltype and donor info
            for ct, d in zip(celltype, donor):
                key = f'celltype-{ct}_donor-{d}'
                colors_dict[key] = color_map[d]
        if color_map == "celltype":
            # Create a dictionary that maps each donor to a color
            color_map = {ct: color for ct, color in zip(np.unique(celltype), colors_list)}
            
            # Create a colors_dict using celltype and donor info
            for ct, d in zip(celltype, donor):
                key = f'celltype-{ct}_donor-{d}'
                colors_dict[key] = color_map[ct]
        if color_map == "combined":
            for i,(ct, d) in enumerate(zip(celltype, donor)):
                key = f'celltype-{ct}_donor-{d}'
                colors_dict[key] = colors_list[i]

            
        return colors_dict
###############################################################################Functions for splatter/SplatPop
def get_dropout_percent(adata, dropout_target ="cells",return_means =True):
    """ Function that calculates t"""  """he dropout for cells/genes
    Args:
        adata: adata object
        dropout_target: str (options "cells", "genes")
    Return:
        dropout: np array with droout percentages. Default:cell_dropout (options: gene dropout)
        cell_means: np.array. Default: cell_means array 
    """
    #get X dims cells* genes
    n_genes = adata.shape[1]
    n_cells = adata.shape[0]
    print("adata shape",adata.X.shape)
    # Drop cells with >95% zeros
    if dropout_target=="cells":
        #calculate cells dropout/ percentage of zeros per cell
        cell_dropout = np.sum(adata.X == 0, axis=1)
        cell_dropout = cell_dropout/n_genes
        cell_means = np.mean(adata.X, axis=0)
        if return_means==True:
            return cell_dropout, cell_means
        else:
            return cell_dropout
    elif dropout_target=="genes":
        #gene dropout: percentage of zeros per gene
        gene_dropout = np.sum(adata.X == 0, axis=0)
        gene_dropout = gene_dropout/n_cells
        # Calculate gene means 
        gene_means = np.mean(adata.X, axis=0)
        if return_means==True:
            return gene_dropout, gene_means
        else:
            return gene_dropout


def formatdata_splatter(adata, donor_col ="individual",celltype_col ="cluster"):
    """Formats data to be read by Splatter: Splat and SplatPop. It adds a Sample_ind col in which the donor_ids are renamed as Sample_i, a Group col where celltypes are renamed as Groupj
        Its needs to be run before get_cells_sample_morecounts,get_bulk
    Args:
        adata adata (anndata object)
        donor_col (str): name of the donor col. It changes depending on the dataset. Default="individual" (Autism V19 dataset)
        celltype_col (str): name of the celltype col. It changes depending on the dataset. Default="cluster" (Autism V19 dataset)
    Returns: adata (anndata object): adata with Sample_ind,Group new columns
    """
    #Format data

    # Add a Sample_ind column equivalent to individual column. This is to match the format from Splatter mock_bulk matrix
    ind_names = np.unique(adata.obs[donor_col]).tolist()
    n_ind = len(ind_names)
    sample_names = ["Sample_"+str(i+1) for i in range(n_ind)]
    #Add new column "Sample_ind"
    adata.obs["Sample_ind"] = np.nan
    #Fill the Sample_ind column according to the individual column
    for ind,sample in zip(ind_names, sample_names):
        adata.obs.loc[adata.obs[donor_col]==ind,"Sample_ind"] = sample

    # Add a Group column equivalent to cluster/celltype column. This is to match the format from Splatter 
    celltype_names = np.unique(adata.obs[celltype_col]).tolist()
    n_groups = len(celltype_names)
    group_names = ["Group"+str(i+1) for i in range(n_groups)]
    #Fill the Sample_ind column according to the individual column
    for cell,group in zip(celltype_names, group_names):
        adata.obs.loc[adata.obs[celltype_col]==cell,"Group"] = group
    return adata
    
def get_bulk(adata,out_folder,sample_col = "Sample_ind",del_zerogenes=True, save_files=True):
    """Returns aggregated bulk matrix from single cell data
    Args:
        adata: adata object with single cell RNA seq data
        out_folder (str): path to out folder
        sample_col (str): column name of the sample. Default:"Sample_ind"
        del_zerogenes (bool): if True removes genes with bulk gene mean = 0
        save_files (bool): if True saves bulk dataframe to out_folder path
    Return: 
        bulk_out (pandas df): Aggregated bulk matrix from single cell data
    """
        #5. Get the bulk_matrix based on the Sample ind column
    # convert your AnnData matrix to a DataFrame
    df = pd.DataFrame(data = adata.X, 
                    index = adata.obs_names,
                    columns = adata.var_names)
    #Add sample col
    df[sample_col] = adata.obs[sample_col].values

    # Create bulk_df by grouping by 'individual' and calculate the mean (or sum)
    bulk_df = df.groupby([sample_col]).mean()

    #I got an error with splatpop. I need to remove zero genes from bulk data
    if del_zerogenes==True:  
        bulk_means=bulk_df.mean(axis=0)
        #identify non zero genes
        zero_genes = bulk_means[bulk_means==0].index
        non_zero_genes = [ i for i in bulk_means.index if i not in zero_genes]
        #select non zero genes
        bulk_out = bulk_df.loc[:,non_zero_genes]
    else:
        bulk_out = bulk_df

    #save
    if save_files ==True:
        bulk_out.to_csv(out_folder+"/bulk_matrix.csv")
    
    #Return bulk_out,non_zero_genes if they were computed and if they are at least 1. Otherwise return bulk_out only
    try:
        if len(non_zero_genes)>0:
                return bulk_out,non_zero_genes
        else:
            return bulk_out

    except:
        print("Non zero genes were not computed and removed from bulk data. Set del_non_zero_genes = True to have them removed")
        return bulk_out


# 6. get cells from individual/Sample_ind with more cell counts
def get_cells_sample_morecounts(adata,sample_col = "Sample_ind",cell_col ="cell"):
    """Returns adata subset from the sample with more cell counts
    Args:
        adata (anndata object)
        sample col (str): Name of the sample/ donor/ individual columns
        cell col (str): Name of the column that has cell names
    Returns:
        adata_sample (anndata object): contains data of the sample with more cells
        sample_morecells (str): name of the sample with more cells
    """
    #get cell counts per sample 
    adata.obs[[cell_col,sample_col]].count()
    ind_cell_counts = adata.obs.groupby(['Sample_ind'])[cell_col].count().reset_index()
    #get ind/Sample_ind with more cells
    sample_morecells = ind_cell_counts.loc[ind_cell_counts[cell_col]==np.max(ind_cell_counts["cell"]),"Sample_ind"].values[0]
    print("Sample_ind with more cells:", sample_morecells)
    #get indexes from individual with more cells
    sample_morecells_indexes = adata.obs.loc[adata.obs["Sample_ind"]==sample_morecells,:].index
    #get single cell counts for a single individual
    adata_sample = subset_adata(adata,sample_morecells_indexes)
    return adata_sample,sample_morecells

######################################## SplatPop simulation utils
def readingHPOsim_format(param_df ,param_col = "similarity.scale"):
    """Set up the params to the right format for reading a HPO splatPop simulation. Simulations are stored in folders with the names and values of the HPO parameters. 
    If parameter is an int --> the format is str(int), if parameter is a float -->the format is str(float)
    Args:
        param_df(pd.DataFrame): df with parameters columns of float type. Note: The param_df is usually stored as HPO_grid.csv or as part HPO_scores.csv
        param_col (str): column name of the parameters. Options = "similarity.scale","de.facLoc"
    Returns:
        param_list (list): list with parameters in the right format for reading a HPO splatpop simulation
    """
    # separate the fractional and integral parts
    frac, integral = np.modf(param_df[param_col]) 
    # use np.where to modify the array
    param_array = np.where(frac == 0, integral.values, param_df[param_col])
    # params list
    param_list  = [str(int(item)) if item.is_integer() else str(item) for item in param_array]

    return param_list

def read_simadata(sim_path,sim_scale,defacLoc ):
    """Read synthetic simulation from SplatPop. The simulation outputs are saved with an output path that specifies the parameters
    Args:
        sim_path (str): simulation path
        sim_scale: (str) similarity scale parameter
        defacLoc: (str): de factor location paramater
    Return:
        adata: adata object cells*genes
    """
    sim_folderpath = sim_path+"/*-"+sim_scale+"*-"+defacLoc
    cm_path = glob.glob(sim_folderpath +"*/count_matrix.csv")[0]
    obs_path = glob.glob(sim_folderpath +"*/meta_sim*")[0]
    var_path = glob.glob(sim_folderpath +"*/means_key*")[0]

    X = pd.read_csv(cm_path, index_col = 0).values.T
    obs = pd.read_csv(obs_path , index_col = 0)
    var = pd.read_csv(var_path , index_col = 0)
    adata = anndata.AnnData(X,obs=obs,var=var)
    return adata




# def plot_rep(adata, shape_col="celltype",color_col="donor",use_rep ="X_pca",markers = ['o', 'v', '^', '<','*'],clustering_scores = None, save_fig = True, outpath = "", showplot=False):
#     """plots a dimensionaly reduced representation of adata
#     Args:
#         adata (anndata object): cells*genes
#         shape_col (str): adata.obs column to be visualized with different marker shapes. Default: "celltype"
#         color_col (str): adata.obs column to be visualized with different color. Default: "donor"
#         use_rep (str): adata.obsm dimensionally reduced representation of adata.X. Dimensions: cells * n components. Default: "X_pca"
#         markers (list): list of markers to be plotted. Default: markers = ['o', 'v', '^', '<','*']
#         clustering_scores (pd.DataFrame with clustering scores for shape_col and color_col): If clustering_scores is not None: Adds the clustering scores of both columns to the title
#         save_fig (bool): If True: save figure
#         outpath (str): output path of the saved figure. Default = ""
#     """
#     print("plotting latent representation:",use_rep)
#     import itertools
#     from matplotlib.lines import Line2D
#     #plot created with help from chat gpt 4
#     plt.ioff() 
#     fig, ax = plt.subplots(figsize = (7,7))
    

    
#     marker_cycler = itertools.cycle(markers)
#     color_cycler = itertools.cycle(plt.cm.tab20.colors)

#     # Get unique elements from  shape and color cols
#     unique_shapecol = np.unique(adata.obs[shape_col])
#     unique_colorcol = np.unique(adata.obs[color_col])

#     # Assign a color to each element of unique_colorcol and a marker to each unique_shapecol
#     colors = {color: next(color_cycler) for color in unique_colorcol}
#     shapes = {shape: next(marker_cycler) for shape in unique_shapecol}

#     #get first 2 components
#     print("latent shape:",adata.obsm[use_rep].shape)
#     c1 = adata.obsm[use_rep][:,0]
#     c2 = adata.obsm[use_rep][:,1]
#     # Create scatter plots for each point of pc1,pc2
#     for shape in unique_shapecol:
#         for color in unique_colorcol:
#             mask = (adata.obs[color_col] == color) & (adata.obs[shape_col] == shape)
#             ax.scatter(c1[mask], c2[mask], color=colors[color], marker=shapes[shape], alpha = 0.5)



#     # Create legend for colors 
#     color_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[c], markersize=10) for c in colors]

#     # Create legend for markers 

#     marker_legend_elements = [Line2D([0], [0], marker=shapes[s], color='black', markerfacecolor='black', markersize=10) for s in shapes]

#     # Add legends to the plot
#     legend1 = ax.legend(handles=color_legend_elements, labels=list(colors.keys()), loc='upper left', bbox_to_anchor=(1,1), title=color_col)
#     plt.gca().add_artist(legend1)  # Add the legend back after it gets overwritten

#     legend2 = ax.legend(handles=marker_legend_elements, labels=list(shapes.keys()), loc='lower center', bbox_to_anchor=(0.5,-0.3),ncol=5, title=shape_col)
    
#     #add clustering scores to the plot
#     if clustering_scores is not None:
#         #calculate scores on the PCA space
#         #df = get_clustering_scores(adata,use_rep = "X_pca", labels = [color_col,shape_col])
#         df = clustering_scores
#         print("Warning: All clustering scores should have been calculated in PCA latent space")
#         df = np.round(df,2)
#         #get lists of clustering scores
#         color_scores_list = [key+":" +str(value) for key, value in df[color_col].items()]
#         shape_scores_list = [key+":" +str(value) for key, value in df[shape_col].items()]
#         #get titles with clustering scores
#         color_title = color_col+" scores - "+' '.join(color_scores_list[1:])
#         shape_title = shape_col+" scores - "+' '.join(shape_scores_list[1:])
#         plt.title(color_title+"\n"+shape_title+"\nscores calculated on PCA latent space")

#     #if use_rep is pca -->add variance ratio to the plot
#     try:
#         #print("trying pca")
#         use_rep.index("pca")
#         #ncomps = int(use_rep.split("X_pca_ncomps")[1])
#         variance_ratio_pc1 = np.round(adata.uns['pca']["variance_ratio"][0]*100,3)
#         variance_ratio_pc2 = np.round(adata.uns['pca']["variance_ratio"][1]*100,3)
#         plt.xlabel("PC 1 ("+str(variance_ratio_pc1)+"%)")
#         plt.ylabel("PC 2("+str(variance_ratio_pc2)+"%)")
            
#     except:
#         plt.xlabel(use_rep+" 1")
#         plt.ylabel(use_rep+" 2")

#     #save fig
#     if (save_fig == True)& (len(outpath)>0):
#         fig.savefig(outpath+"/"+use_rep+"_python_version.png", bbox_extra_artists=(legend1,legend2), bbox_inches='tight')
#     if showplot==True:
#         plt.show()
#     else:
#         plt.close("all")


def plot_rep(adata, shape_col="celltype", color_col="donor", use_rep="X_pca", markers=['o', 'v', '^', '<', '*'], clustering_scores=None, save_fig=True, outpath="", showplot=False, palette_choice="tab20",file_name="latent"):
    """Plots a dimensionally reduced representation of adata."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import itertools
    from matplotlib.lines import Line2D

    print("plotting latent representation:", use_rep)
    plt.ioff()
    fig, ax = plt.subplots(figsize=(7, 7))

    unique_shapecol = np.unique(adata.obs[shape_col])
    unique_colorcol = np.unique(adata.obs[color_col])

    # Choose the color palette based on the palette_choice argument
    if palette_choice == "hsv":
        color_palette = sns.color_palette("hsv", len(unique_colorcol))
    elif palette_choice == "tab20":
        # Ensure tab20 has enough colors, otherwise cycle through them
        color_palette = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(unique_colorcol))]
    elif palette_choice == "Set2":
        color_palette = sns.color_palette("Set2", len(unique_colorcol))
    else:
        raise ValueError("Invalid palette choice. Please choose 'hsv' or 'tab20'.")

    color_map = {color: color_palette[i] for i, color in enumerate(unique_colorcol)}
    shape_map = {shape: markers[i % len(markers)] for i, shape in enumerate(unique_shapecol)}

    c1 = adata.obsm[use_rep][:, 0]
    c2 = adata.obsm[use_rep][:, 1]

    for shape in unique_shapecol:
        for color in unique_colorcol:
            mask = (adata.obs[color_col] == color) & (adata.obs[shape_col] == shape)
            ax.scatter(c1[mask], c2[mask], color=color_map[color], marker=shape_map[shape], alpha=0.25, s=1)

    # Create legends
    color_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[c], markersize=10) for c in unique_colorcol]
    shape_legend_elements = [Line2D([0], [0], marker=shape_map[s], color='black', markerfacecolor='black', markersize=10) for s in unique_shapecol]

    legend1 = ax.legend(handles=color_legend_elements, labels=list(unique_colorcol), loc='upper left', bbox_to_anchor=(1, 1), title=color_col)
    plt.gca().add_artist(legend1)

    legend2 = ax.legend(handles=shape_legend_elements, labels=list(unique_shapecol), loc='lower center', bbox_to_anchor=(0.5, -0.65), ncol=3, title=shape_col)
        #add clustering scores to the plot
    if clustering_scores is not None:
        #calculate scores on the PCA space
        #df = get_clustering_scores(adata,use_rep = "X_pca", labels = [color_col,shape_col])
        df = clustering_scores
        print("Warning: All clustering scores should have been calculated in PCA latent space")
        df = np.round(df,2)
        #get lists of clustering scores
        color_scores_list = [key+":" +str(value) for key, value in df[color_col].items()]
        shape_scores_list = [key+":" +str(value) for key, value in df[shape_col].items()]
        #get titles with clustering scores
        color_title = color_col+" scores - "+' '.join(color_scores_list[1:])
        shape_title = shape_col+" scores - "+' '.join(shape_scores_list[1:])
        plt.title(color_title+"\n"+shape_title+"\nscores calculated on PCA latent space")

    #if use_rep is pca -->add variance ratio to the plot
    try:
        #print("trying pca")
        use_rep.index("pca")
        #ncomps = int(use_rep.split("X_pca_ncomps")[1])
        variance_ratio_pc1 = np.round(adata.uns['pca']["variance_ratio"][0]*100,3)
        variance_ratio_pc2 = np.round(adata.uns['pca']["variance_ratio"][1]*100,3)
        plt.xlabel("PC 1 ("+str(variance_ratio_pc1)+"%)")
        plt.ylabel("PC 2("+str(variance_ratio_pc2)+"%)")
            
    except:
        plt.xlabel(use_rep+" 1")
        plt.ylabel(use_rep+" 2")
    if save_fig and outpath:
        fig.savefig(f"{outpath}/{use_rep}_{file_name}.png", bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
    if showplot:
        plt.show()
    else:
        plt.close("all")

    






######################################## For comparing models
# Docstring written with chatgpt4
def compute_kmeans_acc(n_clusters, x, x_test, y_test):
    """
    Computes the accuracy of KMeans clustering on a given dataset and compares it 
    with provided labels.

    Parameters:
    -----------
    n_clusters : int
        The number of clusters to be formed using KMeans.
    
    x : array-like or pd.DataFrame, shape (n_samples, n_features)
        Training data to fit KMeans.
    
    x_test : array-like or pd.DataFrame, shape (n_samples, n_features)
        Test data to predict clusters using the trained KMeans model.

    y_test : array-like or pd.DataFrame, shape (n_samples,)
        True labels for the test data. Can be one-hot encoded or class labels.

    Returns:
    --------
    kmeans_accuracy : float
        Accuracy of KMeans clustering on y_test labels.

    Notes:
    ------
    - Assumes that the number of clusters in y_test matches n_clusters.
    - If y_test is one-hot encoded, it will be converted to class labels internally.

    Examples:
    ---------
    >>> x = [[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]]
    >>> x_test = [[1, 2], [5, 4], [3.5, 1.8], [8, 7]]
    >>> y_test = [1, 0, 1, 0]
    >>> compute_kmeans_acc(2, x, x_test, y_test)
    """
    from sklearn.metrics import accuracy_score
    from sklearn.cluster import KMeans
    
    # Use KMeans clustering on x (latent space)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(x)
    y_test_pred = kmeans.predict(x_test)

    # Calculate accuracy of KMeans on y (labels)
    if len(y_test.shape) > 1:  # if hot encoded: convert to index
        kmeans_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_test_pred)
    else:
        kmeans_accuracy = accuracy_score(y_test, y_test_pred)

    return kmeans_accuracy



# def calculate_merge_scores(latent_list, adata, labels: list):
#     """
#     Calculates clustering scores for multiple latent representations using the provided labels,
#     and then aggregates them into a single dataframe.

#     Parameters:
#     - latent_list (list): List of latent representations for which clustering scores need to be calculated.
#     - adata (anndata object): The annotated data matrix object containing cells, genes, and other annotations.
#     - labels (list): List of label columns in the adata object to be used for calculating clustering scores.  Example: ["donor","celltype"]

#     Returns:
#     - combined_df (pd.DataFrame): Dataframe containing the aggregated clustering scores for all the provided latent representations,
#                                   indexed by the names of the latent representations.
#     """
#     #put all scores in single df
#     for i,lat in enumerate(latent_list):
        
            
#         # scores = get_clustering_scores(adata,lat, labels)
#         scores = get_clustering_scores_optimized(adata,lat, labels)
#         scores_row = restructure_dataframe(scores,labels)
#         if i == 0:
#             combined_df = scores_row 
#         else:
#             combined_df = pd.concat([combined_df, scores_row], ignore_index=True)
#     combined_df.index = latent_list
#     return combined_df
#     import pandas as pd

def calculate_merge_scores(latent_list, adata, labels):
    """
    Calculates clustering scores for multiple latent representations using the provided labels,
    and then aggregates them into a single DataFrame.
    """
    # Initialize a list to hold the DataFrame rows before concatenation
    scores_list = []

    # Iterate over each latent representation and calculate clustering scores
    for latent in latent_list:
        # Assuming get_clustering_scores_optimized returns a DataFrame with the scores
        scores = get_clustering_scores(adata, latent, labels)
        
        # Assuming restructure_dataframe restructures the scores DataFrame as needed for aggregation
        scores_row = restructure_dataframe(scores, labels)
        
        # Append the restructured row to the list
        scores_list.append(scores_row)

    # Concatenate all DataFrame rows at once outside the loop
    combined_df = pd.concat(scores_list, ignore_index=True)

    # Assign the latent representation names as the DataFrame index
    combined_df.index = latent_list

    return combined_df


def restructure_dataframe(df, labels):
    """
    Restructures a given dataframe based on specific clustering scores and labels, creating a multi-level column format.
    It is useful to restructure the output of calculate_merge_scores.

    Parameters:
    - df (pd.DataFrame): Input dataframe that contains clustering scores.
    - labels (list): List of label columns (e.g., 'donor', 'celltype') that will be used to reorder and index the dataframe.

    Returns:
    - new_df (pd.DataFrame): Restructured dataframe with multi-level columns based on the unique index names and original columns.
                             The dataframe contains clustering scores for the specified labels.
    
    Notes:
    The function specifically looks for the clustering scores "ch", "1/db", and "silhouette" in the input dataframe.
    """

    # reorder
    df = df.loc[["ch","1/db","silhouette"],labels].T
    # Get the unique index names (donor, celltype, etc.)
    index_names = df.index.unique().tolist()

    # Extracting the column names
    cols = df.columns.tolist()

    # Creating multi-level column names based on index_names and cols
    new_columns = pd.MultiIndex.from_product([index_names, cols])

    # Flatten the data for the new structure
    values = df.values.flatten()

    # Creating the new DataFrame
    new_df = pd.DataFrame([values], columns=new_columns)
    return new_df



def plot_table(df, out_name, model_path,showplot=False):
    """
    Plots a dataframe as a table in a figure and saves it to a specified path. It is useful to plot a clustering scores df

    Parameters:
    - df (pd.DataFrame): The dataframe to be plotted as a table.
    - out_name (str): The suffix to be added to the filename when saving the figure.
    - model_path (str): The directory path where the figure should be saved.
    - showplot (bool): show plot if True.

    Returns:
    None. The function saves the figure to the specified directory and displays it.
    """
    fig, ax = plt.subplots(figsize=(12, 4))  # set the size that you'd like (width, height)

    # Hide axes
    ax.axis('off')

    # Plot table
    tbl = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')

    # Make the cells larger to fit the text
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))

    # Save the figure
    plt.savefig(model_path+"/table_scores"+out_name+".png", dpi=300, bbox_inches='tight')
    if showplot==True:
        plt.show()
    else:
        plt.close("all")




# def get_split_paths(base_path, fold):
#     """
#     Generates the paths for the train, validation, and test data directories based on a base path and fold number.

#     Parameters:
#     - base_path (str): The base path where the split directories are located.
#     - fold (int): The specific fold number to construct the paths for.

#     Returns:
#     - tuple: A tuple containing the paths for the training, val, and test directories.
#     """
#     split_base = f'{base_path}/split_{fold}'
#     train_path = os.path.join(split_base, 'train')
#     test_path = os.path.join(split_base, 'test')
#     val_path = os.path.join(split_base, 'val')

#     return train_path, val_path, test_path

def get_split_paths(base_path, fold):
    """
    Generates the paths for the train, validation, and test data directories based on a base path and fold number.

    Parameters:
    - base_path (str): The base path where the split directories are located.
    - fold (int): The specific fold number to construct the paths for.

    Returns:
    - paths_dict: A dictionary containing the paths for the training, val, and test directories.
    """
    split_base = f'{base_path}/split_{fold}'
    paths_dict = {
        'train': os.path.join(split_base, 'train'),
        'val': os.path.join(split_base, 'val'),
        'test': os.path.join(split_base, 'test')
    }

    return paths_dict
import seaborn as sns


class ExpressionAnalysis_TopDiffGenes:
    """
    A class for analyzing expression of top differentially expressed genes in single-cell RNA-seq data.

    Attributes:
    - adata (AnnData): An AnnData object containing single-cell RNA-seq data.

    Methods:
    - extract_diff_expr_results: Extracts differential expression results into a DataFrame.
    - process_top_diff_genes: Processes top differentially expressed genes and returns a subset of the data.
    - analyze_and_plot_mean_expression: Analyzes and plots the mean expression of genes.
    - plot_onecell_allgenes_expression_histograms: Plots expression histograms for all genes in specified cells.
    - compute_gene_CR_stats: Computes Coefficient of Variation statistics for genes.
    - plot_allcells_onegene_expression_histograms: Plots histograms of gene expression across groups for specified genes.

    Requirenments: 
    1. Preprocessing step: Log transform data ln(x+1), I need it for computing stats.
    sc.pp.log1p(adata)

    2. Perform differential expression analysis (expects logarithmized data)
    sc.tl.rank_genes_groups(adata, groupby='Capbatch', method='wilcoxon')
    """

    def __init__(self, adata):
        self.adata = adata

    def extract_diff_expr_results(self, keys_to_extract, obs_key='Capbatch',rank_genes_groups='rank_genes_groups'):
        """
        Extracts differential expression results for specified keys and groups them by an observation key.

        Parameters:
        - keys_to_extract (list of str): List of keys to extract from differential expression results.
        - obs_key (str): Observation key to group results by (default: 'Capbatch').

        Returns:
        - DataFrame: A DataFrame containing the extracted differential expression results.
        """
        diff_expr_results = self.adata.uns[rank_genes_groups]
        df = pd.DataFrame()
        for group in self.adata.obs[obs_key].cat.categories:
            group_data = {key: diff_expr_results[key][group] for key in diff_expr_results.keys() if key in keys_to_extract}
            group_df = pd.DataFrame(group_data)
            group_df['group'] = group
            df = pd.concat([df, group_df], ignore_index=True)
        return df

    def process_top_diff_genes(self, keys_to_extract, obs_key='Capbatch', n_top_genes=100,rank_genes_groups='rank_genes_groups'):
        """
        Extracts differential expression results for specified keys and groups them by an observation key.

        Parameters:
        - keys_to_extract (list of str): List of keys to extract from differential expression results.
        - obs_key (str): Observation key to group results by (default: 'Capbatch').

        Returns:
        - DataFrame: A DataFrame containing the extracted differential expression results.
        """
        diff_expr_results = self.adata.uns[rank_genes_groups]
        all_top_genes_df = pd.DataFrame()
        for group in self.adata.obs[obs_key].cat.categories:
            group_data = {key: diff_expr_results[key][group] for key in diff_expr_results.keys() if key in keys_to_extract}
            group_df = pd.DataFrame(group_data)
            group_df_sorted = group_df.sort_values(by=['pvals_adj', 'logfoldchanges'], ascending=[True, False], key=lambda col: col.abs() if col.name == 'logfoldchanges' else col)
            top_genes = group_df_sorted.head(n_top_genes)
            top_genes['group'] = group
            all_top_genes_df = pd.concat([all_top_genes_df, top_genes], ignore_index=True)
        top_diff_exp_genes = np.unique(all_top_genes_df["names"])
        self.adata.var['is_top_diff_exp_gene'] = False
        self.adata.var['group'] = None
        for gene in top_diff_exp_genes:
            if gene in self.adata.var_names:
                self.adata.var.at[gene, 'is_top_diff_exp_gene'] = True
                gene_rows = all_top_genes_df[all_top_genes_df['names'] == gene]
                groups = gene_rows['group'].unique()
                self.adata.var.at[gene, 'group'] = ', '.join(groups)
        adata_subset = self.adata[:, top_diff_exp_genes]
        return adata_subset, all_top_genes_df

    def analyze_and_plot_mean_expression(self, adata_subset, obs_key='Capbatch'):
        """
        Analyzes and plots the mean expression of genes across different groups.

        Parameters:
        - adata_subset (AnnData): Subset of the AnnData object to analyze.
        - obs_key (str): Observation key to group data by (default: 'Capbatch').

        Returns:
        - dict: A dictionary containing statistical information about the mean expression of genes in each group.
        """

        group_mean_expr_stats = {}
        groups = adata_subset.obs[obs_key].cat.categories
        for group in groups:
            group_mean_expression = np.array(adata_subset.X[adata_subset.obs[obs_key] == group, :].mean(axis=1).flatten())
            group_mean_expr_stats[group] = {
                "mean_gene_expression": group_mean_expression,
                "mean": np.mean(group_mean_expression),
                "std": np.std(group_mean_expression)
            }
        means_of_group_means = [group_mean_expr_stats[group]["mean"] for group in group_mean_expr_stats]
        stds_of_group_means = [group_mean_expr_stats[group]["std"] for group in group_mean_expr_stats]
        overall_mean_of_group_means = np.mean(means_of_group_means)
        overall_std_of_group_means = np.std(means_of_group_means)
        mean_of_stds = np.mean(stds_of_group_means)
        CR = overall_std_of_group_means / mean_of_stds
        group_mean_expr_stats['overall'] = {
            "mean_of_means": overall_mean_of_group_means,
            "std_of_means": overall_std_of_group_means,
            "mean_of_stds": mean_of_stds,
            "CR": CR
        }
        plt.figure(figsize=(8, 6))
        sns.histplot(means_of_group_means, color='skyblue', edgecolor='black', kde=True, bins=20)
        plt.title("Histogram of Mean of Group Means")
        plt.xlabel("Mean Gene Expression")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
        plt.figure(figsize=(10, 8))
        colors = sns.color_palette('hsv', len(groups))
        for i, group in enumerate(groups):
            sns.kdeplot(group_mean_expr_stats[group]["mean_gene_expression"], color=colors[i], label=group)
        plt.title("Mean Expression Histograms for Top 100 Diff Expressed Genes of Each Group")
        plt.xlabel("Expression Level")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
        return group_mean_expr_stats

    def plot_onecell_allgenes_expression_histograms(self,adata, num_cells=11):
        """
        Plot expression histograms for the first 'num_cells' cells in adata.

        Parameters:
        - adata: AnnData object containing the subset of data for plotting.
        - num_cells: Number of cells to plot (default: 11).
        """

        # Check the shape of adata_subset to ensure correct indexing
        print("adata shape:", adata.shape)

        # Create a single plot for all histograms
        plt.figure(figsize=(10, 8))

        # Define a color palette with a distinct color for each cell
        colors = sns.color_palette('hsv', num_cells)

        # Loop to overlay histograms for each cell with a different color
        for i in range(num_cells):
            sns.histplot(adata.X[i, :].flatten(), kde=True, color=colors[i], label=f"Cell {i+1}")

        plt.title(f"Expression Histograms for First {num_cells} Cells")
        plt.xlabel("Expression Level")
        plt.ylabel("Frequency")
        plt.legend()  # Show legend
        plt.show()
    def calculate_and_sort_R_dataframe(self,df, column_A, column_B):
        # Filter rows where A > B
        valid_df = df[df[column_A] > df[column_B]]

        # Calculate R for each valid row
        valid_df['R'] = valid_df[column_A] / valid_df[column_B]

        # Sort the DataFrame by the R value in descending order
        sorted_df = valid_df.sort_values(by='R', ascending=False)

        return sorted_df
    
    def compute_gene_CR_stats(self,adata_subset, obs_key='Capbatch', num_genes=10,out_path = None,file_name=None):
        """
        Compute the Coefficient of Variation (CR_v1 and CR_v2) for genes in an AnnData object.

        Parameters:
        - adata_subset: AnnData object containing the data.
        - obs_key: Key to use for categorizing data (default: 'Capbatch').
        - num_genes: Number of genes to compute CR stats for (default: 10).
        - out_path: Directory where the plot will be saved (optional).
        - file_name: Name of the file to save the plot (optional).

        Returns:
        - DataFrame containing gene names and their CR_v2 values, sorted by CR_v2.
        """

        # Retrieve the categories (groups) from the observation key
        groups = adata_subset.obs[obs_key].cat.categories

        # Initialize a dictionary to store gene expression statistics
        gene_expr_stats = {}

        # Prepare a list to store data for DataFrame creation
        data = []

        # Loop through the specified number of genes
        for gene_name in adata_subset.var.index[:num_genes]:
            # Check if the gene is in the variable names of the subset
            if gene_name in adata_subset.var_names:
                # Get the index of the gene
                gene_index = np.where(adata_subset.var_names == gene_name)[0][0]

                # Initialize a sub-dictionary for this gene
                gene_expr_stats[gene_name] = {}

                # Extract the data for this gene across all groups
                overall_gene_data = np.array(adata_subset.X[:, gene_index])

                # Calculate the overall standard deviation for this gene
                overall_std = np.std(overall_gene_data)

                # Lists to store group-specific means and standard deviations
                stds = []
                means = []
                # Create dict for a dataframe with all gene stats
                gene_dict = {'gene_name': gene_name}

                # Loop through each group to calculate statistics
                for group in groups:
                    # Extract the data for this gene in the current group
                    group_data = np.array(adata_subset.X[adata_subset.obs[obs_key] == group, gene_index])

                    # Calculate mean and standard deviation for the group
                    group_mean = np.mean(group_data)
                    group_std = np.std(group_data)

                    # Add the statistics to the lists
                    stds.append(group_std)
                    means.append(group_mean)

                    # Store group-specific stats in the gene's dictionary
                    gene_expr_stats[gene_name][group] = {"mean": group_mean, "std": group_std}
                    
                    # Add group-specific mean and std to the gene dictionary
                    gene_dict[f'{group}_mean'] = group_mean
                    gene_dict[f'{group}_std'] = group_std
                # Calculate the mean of standard deviations and standard deviation of means
                mean_of_stds = np.mean(stds)
                std_of_means = np.std(means)

                # Calculate CR_v1 and CR_v2
                # overall variability of a gene's expression/ Mean variability of gene expression within each group
                CR_v1 = overall_std / mean_of_stds if mean_of_stds != 0 else np.nan
                # Variability between groups mean expression / Mean variability of gene expression within each group
                CR_v2 = std_of_means / mean_of_stds if mean_of_stds != 0 else np.nan

                # Store these values in the gene's dictionary
                gene_expr_stats[gene_name]["CR_v1"] = CR_v1
                gene_expr_stats[gene_name]["CR_v2"] = CR_v2

                # add CR info to the gene_dict for the dataframe
                gene_dict['CR'] = CR_v2
                gene_dict['between_cluster_std'] = std_of_means
                gene_dict['within_cluster_std'] = mean_of_stds

                # Prepare a row for the DataFrame (we are just using CR_V2 as CR)
                data.append(gene_dict)
                # data.append({'gene_name': gene_name, 'CR': CR_v2,'between_cluster_std':std_of_means,'within_cluster_std':mean_of_stds})

        # Convert the list of rows into a DataFrame
        cr_df = pd.DataFrame(data)

        # Filter 'between>within',ie:'std_of_means'> 'mean_of_stds'. Sort the DataFrame by CR_v2 in descending order
        cr_df_sorted = cr_df.sort_values(by='CR', ascending=False)
        # cr_v2_df_sorted = calculate_and_sort_R_dataframe(cr_v2_df, 'std_of_means', 'mean_of_stds')
        cr_df_sorted.reset_index(inplace=True)
        cr_df_sorted['between>within'] = cr_df_sorted['between_cluster_std'] > cr_df_sorted['within_cluster_std']
        if out_path is not None:
            # Ensure that the output path ends with a slash
            if not out_path.endswith('/'):
                out_path += '/'

            # Construct the file path
            file_path = f"{out_path}CR_stats{file_name}.csv"

            # Save the DataFrame to CSV
            cr_df_sorted.to_csv(file_path)

        return gene_expr_stats,cr_df_sorted


    def compute_gene_CR_MAD_stats(self,adata_subset, obs_key='Capbatch', num_genes=10,out_path = None,file_name=None):
        """
        Compute the Coefficient of Variation (CR_v1 and CR_v2) for genes in an AnnData object.

        Parameters:
        - adata_subset: AnnData object containing the data.
        - obs_key: Key to use for categorizing data (default: 'Capbatch').
        - num_genes: Number of genes to compute CR stats for (default: 10).
        - out_path: Directory where the plot will be saved (optional).
        - file_name: Name of the file to save the plot (optional).

        Returns:
        - DataFrame containing gene names and their CR_v2 values, sorted by CR_v2.
        """

        # Retrieve the categories (groups) from the observation key
        groups = adata_subset.obs[obs_key].cat.categories

        # Initialize a dictionary to store gene expression statistics
        gene_expr_stats = {}

        # Prepare a list to store data for DataFrame creation
        data = []

        # Loop through the specified number of genes
        for gene_name in adata_subset.var.index[:num_genes]:
            # Check if the gene is in the variable names of the subset
            if gene_name in adata_subset.var_names:
                # Get the index of the gene
                gene_index = np.where(adata_subset.var_names == gene_name)[0][0]

                # Initialize a sub-dictionary for this gene
                gene_expr_stats[gene_name] = {}

                # Extract the data for this gene across all groups
                overall_gene_data = np.array(adata_subset.X[:, gene_index])

                # Lists to store group-specific means and standard deviations
                mads = []
                medians = []
                # Create dict for a dataframe with all gene stats
                gene_dict = {'gene_name': gene_name}

                # Loop through each group to calculate statistics
                for group in groups:
                    # Extract the data for this gene in the current group
                    group_data = np.array(adata_subset.X[adata_subset.obs[obs_key] == group, gene_index])


                    # Calculate the Median and MAD for the group
                    # Calculate the Median
                    group_median = np.median(group_data)

                    # Calculate the absolute deviations from the median

                    # Calculate the MAD
                    group_mad = np.median(np.abs(group_data - group_median))


                    # Add the statistics to the lists
                    mads.append(group_mad)
                    medians.append(group_median)

                    # Store group-specific stats in the gene's dictionary
                    gene_expr_stats[gene_name][group] = {"median": group_median, "mad": group_mad,"mean":np.mean(group_data),"std":np.std(group_data, ddof=1)}
                    
                    # Add group-specific mean and std to the gene dictionary
                    gene_dict[f'{group}_median'] = group_median
                    gene_dict[f'{group}_mad'] = group_mad
                    gene_dict[f'{group}_mean'] = np.mean(group_data)
                    gene_dict[f'{group}_std'] = np.std(group_data, ddof=1)
                # Calculate the mean of standard deviations and standard deviation of means
                median_of_mads_within = np.median(mads)
                median_of_medians = np.median(medians)
                # mad between: mad of medians
                mad_between = np.median(np.abs(medians- median_of_medians))


                # Variability between groups mean expression / Mean variability of gene expression within each group
                CR_mad = mad_between/ median_of_mads_within if median_of_mads_within != 0 else np.nan

                # Store these values in the gene's dictionary
                gene_expr_stats[gene_name]["CR"] = CR_mad

                # add CR info to the gene_dict for the dataframe
                gene_dict['CR'] = CR_mad
                gene_dict['mad_between'] = mad_between
                gene_dict['median_mad_within'] = median_of_mads_within 

                # Prepare a row for the DataFrame (we are just using CR_V2 as CR)
                data.append(gene_dict)
                # data.append({'gene_name': gene_name, 'CR': CR_v2,'between_cluster_std':std_of_means,'within_cluster_std':mean_of_stds})

        # Convert the list of rows into a DataFrame
        cr_df = pd.DataFrame(data)

        # Filter 'between>within',ie:'std_of_means'> 'mean_of_stds'. Sort the DataFrame by CR_v2 in descending order
        cr_df_sorted = cr_df.sort_values(by='CR', ascending=False)
        # cr_v2_df_sorted = calculate_and_sort_R_dataframe(cr_v2_df, 'std_of_means', 'mean_of_stds')
        cr_df_sorted.reset_index(inplace=True)
        cr_df_sorted['between>within'] = cr_df_sorted['mad_between'] > cr_df_sorted['median_mad_within']
        if out_path is not None:
            # Ensure that the output path ends with a slash
            if not out_path.endswith('/'):
                out_path += '/'

            # Construct the file path
            file_path = f"{out_path}CR_stats{file_name}.csv"

            # Save the DataFrame to CSV
            cr_df_sorted.to_csv(file_path)

        return gene_expr_stats,cr_df_sorted


    # def plot_allcells_onegene_expression_histograms(self, adata, genes_of_interest, obs_key='Capbatch', CR_df=None,out_path=None,file_name=None):
    #     """
    #     Plot histograms of gene expression across groups for specified genes.

    #     Parameters:
    #     - adata: AnnData object containing the data.
    #     - genes_of_interest: Single gene name or list of gene names to plot histograms for.
    #     - obs_key: Key used to group the data (default: 'Capbatch').
    #     """

    #     # Convert genes_of_interest to a list if it's not already
    #     if not isinstance(genes_of_interest, list):
    #         genes_of_interest = [genes_of_interest]

    #     # Extract the groups from the specified observation
    #     groups = adata.obs[obs_key].cat.categories

    #     # Define a color palette with a distinct color for each group
    #     colors = sns.color_palette("hsv", len(groups))

    #     # Create a figure with subplots in a single row
    #     fig, axes = plt.subplots(nrows = len(genes_of_interest),ncols=1, figsize=(6, 3 * len(genes_of_interest)))
    #     if len(genes_of_interest) == 1:
    #         axes = [axes]  # Ensure axes is iterable when there's only one plot

    #     # Iterate through each gene of interest
    #     for i, gene_of_interest in enumerate(genes_of_interest):
    #         ax = axes[i]

    #         # Find the index of the gene of interest
    #         gene_index = np.where(adata.var_names == gene_of_interest)[0]
    #         if len(gene_index) > 0:
    #             gene_index = gene_index[0]
    #         else:
    #             print(f"Gene {gene_of_interest} not found.")
    #             continue

    #         # Plot histograms for each group
    #         for group in groups:
    #             group_data = adata.X[adata.obs[obs_key] == group, gene_index].flatten()
    #             #print("shape",group_data.shape)
    #             sns.histplot(group_data, ax=ax, color=colors[groups.tolist().index(group)], kde=True, label=f'Group {group}')
    #         if CR_df is not None:
    #             CR = CR_df.loc[CR_df["gene_name"]==gene_of_interest,"CR"]                
    #             between_cluster_std = np.round(CR_df.loc[CR_df["gene_name"]==gene_of_interest,"between_cluster_std"].values[0],2)
    #             within_cluster_std = np.round(CR_df.loc[CR_df["gene_name"]==gene_of_interest,"within_cluster_std"].values[0],2)
    #             print(gene_of_interest,"CR",CR,"between_cluster_std",between_cluster_std,"within_cluster_std",within_cluster_std)
    #             ax.set_title(f'Expression of {gene_of_interest}\nBetween cluster std: {between_cluster_std}, Within cluster std:{within_cluster_std}')
    #         # Plot title and labels
    #         else:
    #             ax.set_title(f'Expression of {gene_of_interest}')
    #         ax.set_xlabel('Expression Level')
    #         ax.set_ylabel('Frequency')
    #     #ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    #     # Create a shared legend for the figure
    #     handles = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', label=group) for group, color in zip(groups, colors)]
    #     fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize='small')

    #     plt.tight_layout()
    #     plt.show()

    #     if out_path and file_name:
    #         # Create the output directory if it does not exist
    #         os.makedirs(out_path, exist_ok=True)
    #         # Define the full file path
    #         file_path = os.path.join(out_path, file_name+".png")
    #         # Save the figure
    #         fig.savefig(file_path, bbox_inches='tight')
    #         print(f"Figure saved to {file_path}")




    def plot_allcells_onegene_expression_histograms(self, adata, genes_of_interest, obs_key='Capbatch', CR_df=None, out_path=None, file_name=None,selection_criteria ='std'):
        # Convert genes_of_interest to a list if it's not already
        if not isinstance(genes_of_interest, list):
            genes_of_interest = [genes_of_interest]

        # Extract the groups from the specified observation
        groups = adata.obs[obs_key].cat.categories

        # Define a color palette with a distinct color for each group
        colors = sns.color_palette("hsv", len(groups))


        # First, find the maximum value across 90 percentile all genes of interest
        max_value = 0
        for gene_of_interest in genes_of_interest:
            gene_index = np.where(adata.var_names == gene_of_interest)[0]
            if len(gene_index) > 0:
                gene_index = gene_index[0]
                # Use the 95th percentile as the maximum value for the gene
                max_value_gene = np.percentile(adata.X[:, gene_index].flatten(), 100)
                max_value = max(max_value, max_value_gene)


        # Create a figure with subplots, all subplots share the X-axis
        fig, axes = plt.subplots(nrows=len(genes_of_interest), ncols=1, figsize=(6, 3 * len(genes_of_interest)), sharex=True)
        if len(genes_of_interest) == 1:
            axes = [axes]  # Ensure axes is iterable when there's only one plot

        # Iterate through each gene of interest
        for i, gene_of_interest in enumerate(genes_of_interest):
            ax = axes[i]

            # Find the index of the gene of interest
            gene_index = np.where(adata.var_names == gene_of_interest)[0]
            if len(gene_index) > 0:
                gene_index = gene_index[0]
            else:
                print(f"Gene {gene_of_interest} not found.")
                continue

            # Plot histograms for each group
            for group in groups:
                group_data = adata.X[adata.obs[obs_key] == group, gene_index].flatten()

                # Plot histogram
                # Calculating automatic bin edges with numpy
                bin_edges = np.histogram_bin_edges(group_data, bins='auto')

                # Ensuring bin edges start from zero if the lowest edge is negative (By definition all single cell count data is positive >=0)
                if bin_edges[0] < 0:
                    bin_edges[0] = 0.01  # Adjust to ensure positivity, could be 0 for truly zero-bound data

                sns.histplot(group_data,bins=bin_edges, ax=ax, color=colors[groups.tolist().index(group)], kde=True, label=f'Group {group}')
                


            if CR_df is not None:
                CR = CR_df.loc[CR_df["gene_name"] == gene_of_interest, "CR"]
                if selection_criteria =="std":
                    between_cluster_std = np.round(CR_df.loc[CR_df["gene_name"] == gene_of_interest, "between_cluster_std"].values[0], 2)
                    within_cluster_std = np.round(CR_df.loc[CR_df["gene_name"] == gene_of_interest, "within_cluster_std"].values[0], 2)
                    ax.set_title(f'Expression of {gene_of_interest}\nBetween cluster std: {between_cluster_std}, Within cluster std: {within_cluster_std}')
                if selection_criteria =="mad":
                    between_cluster_std = np.round(CR_df.loc[CR_df["gene_name"] == gene_of_interest, "mad_between"].values[0], 2)
                    within_cluster_std = np.round(CR_df.loc[CR_df["gene_name"] == gene_of_interest, "median_mad_within"].values[0], 2)
                    ax.set_title(f'Expression of {gene_of_interest}\nMAD between: {between_cluster_std}, Median MAD within: {within_cluster_std}')
            else:
                ax.set_title(f'Expression of {gene_of_interest}')
            ax.set_xlabel('Expression Level')
            ax.set_ylabel('Frequency')
                # It's crucial to apply x-axis limits after all plotting commands
        for ax in axes:
            # Set the X-axis limits from 0 to max_value for all subplots
            ax.set_xlim(left=0, right=max_value)  # Apply at the end to ensure it takes effect

        # Create a shared legend for the figure
        handles = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', label=group) for group, color in zip(groups, colors)]
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.5, 1), fontsize='small')

        plt.tight_layout()
        plt.show()

        if out_path and file_name:
            # Create the output directory if it does not exist
            os.makedirs(out_path, exist_ok=True)
            # Define the full file path
            file_path = os.path.join(out_path, file_name + ".png")
            # Save the figure
            fig.savefig(file_path, bbox_inches='tight')
            print(f"Figure saved to {file_path}")



    def merge_and_rename_datasets(self,dataset1, dataset2, dataset1_name="dataset1", dataset2_name="dataset2",out_path=None):
        """
        Merges two datasets on 'index' and 'gene_name' columns and renames the other columns
        to include the specified dataset names as a prefix.

        Parameters:
        dataset1 (DataFrame): The first dataset.
        dataset2 (DataFrame): The second dataset.
        dataset1_name (str): Custom name for the first dataset to use as a prefix in column names.
        dataset2_name (str): Custom name for the second dataset to use as a prefix in column names.

        Returns:
        DataFrame: Merged dataset with renamed columns.
        """
        # Add a 'rank' column based on the index

        df1 = dataset1.copy()
        df2 = dataset2.copy()
        df1['rank'] = df2.index
        df2['rank'] = df2.index

        # Rename columns in dataset1
        df1.columns = ['index', 'gene_name'] + [f'{dataset1_name}_{col}' for col in df1.columns if col not in ['index', 'gene_name']]

        # Rename columns in dataset2
        df2.columns = ['index', 'gene_name'] + [f'{dataset2_name}_{col}' for col in df2.columns if col not in ['index', 'gene_name']]

        # Merge the datasets on 'index' and 'gene_name'
        merged_dataset = pd.merge(df1, df2, on=['index', 'gene_name'])
        # save merged dataset
        if out_path is not None:
            # Ensure that the output path ends with a slash
            if not out_path.endswith('/'):
                out_path += '/'

            # Construct the file path
            file_path = f"{out_path}merged_CR_stats_{dataset1_name}_{dataset2_name}.csv"

            # Save the DataFrame to CSV
            merged_dataset.to_csv(file_path)


        return merged_dataset
    





class H5ADLoader:

    """
    Loader class for reading .h5ad files and converting them into AnnData objects. 
    This loader is useful to load Heart datasets from Yu et al 2023.
    Link to Heart dataset downloads: https://figshare.com/articles/dataset/Batch_Alignment_of_single-cell_transcriptomics_data_using_Deep_Metric_Learning/20499630/2?file=38987759

    Attributes:
        file_path (str): Path to the .h5ad file to be loaded.
        adata_dict (dict): Dictionary containing the loaded data from the .h5ad file. Initially set to None.

    Methods:
        load_h5ad():
            Reads the .h5ad file specified by `file_path`, extracting its components into `adata_dict`. 
            It handles the main data matrix 'X' as a sparse matrix and other components such as 'obs', 'var', 
            'obsm', 'obsp', 'uns', 'layers', 'varm', 'varp'. For 'obs', it particularly converts categories into 
            pandas.Categorical objects.

        _handle_obs(obs_group):
            Helper method to process the 'obs' group from the .h5ad file. It converts categorical data into 
            pandas.Categorical objects for easier manipulation in pandas DataFrames.

        _handle_other_components(adata, file, key):
            Helper method to process components other than 'X' and 'obs'. This includes 'var', 'obsm', 'obsp', 
            'uns', 'layers', 'varm', 'varp'. Data is added to `adata_dict` in an appropriate format (numpy arrays or dictionaries).

        create_anndata():
            Converts `adata_dict` into an AnnData object. This method should be called after `load_h5ad` to ensure 
            data has been loaded. It handles the creation of the AnnData object by properly assigning 'X', 'obs', 
            and 'var', and additionally includes 'obsm', 'obsp', 'layers', 'uns' if they are present in `adata_dict`.

    Usage:
        loader = H5ADLoader(file_path)
        loader.load_h5ad() # Loads data from .h5ad file
        adata = loader.create_anndata() # Converts loaded data into AnnData object
    """

    def __init__(self, file_path):

        self.file_path = file_path
        self.adata_dict = None

    def load_h5ad(self):
        with h5py.File(self.file_path, "r") as f:
            adata = {}
            if {'data', 'indices', 'indptr'}.issubset(f['X']):
                print("Reading X")
                data = f['X']['data'][:]
                indices = f['X']['indices'][:]
                indptr = f['X']['indptr'][:]
                shape = f['X'].attrs['shape']

                # Check if matrix is transposed based on indptr length
                transposed = len(indptr) != shape[0] + 1
                correct_shape = (shape[1], shape[0]) if transposed else shape
                
                try:
                    X = csr_matrix((data, indices, indptr), shape=correct_shape)
                    if transposed:
                        adata['X'] = X.T
                    else:
                        adata['X']=X
                    #adata['transposed'] = transposed
                    print(f"CSR matrix created successfully with shape: {correct_shape}, Transposed: {transposed}")
                except ValueError as e:
                    print(f"Error creating CSR matrix: {e}")
            else:
                print("Required keys ('data', 'indices', 'indptr') not found in 'X'.")


            for key in f.keys():
                print("Reading:",key)
                if key !='X':
                    if key == 'obs':
                        obs_data = self._handle_obs(f[key])
                        adata['obs'] = pd.DataFrame(obs_data)
                    else:
                        self._handle_other_components(adata, f, key)

                    # Convert 'var' to DataFrame
                    if 'var' in adata:
                        adata['var'] = pd.DataFrame(adata['var'])


            self.adata_dict = adata

    def _handle_obs(self, obs_group):
        obs_data = {}
        for subkey in obs_group:
            item = obs_group[subkey]
            if isinstance(item, h5py.Group):
                categories_dataset = item['categories']
                codes_dataset = item['codes']
                categories = categories_dataset[:]
                codes = codes_dataset[:]
                obs_data[subkey] = pd.Categorical.from_codes(codes, categories.astype('U'))
            else:
                obs_data[subkey] = item[:]
        return obs_data

    def _handle_other_components(self, adata, file, key):
        if isinstance(file[key], h5py.Group):
            adata[key] = {subkey: np.array(file[key][subkey]) for subkey in file[key]}
            if not adata[key]:  # Check if the dictionary is empty
                print(f"No data found under key '{key}'")
        else:
            adata[key] = np.array(file[key])
            if adata[key].size == 0:  # Check if the array is empty
                print(f"No data found under key '{key}'")

    def create_anndata(self):
        if self.adata_dict is None:
            print("Data not loaded. Please load the .h5ad file first.")
            return None

        # Extract data from the dictionary
        X = self.adata_dict.get('X', None)
        obs = self.adata_dict.get('obs', None)
        var = self.adata_dict.get('var', None)

        # Create AnnData object
        ann_data = anndata.AnnData(X=X, obs=obs, var=var)

        # Add other components if they exist
        for key in ['obsm', 'obsp', 'layers', 'uns']:
            if key in self.adata_dict:
                setattr(ann_data, key, self.adata_dict[key])

        return ann_data


def scRNAseq_QA_Yu2023(adata, min_genes_per_cell=10, min_cells_per_gene=3, total_counts_per_cell=10000, n_top_genes=1000, standard_scale=True, hvg=True):
    """
    Perform basic quality assurance (QA) preprocessing on single-cell RNA sequencing data.

    Parameters:
    - adata: AnnData object containing the single-cell RNA sequencing data.
    - min_genes_per_cell: int, minimum number of genes to be detected in a cell.
    - min_cells_per_gene: int, minimum number of cells in which a gene must be detected.
    - total_counts_per_cell: int, target sum of counts for each cell after normalization.
    - n_top_genes: int, number of top highly variable genes to select.
    - scale: bool, whether to scale data to zero mean and unit variance.
    - hvg: bool, whether to filter the data to keep only highly variable genes.

    Returns:
    - AnnData object with the preprocessing applied.
    """
    import scanpy as sc
    # Create a copy of the data to keep the raw data unchanged
    adata_copy = adata.copy()
    
    # Filter cells and genes based on quality metrics
    sc.pp.filter_cells(adata_copy, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata_copy, min_cells=min_cells_per_gene)

    # Normalize and logarithmize the data
    sc.pp.normalize_total(adata_copy, target_sum=total_counts_per_cell)
    sc.pp.log1p(adata_copy)
    
    # Optionally select highly variable genes and scale the data
    if hvg:
        sc.pp.highly_variable_genes(adata_copy, n_top_genes=n_top_genes, subset=True)
    if standard_scale:
        sc.pp.scale(adata_copy)
    
    return adata_copy


def scRNAseq_pipeline_Yu2023(adata, min_genes_per_cell=10, min_cells_per_gene=3, total_counts_per_cell=10000, n_top_genes=1000, n_components=50, standard_scale=True, hvg=True):
    """
    Apply a complete preprocessing and visualization pipeline to single-cell RNA sequencing data.

    Parameters:
    - adata: AnnData object containing the single-cell RNA sequencing data.
    - min_genes_per_cell: int, minimum number of genes to be detected in a cell.
    - min_cells_per_gene: int, minimum number of cells in which a gene must be detected.
    - total_counts_per_cell: int, target sum of counts for each cell after normalization.
    - n_top_genes: int, number of top highly variable genes to select for the analysis.
    - n_components: int, number of principal components to use in PCA.
    - scale: bool, whether to scale data to zero mean and unit variance.
    - hvg: bool, whether to filter the data to keep only highly variable genes.

    Returns:
    - AnnData object with preprocessing and visualization (PCA and UMAP) applied.
    """
    import scanpy as sc
    # Preprocess the data using the QA function
    adata_copy = scRNAseq_QA_Yu2023(adata, min_genes_per_cell, min_cells_per_gene, total_counts_per_cell, n_top_genes, standard_scale, hvg)

    # Apply PCA, compute the neighborhood graph, and run UMAP for visualization
    sc.tl.pca(adata_copy, svd_solver='arpack', n_comps=n_components)
    sc.pp.neighbors(adata_copy)
    sc.tl.umap(adata_copy)

    return adata_copy

                
