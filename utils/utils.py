import anndata
import pandas as pd
import os
import glob
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import time


# create_folder,read_adata,get_OHE,min_max_scaling,plot_rep,calculate_merge_scores,get_split_paths,calculate_zscores,get_clustering_scores_optimized

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



def get_clustering_scores_optimized(adata, use_rep, labels, sample_size=None):
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
    """
    Optimized version of getting clustering scores from an AnnData object with optional subsampling for all scores.

    Args:
    - adata: An AnnData object.
    - use_rep: Representation to use. Any key for .obsm is valid. If 'X', then adata.X is used.
    - labels: Labels for clustering. Any key for .obs is valid.
    - sample_size: Optional integer. If specified, subsamples this number of instances for score calculation.

    Returns:
    - A pandas DataFrame with Davies-Bouldin (DB), inverse DB, Calinski-Harabasz (CH), and Silhouette scores for each label.
    """
    
    start_time = time.time()
    print("Computing scores..")
    if sample_size is not None:
        print("sample_size:",sample_size)
    else:
        print("all cells used")
        
    print("\nDB: lower values --> better clustering")
    print("1/DB, CH, Silhouette: higher values --> better clustering")

    # Determine the data representation
    data_rep = adata.X if use_rep == "X" else adata.obsm[use_rep]

    dict_scores = {}
    time_per_score = {}

    # Subsample the data if a sample size is specified
    if sample_size and sample_size < data_rep.shape[0]:
        indices = np.random.choice(data_rep.shape[0], sample_size, replace=False)
        subsampled_data_rep = data_rep[indices]
    else:
        subsampled_data_rep = data_rep

    # Compute scores for each label and store in dict
    for label in labels:
        labels_array = adata.obs[label].to_numpy()
        if sample_size and sample_size < data_rep.shape[0]:
            subsampled_labels = labels_array[indices]
        else:
            subsampled_labels = labels_array

        score_start_time = time.time()
        db_score = davies_bouldin_score(subsampled_data_rep, subsampled_labels)
        db_time = time.time() - score_start_time

        score_start_time = time.time()
        ch_score = calinski_harabasz_score(subsampled_data_rep, subsampled_labels)
        ch_time = time.time() - score_start_time

        score_start_time = time.time()
        silhouette = silhouette_score(subsampled_data_rep, subsampled_labels)
        silhouette_time = time.time() - score_start_time

        dict_scores[label] = [db_score, 1/db_score, ch_score, silhouette]
        time_per_score[label] = [db_time, ch_time, silhouette_time]

    # Create DataFrame from dict
    df_scores = pd.DataFrame(dict_scores, index=['db', '1/db', 'ch', 'silhouette'])
    total_time = time.time() - start_time

    print(f"Total computation time: {total_time} seconds")
    for label, times in time_per_score.items():
        print(f"Time per score for {label} - DB: {times[0]:.4f}, CH: {times[1]:.4f}, Silhouette: {times[2]:.4f} seconds")

    return df_scores



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


def read_adata(folder_path, issparse=False):
    """
    Reads data from a folder and loads it into memory.

    Parameters:
    folder_path (str): The path to the folder containing the data.
    isdense (bool): A flag indicating if the data is stored in dense format. If True, allows pickling.

    Returns:
    tuple: A tuple containing the expression matrix (X), variable annotations (var), and observation annotations (obs).
    """
    print("Reading data from:",folder_path)
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
    Note: The function doesn't alter the order of the rows in the original AnnData object.
    Instead, it affects how the data is encoded into one-hot format based on the column specified. 

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



# def plot_rep(adata, shape_col="celltype", color_col="donor", use_rep="X_pca", markers=['o', 'v', '^', '<', '*'], clustering_scores=None, save_fig=True, outpath="", showplot=False, palette_choice="tab20",file_name="latent"):
#     """Plots a dimensionally reduced representation of adata."""
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import numpy as np
#     import itertools
#     from matplotlib.lines import Line2D

#     print("plotting latent representation:", use_rep)
#     plt.ioff()
#     fig, ax = plt.subplots(figsize=(7, 7))

#     unique_shapecol = np.unique(adata.obs[shape_col])
#     unique_colorcol = np.unique(adata.obs[color_col])
#     print("unique_colorcol",unique_colorcol)

#     # Choose the color palette based on the palette_choice argument
#     if isinstance(palette_choice, list):
#         color_palette = palette_choice
#     elif palette_choice == "hsv":
#         color_palette = sns.color_palette("hsv", len(unique_colorcol))
#     elif palette_choice == "tab20":
#         # Ensure tab20 has enough colors, otherwise cycle through them
#         color_palette = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(unique_colorcol))]
#     elif palette_choice == "Set2":
#         color_palette = sns.color_palette("Set2", len(unique_colorcol))
#     else:
#         raise ValueError("Invalid palette choice. Please choose 'hsv', 'tab20', 'Set2', or provide a list of colors.")
    

#     color_map = {color: color_palette[i] for i, color in enumerate(unique_colorcol)}
#     shape_map = {shape: markers[i % len(markers)] for i, shape in enumerate(unique_shapecol)}
#     if use_rep =='X':
#         c1 = adata.X[:, 0]
#         c2 = adata.X[:, 1]
#     else:
#         c1 = adata.obsm[use_rep][:, 0]
#         c2 = adata.obsm[use_rep][:, 1]


#     for shape in unique_shapecol:
#         for color in unique_colorcol:
#             mask = (adata.obs[color_col] == color) & (adata.obs[shape_col] == shape)
#             ax.scatter(c1[mask], c2[mask], color=color_map[color], marker=shape_map[shape], alpha=0.7, s=1)

#     # Create legends
#     color_legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[c], markersize=10) for c in unique_colorcol]
#     shape_legend_elements = [Line2D([0], [0], marker=shape_map[s], color='black', markerfacecolor='black', markersize=10) for s in unique_shapecol]

#     legend1 = ax.legend(handles=color_legend_elements, labels=list(unique_colorcol), loc='upper left', bbox_to_anchor=(1, 1), title=color_col)
#     plt.gca().add_artist(legend1)

#     legend2 = ax.legend(handles=shape_legend_elements, labels=list(unique_shapecol), loc='lower center', bbox_to_anchor=(0.5, -0.65), ncol=3, title=shape_col)
#         #add clustering scores to the plot
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
#     if save_fig and outpath:
#         fig.savefig(f"{outpath}/{use_rep}_{file_name}.png", bbox_extra_artists=(legend1, legend2), bbox_inches='tight')
#     if showplot:
#         plt.show()
#     else:
#         plt.close("all")
# def plot_rep(
#     adata,
#     shape_col: str = "celltype",
#     color_col: str = "donor",
#     use_rep: str = "X_pca",
#     markers=('o', 'v', '^', '<', '*'),
#     clustering_scores=None,
#     save_fig: bool = True,
#     outpath: str = "",
#     showplot: bool = False,
#     palette_choice="tab20",
#     file_name: str = "latent",
#     axes: bool = True,               
#     legend_box: bool = True,          
# ):
#     """
#     Scatter UMAP/t-SNE/PCA with optional axis-free, border-free view.
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import numpy as np
#     from matplotlib.lines import Line2D

#     plt.ioff()
#     fig, ax = plt.subplots(figsize=(7, 7))

#     #  palette
#     uniq_color = np.unique(adata.obs[color_col])
#     if isinstance(palette_choice, list):
#         palette = palette_choice
#     elif palette_choice == "hsv":
#         palette = sns.color_palette("hsv", len(uniq_color))
#     elif palette_choice == "tab20":
#         palette = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(uniq_color))]
#     elif palette_choice == "Set2":
#         palette = sns.color_palette("Set2", len(uniq_color))
#     else:
#         raise ValueError("Invalid palette_choice")

#     color_map = dict(zip(uniq_color, palette))
#     shape_map = {s: markers[i % len(markers)]
#                  for i, s in enumerate(np.unique(adata.obs[shape_col]))}

#     # coordinates
#     c1, c2 = (adata.X.T[:2] if use_rep == "X"
#               else adata.obsm[use_rep].T[:2])

#     #  scatter 
#     for s in shape_map:
#         for c in color_map:
#             m = (adata.obs[shape_col] == s) & (adata.obs[color_col] == c)
#             ax.scatter(c1[m], c2[m], s=1, alpha=0.7,
#                        marker=shape_map[s], color=color_map[c])

#     #  legends 
#     kw = dict(frameon=legend_box)
#     leg1 = ax.legend([Line2D([0], [0], marker='o', color='w',
#                              markerfacecolor=color_map[c], markersize=8)
#                       for c in uniq_color],
#                      uniq_color, loc='upper left', bbox_to_anchor=(1, 1),
#                      title=color_col, **kw)
#     ax.add_artist(leg1)

#     leg2 = ax.legend([Line2D([0], [0], marker=shape_map[s], color='black',
#                              markerfacecolor='black', markersize=8)
#                       for s in shape_map],
#                      list(shape_map), loc='lower center',
#                      bbox_to_anchor=(0.5, -0.65), ncol=3,
#                      title=shape_col, **kw)

#     # axis & box handling 
#     if not axes:
#         ax.set_axis_off()                     # remove frame, ticks, labels
#     else:
#         ax.set_xlabel(f"{use_rep} 1")
#         ax.set_ylabel(f"{use_rep} 2")

#     #  save / show 
#     if save_fig and outpath:
#         fig.savefig(f"{outpath}/{use_rep}_{file_name}.png",
#                     bbox_extra_artists=(leg1, leg2),
#                     bbox_inches='tight', dpi=300)
#     if showplot:
#         plt.show()
#     plt.close(fig)


def plot_rep(
    adata,
    shape_col: str = "celltype",
    color_col: str = "donor",
    use_rep: str = "X_pca",
    markers=('o', 'v', '^', '<', '*'),
    clustering_scores=None,
    save_fig: bool = True,
    outpath: str = "",
    showplot: bool = False,
    palette_choice="tab20",
    file_name: str = "latent",
    axes: bool = True,          # show / hide axis & ticks
    legend_box: bool = True     # True ? legend with box, False ? *no* legend
):
    """
    Scatter UMAP/t-SNE/PCA with optional axis-free view and optional legends.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.lines import Line2D

    plt.ioff()
    fig, ax = plt.subplots(figsize=(7, 7))

    #  palette 
    uniq_color = np.unique(adata.obs[color_col])
    if isinstance(palette_choice, list):
        palette = palette_choice
    elif palette_choice == "hsv":
        palette = sns.color_palette("hsv", len(uniq_color))
    elif palette_choice == "tab20":
        palette = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(uniq_color))]
    elif palette_choice == "Set2":
        palette = sns.color_palette("Set2", len(uniq_color))
    else:
        raise ValueError("Invalid palette_choice")

    color_map = dict(zip(uniq_color, palette))
    shape_map = {s: markers[i % len(markers)]
                 for i, s in enumerate(np.unique(adata.obs[shape_col]))}

    #  scatter
    c1, c2 = (adata.X.T[:2] if use_rep == "X"
              else adata.obsm[use_rep].T[:2])

    for s in shape_map:
        for c in color_map:
            m = (adata.obs[shape_col] == s) & (adata.obs[color_col] == c)
            ax.scatter(c1[m], c2[m], s=1, alpha=0.7,
                       marker=shape_map[s], color=color_map[c])

    #  legends (optional) 
    extra_artists = []
    if legend_box:                      # ? only build legends when True
        leg1 = ax.legend(
            [Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=color_map[c], markersize=8)
             for c in uniq_color],
            uniq_color,
            loc='upper left', bbox_to_anchor=(1, 1),
            title=color_col, frameon=True
        )
        ax.add_artist(leg1)

        leg2 = ax.legend(
            [Line2D([0], [0], marker=shape_map[s], color='black',
                    markerfacecolor='black', markersize=8)
             for s in shape_map],
            list(shape_map),
            loc='lower center', bbox_to_anchor=(0.5, -0.65),
            ncol=3, title=shape_col, frameon=True
        )
        extra_artists.extend([leg1, leg2])

    #  axis handling 
    if not axes:
        ax.set_axis_off()
    else:
        ax.set_xlabel(f"{use_rep} 1")
        ax.set_ylabel(f"{use_rep} 2")

    #  save / show 
    if save_fig and outpath:
        fig.savefig(f"{outpath}/{use_rep}_{file_name}.png",
                    bbox_extra_artists=extra_artists,
                    bbox_inches='tight', dpi=300)
    if showplot:
        plt.show()
    plt.close(fig)


def plot_rep_simple(
    adata,
    shape_col: str = "celltype",
    color_col: str = "donor",
    use_rep: str = "X_pca",
    markers=('o', 'v', '^', '<', '*'),       # ignored
    clustering_scores=None,                  # ignored
    save_fig: bool = True,
    outpath: str = "",
    showplot: bool = False,
    palette_choice: str = "tab20",
    file_name: str = "latent",
    axes: bool = True,
    legend_box: bool = True,                 # keeps old name
    *args, **kwargs                          # swallow anything else
):
    """
    Simple scatter of a 2-D latent representation.
    Points are coloured by `color_col`; `shape_col` and all shape-related
    options are ignored (kept only for API compatibility).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # ---------- palette ----------
    uniq_color = np.unique(adata.obs[color_col])
    if isinstance(palette_choice, list):
        palette = palette_choice
    elif palette_choice == "hsv":
        palette = sns.color_palette("hsv", len(uniq_color))
    elif palette_choice == "tab20":
        palette = [plt.cm.tab20(i) for i in np.linspace(0, 1, len(uniq_color))]
    elif palette_choice == "Set2":
        palette = sns.color_palette("Set2", len(uniq_color))
    else:
        raise ValueError("Invalid palette_choice")

    color_map = dict(zip(uniq_color, palette))
    #print("using color_map:",color_map.items())

    # ---------- coordinates ----------
    coords = adata.X if use_rep == "X" else adata.obsm[use_rep]
    x, y = coords[:, 0], coords[:, 1]

    # ---------- plot ----------
    plt.ioff()
    fig, ax = plt.subplots(figsize=(7, 7))
    for grp in uniq_color:
        m = adata.obs[color_col] == grp
        #print("plotting ",grp,color_map[grp])
        #print(len(x[m]),len(y[m]))


        ax.scatter(x[m], y[m], s=2, alpha=0.7, color=color_map[grp], label=grp)

    # axis & labels
    if axes:
        ax.set_xlabel(f"{use_rep}-1")
        ax.set_ylabel(f"{use_rep}-2")
    else:
        ax.set_axis_off()

    # legend (colour only)
    if legend_box:
        ax.legend(title=color_col, loc="upper left", bbox_to_anchor=(1, 1), 
        scatterpoints=1,   # one marker per label
        markerscale=8  )

    # save / show
    if save_fig and outpath:
        fig.savefig(f"{outpath}/{use_rep}_{file_name}.png",
                    bbox_inches="tight", dpi=300)
    if showplot:
        plt.show()
    plt.close(fig)

    
######################################## For comparing models

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





def calculate_merge_scores(latent_list, adata, labels,sample_size=None):
    """
    Calculates clustering scores for multiple latent representations using the provided labels,
    and then aggregates them into a single DataFrame.
    """
    # Initialize a list to hold the DataFrame rows before concatenation
    scores_list = []

    # Iterate over each latent representation and calculate clustering scores
    for latent in latent_list:
        # Assuming get_clustering_scores_optimized returns a DataFrame with the scores
        scores = get_clustering_scores_optimized(adata, latent, labels,sample_size)
        
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

                
def calculate_zscores(data):
    import scipy.stats as stats
    """
    Calculate z-scores for the given data, ignoring columns with zero variance. (Equivalent to scanpy.pp.scale)

    Args:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: Array of z-scores.
    """
    # Replace NaNs with zeros
    data_no_nan = np.nan_to_num(data, nan=0.0)

    # Identify columns with zero variance
    std_dev = np.std(data_no_nan, axis=0, ddof=1)
    zero_variance_columns = std_dev == 0

    # Calculate z-scores, ignoring zero variance columns
    z_scores = np.zeros_like(data_no_nan)
    non_zero_var_columns = ~zero_variance_columns
    z_scores[:, non_zero_var_columns] = stats.zscore(data_no_nan[:, non_zero_var_columns], axis=0, ddof=1)

    # Check for NaNs in the z-scores
    if np.isnan(z_scores).any():
        print("Input contains NaNs")
    else:
        print("Input does not contain NaNs")

    return z_scores