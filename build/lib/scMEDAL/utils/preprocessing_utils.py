import scanpy as sc

def scRNAseq_pipeline(adata, min_genes_per_cell=10, min_cells_per_gene=3, total_counts_per_cell=10000, n_top_genes=1000, n_components=50):
    # 1. Filter low quality cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    # 2. Normalize with total UMI count per cell
    sc.pp.normalize_total(adata, target_sum=total_counts_per_cell)

    # 3. Logarithmize the data
    sc.pp.log1p(adata)

    # 4. Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # Filter the data for highly variable genes
    adata = adata[:, adata.var.highly_variable]

    # 5. Apply PCA: This solver uses the ARPACK implementation of the truncated singular value decomposition (SVD). It's more suitable for smaller datasets or when you need a small number of components (principal components). It can be slower but is typically more accurate, especially when the number of requested components is much smaller than the number of features.
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_components)

    return adata

def plot_pca(adata, sample_id_column='sampleID'):

    sc.pl.pca(adata, color=sample_id_column)

# import scanpy as sc


def scRNAseq_pipeline_log(adata, min_genes_per_cell=10, min_cells_per_gene=3, total_counts_per_cell=10000):
    # Create a copy of the data to keep the raw data unchanged
    adata_copy = adata.copy()
    
    # 1. Filter low quality cells and genes
    sc.pp.filter_cells(adata_copy, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata_copy, min_cells=min_cells_per_gene)

    # 2. Normalize with total UMI count per cell
    sc.pp.normalize_total(adata_copy, target_sum=total_counts_per_cell)

    # 3. Logarithmize the data
    sc.pp.log1p(adata_copy)
    return adata_copy

def scRNAseq_pipeline_loghvg(adata, min_genes_per_cell=10, min_cells_per_gene=3, total_counts_per_cell=10000,n_top_genes=3000):
    # Create a copy of the data to keep the raw data unchanged
    adata_copy = adata.copy()
    
    # 1. Filter low quality cells and genes
    sc.pp.filter_cells(adata_copy, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata_copy, min_cells=min_cells_per_gene)

    # 2. Normalize with total UMI count per cell
    sc.pp.normalize_total(adata_copy, target_sum=total_counts_per_cell)

    # 3. Logarithmize the data
    sc.pp.log1p(adata_copy)
    # 4. Select highly variable genes and subset
    sc.pp.highly_variable_genes(adata_copy, n_top_genes=n_top_genes, subset=True)
    return adata_copy

def scRNAseq_pipeline_v2(adata, min_genes_per_cell=10, min_cells_per_gene=3, total_counts_per_cell=10000, n_top_genes=1000, n_components=50):
    # Create a copy of the data to keep the raw data unchanged
    adata_copy = adata.copy()
    
    # 1. Filter low quality cells and genes
    sc.pp.filter_cells(adata_copy, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata_copy, min_cells=min_cells_per_gene)

    # 2. Normalize with total UMI count per cell
    sc.pp.normalize_total(adata_copy, target_sum=total_counts_per_cell)

    # 3. Logarithmize the data
    sc.pp.log1p(adata_copy)

    # 4. Select highly variable genes and subset
    sc.pp.highly_variable_genes(adata_copy, n_top_genes=n_top_genes, subset=True)

    # 5. Scale the data
    sc.pp.scale(adata_copy)

    # 6. Apply PCA
    sc.tl.pca(adata_copy, svd_solver='arpack', n_comps=n_components)

    # 7. Compute the neighborhood graph
    sc.pp.neighbors(adata_copy)

    # 8. Run UMAP
    sc.tl.umap(adata_copy)

    # Return the processed data
    return adata_copy

def plot_results(adata_copy, batch_column="BATCH", celltype_column="celltype",file_name ='heart_latent'):
    # Plot UMAP with color annotations for batch and cell type
    sc.pl.pca(adata_copy,color=[batch_column, celltype_column],save=file_name+'.png')
    sc.pl.umap(adata_copy,color=[batch_column, celltype_column],save=file_name+'.png')


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
    


