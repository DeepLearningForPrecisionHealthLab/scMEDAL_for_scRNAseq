import scanpy as sc
import h5py
from scipy.sparse import csr_matrix
import anndata
import numpy as np
import pandas as pd
import zipfile
import glob
import os


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

# Loader for the Healthy Heart dataset
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
    
# Loader for AML dataset

class AMLDataReader:
    """Class to read adata from vanGalen et al 2019. Downloaded from Gene Expression Obnibus with acession number GSE116256. A count matrix for each individual is stored separately in a zip file"""
    def __init__(self, parent_path:str):
        self.parent_path = parent_path
        self._check_parentpath_zipped()


    def _check_parentpath_zipped(self):
        if self.parent_path.endswith(".zip"):
            try:
                new_dir_name = os.path.split(self.parent_path)[-1].split(".")[0]
                new_dir = os.path.join(os.path.dirname(self.parent_path), new_dir_name)
                os.makedirs(new_dir, exist_ok=True)
                zipfile.ZipFile(file=self.parent_path).extractall(path=os.path.dirname(self.parent_path))
                # Update parent path:
                self.parent_path = new_dir
            except Exception as e:
                raise e
        


    def extract_accession(self,annotation):
        id = annotation.split("/")[-1].split('-')[0].split('_')[0]
        return id

    def extract_id(self,annotation):
        id = annotation.split("/")[-1].split('-')[0].split('_')[1].split('.')[0]
        return id

    def extract_file_note(self, annotation):
        
        ID = self.extract_id(annotation)
        day_part = annotation.split("D")[-1].split('.')[0]
        note = annotation.split("BM")[-1].split('.')[0]
        return_ = np.nan
        if 'AML' in ID:
            return_ = 'D' + day_part
        elif 'BM' in ID:
            return_ = 'BM' + note
        return return_

    def add_Day_column(self,df_paths):
    # PREREQUISITE: Have file_note col
        result_list = []
        for d in df_paths['file_note']:
            if isinstance(d, str) and 'D' in d:
                result_list.append(d)  # Keep the string if it contains 'D'
            else:
                result_list.append(np.nan)  # Replace non-matching entries with np.nan
        df_paths['Day'] = result_list
        return df_paths

    def add_real_id_column(self, df_paths):
        df_paths["unique_id"] = df_paths["id"].copy()
        df_paths["Patient_group"] = np.nan
        for i in range(len(df_paths)):
            day = df_paths.loc[i, 'Day']
            id = df_paths.loc[i, 'id']
            if "AML" in id:
                df_paths.loc[i, "Patient_group"] = "AML"
            elif "BM" in id:
                df_paths.loc[i, "Patient_group"] = "control"
            elif id in ['MUTZ3', 'OCI']:
                df_paths.loc[i, "Patient_group"] = "cellline"
            if not pd.isna(day):
                df_paths.at[i, "unique_id"] = id + "_" + str(day)
            elif id == "BM5":
                df_paths.at[i, "unique_id"] = df_paths.loc[i, 'file_note']
        return df_paths

    def get_df_paths(self):
        anno_list = glob.glob(self.parent_path + '/*anno*')
        matrix_list = glob.glob(self.parent_path + '/*dem*')

        anno_df = pd.DataFrame({'anno_path': anno_list,
                                'id': [self.extract_id(m) for m in anno_list],
                                'file_note': [self.extract_file_note(m) for m in anno_list],
                                'accession_anno_num': [self.extract_accession(m) for m in anno_list]})
        matrix_df = pd.DataFrame({'matrix_path': matrix_list,
                                  'id': [self.extract_id(m) for m in matrix_list],
                                  'file_note': [self.extract_file_note(m) for m in matrix_list],
                                  'accession_matrix_num': [self.extract_accession(m) for m in matrix_list]})
        df_paths = pd.merge(matrix_df, anno_df, on=["id", "file_note"], how="outer")
        df_paths = self.add_Day_column(df_paths)
        df_paths = self.add_real_id_column(df_paths)

        # count the occurrences id, Patient_group
        grouped_counts = df_paths.groupby(['id', 'Patient_group']).size().reset_index(name='counts')

        print(grouped_counts)
        return df_paths
        #return anno_df, matrix_df

    def create_adata_dict(self, df_paths):
        adata_dict = {}
        for i in range(len(df_paths)):
            matrix_path_i = df_paths.loc[i, 'matrix_path']
            cm = pd.read_csv(matrix_path_i, sep='\t', index_col=0).T
            anno_path_i = df_paths.loc[i, 'anno_path']
            meta = pd.read_csv(anno_path_i, sep='\t', index_col=0).reset_index()
            for col in ['id', 'Day', 'unique_id', 'Patient_group']:
                meta[col] = df_paths.loc[i, col]
            unique_id = df_paths.loc[i, 'unique_id']
            if meta.shape[0] == cm.shape[0]:
                adata = anndata.AnnData(X=cm.values, obs=meta, var=pd.DataFrame(cm.columns))
                adata_dict[unique_id] = adata
        return adata_dict

    def merge_adata_objects(self, adata_dict):
        equal_genes = True
        key_0 = next(iter(adata_dict.keys()))
        for key in adata_dict:
            if key != key_0 and not adata_dict[key_0].var.equals(adata_dict[key].var):
                equal_genes = False
                print(f"{key} genes are different")
                break
        if not equal_genes:
            print("Check that your genes are equal for all individual count matrices")
            return None
        all_obs = pd.concat([adata.obs for adata in adata_dict.values()], ignore_index=True)
        all_X = np.concatenate([adata.X for adata in adata_dict.values()], axis=0)
        all_var = next(iter(adata_dict.values())).var
        return anndata.AnnData(X=all_X, obs=all_obs, var=all_var)

