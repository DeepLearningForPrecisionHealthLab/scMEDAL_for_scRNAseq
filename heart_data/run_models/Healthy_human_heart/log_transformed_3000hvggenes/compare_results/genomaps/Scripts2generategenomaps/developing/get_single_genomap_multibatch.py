
import pandas as pd
import glob
import os
import re
import numpy as np
# import scipy.stats as stats

from anndata import AnnData
import scanpy as sc

import pandas as pd # Please install pandas and matplotlib before you run this example
import matplotlib.pyplot as plt
import matplotlib
# Set the Matplotlib backend to 'Agg'
matplotlib.use('Agg')
import numpy as np
import scipy
# import genomap as gp
import numpy as np
import scipy.stats as stats
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sklearn.metrics as mpd
# from genomap.genomapOPT import create_space_distributions, gromov_wasserstein_adjusted_norm
# from genomap.genomap import createMeshDistance

import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
from compare_results_utils import get_recon_paths_df,get_input_paths_df
from genomaps_utils import process_and_plot_genomaps_singlepath
# import the path
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/compare_results")
# Make sure you update the expt. For example expt=="expt3_batch_cf" has an example of getting counterfactuals of batch effects
from path2results import run_names_dict,results_path_dict,run_names_dict,compare_models_path 

# I run this script with Aixa_genomap env


# 1. Get input paths df and load data
# Define base paths

# get input paths df
data_base_path = "/archive/bioinformatics/DLLab/AixaAndrade/data/Genomic_data/heart_data/"
# Paths
scenario_id = "Healthy_human_heart_data/log_transformed_3000hvggenes"
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')

# Merge paths
df_recon = get_recon_paths_df(results_path_dict,get_batch_recon_paths = True)
df_inputs = get_input_paths_df(input_base_path)
df = pd.merge(df_recon, df_inputs, on=["Split", "Type"], how="left")
print("df paths:",df.head(5))


# # Lists of models, types, and splits
models = ['AE_RE']  # Add all your models to this list
types = ['train']  # Add all types you need to iterate through
splits = [1] 

#Lists of models, types, and splits
# models = np.unique(df['Key']).tolist() # Add all your models to this list
# types = ['train', 'val']  # Add all types you need to iterate through
# splits = [1, 2, 3,4,5] 

out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])
if not os.path.exists(out_name):
    os.makedirs(out_name)




# 2. GET GENOMAPS

# Construct genomaps
colNum=54 # Column number of genomap
rowNum=54 # Row number of genomap

# Change input data and folder name

inputs_path = "/pathto/CMmultibatch directory"
genomap_name = "genomap_name"
n_cells = 300*149 # 147 batches +input +fe
out_genomaps_path = os.path.join(out_name,genomap_name)


# Get gene_names: The splits did not store the real gene_ids
gene_ids_path = "/archive/bioinformatics/DLLab/AixaAndrade/data/Genomic_data/heart_data/Healthy_human_heart_data/log_transformed_3000hvggenes/geneids.csv"

var = pd.read_csv(gene_ids_path,index_col="_index")
gene_names = var.index[0:2916]


process_and_plot_genomaps_singlepath(inputs_path,
                                        ncells=n_cells,
                                        ngenes = 2916,
                                        rowNum=rowNum,
                                        colNum=colNum,
                                        epsilon = 0.0,
                                        num_iter=100,
                                        output_folder = out_genomaps_path,
                                        genomap_name = genomap_name,
                                        gene_names=gene_names)


#todo: change paths to new expt
# change df to include recon_batches
#  also change process_and_plot_genomaps
#Once T plotted perturb the genomap