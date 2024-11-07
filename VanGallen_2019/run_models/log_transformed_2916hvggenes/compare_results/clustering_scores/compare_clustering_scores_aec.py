# Set expt in path2results.py

import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/VanGallen_2019/run_models/log_transformed_2916hvggenes/compare_results")
from path2results import run_names_dict,results_path_dict,run_names_dict,compare_models_path
import pandas as pd
import glob
import os
import numpy as np
import scipy.stats as stats

sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
# from compare_results_utils import preprocess_results_model_pca_format,process_single_model_format,calculate_and_append_ci,glob_like
from compare_results_utils import aggregate_paths,read_and_aggregate_scores,filter_min_max_silhouette_scores,process_all_results, process_confidence_intervals


#Normally I run it with Aixa_ARMED_2.

"""This is script is to reorganize clustering scores into a single table. It also adds 95%CI"""


# 1. Find all file paths that you want to compare for all models and store them in a dataframe
# If you are developing use val dataset. If this is your final result use test dataset
# dataset_type='val'
dataset_type='val'


# 1. Define how to name your output files
out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])
# Check if the directory exists, if not, create it
if not os.path.exists(out_name):
    os.makedirs(out_name)

print(f"Directory created or already exists: {out_name}")

# 2. Get paths

df_all_paths = aggregate_paths(results_path_dict, pattern = f'mean_scores_{dataset_type}_samplesize')
print("\n\nmean scores paths",df_all_paths.head())


df_all_paths_allscores = aggregate_paths(results_path_dict, pattern = f'all_scores_{dataset_type}_samplesize')
print("\nall scores paths",df_all_paths_allscores)

print("\nall scores paths cols",df_all_paths_allscores.columns )

# 3. Read all scores
df_allscores = read_and_aggregate_scores(df_all_paths_allscores)
print("\ndf with all scores",df_allscores)
df_allscores.to_csv(os.path.join(out_name, f"{dataset_type}_allscores.csv"))
# for index, row in df_all_paths_allscores.iterrows():
#    print("\n",row['model_name'], row['path'].split('/')[-1])
# print("\n\n\ndf_allscores.columns",df_allscores.columns)
# Getting rows with max and min silhouette score

# 4. Filter min and max_sil scores from batch 
for sample_size in np.unique(df_allscores["sample_size"]):
    df_min_silhouette, df_max_silhouette = filter_min_max_silhouette_scores(df_allscores, batch_col="batch")
    df_min_silhouette.to_csv(os.path.join(out_name, f"{dataset_type}_scores_{sample_size}_min_silhouette_batch.csv"))
    df_max_silhouette.to_csv(os.path.join(out_name, f"{dataset_type}_scores_{sample_size}_max_silhouette_batch.csv"))


    # Print the resulting DataFrames
    print("DataFrame with minimum silhouette scores:")
    print(df_min_silhouette)

    print("\nDataFrame with maximum silhouette scores:")
    print(df_max_silhouette)


# 5. Define how to format your results depending on the model_confil.py settings
# Edit this dictionary depending of how you want to process each data, 
# if get_pca ==True for any model when you run the pipeline. You will need to use "preprocess_results_model_pca_format"
# else use "process_single_model_format" or "" when you dont want to process it
models2process_dict = {"AEC_reconloss100_classloss0.1":"preprocess_results_model_pca_format",
                        "AEC_reconloss1_classloss0.1":"preprocess_results_model_pca_format",
                        "AEC_reconloss10_classloss0.1":"preprocess_results_model_pca_format"}


print("\n\nGet df_sample_size")

df_sample_size = process_all_results(df_all_paths, models2process_dict, out_name, dataset_type)



# 4. Get 95% CI
print("\n\nGet 95% CI")
for sample_size, df_all in df_sample_size.items():
    df_mean_ci_results = process_confidence_intervals(df_all, out_name, dataset_type, sample_size)
    print("sample_size",sample_size,df_mean_ci_results)


