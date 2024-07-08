# Set expt in path2results.py
# "expt_3_with_batch_norm_clusteringlosscurves_HP0_adjustment4_500epochs_linear_latentspace50" or expt=="expt3_batch_cf"
import sys
sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/compare_results")
from path2results import run_names_dict,results_path_dict,run_names_dict, glob_like,compare_models_path ,get_pca
import pandas as pd
import glob
import os
import numpy as np
import scipy.stats as stats

sys.path.append("/archive/bioinformatics/DLLab/AixaAndrade/src/ARMED_genomics_git/utils")
from compare_results_utils import preprocess_results_model_pca_format,process_single_model_format,calculate_and_append_ci


#Normally I run it with Aixa_ARMED_2.

"""This is script is to reorganize clustering scores into a single table. It also adds 95%CI"""


# 1. Find all file paths that you want to compare for all models and store them in a dataframe
# If you are developing use val dataset. If this is your final result use test dataset

paths_dict = {}
for model_name,path in results_path_dict.items(): 
    print(model_name,path)
    paths_dict[model_name]=glob_like(path, 'mean_scores_val_samplesize')

# Build df with all results path
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
    df_all_paths = df_all_paths.append(df, ignore_index=True)  # Ensure to reassign and use ignore_index=True

# Now df_all contains all the data combined from the different files and models
df_all_paths



# 2. Define how to format your results depending on the model_confil.py settings
# Edit this dictionary depending of how you want to process each data, 
# if get_pca ==True for any model when you run the pipeline. You will need to use "preprocess_results_model_pca_format"
# else use "process_single_model_format" or "" when you dont want to process it
models2process_dict = {"AE":"preprocess_results_model_pca_format",
                                        "AEC":"preprocess_results_model_pca_format",
                                        "AEC_DA":"process_single_model_format",
                                        "AE_DA":"process_single_model_format",
                                        "AE_RE":"process_single_model_format"
                                        }
                                    

# 3. Define how to name your output files
out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])
process_single_format = {}
type_of_processing_dict = {"AE"}
df_sample_size = {}

# 4. Merge all your results into a single dataframe
# for loop over sample_size and model_name
for sample_size in np.unique(df_all_paths["sample_size"]):
    print("sample size",sample_size)
    model_path_dict={}
    # df for single model format
    df_smf = pd.DataFrame()
    df_pcaf = pd.DataFrame()
    count_i=0
    for model_name in np.unique(df_all_paths["model_name"]):

        # getting results file path
        print(model_name)
        file_paths = df_all_paths.loc[(df_all_paths["sample_size"]==sample_size)&(df_all_paths["model_name"]==model_name),"path"].values        
        if len(file_paths)>0:
            file_path = file_paths[0]
            print(model_name,"file path",file_path)
            model_path_dict[model_name]= file_path
        # preprocessing models without pca results
        if models2process_dict[model_name] == "process_single_model_format":
            df_smf = df_smf.append(process_single_model_format(file_path=model_path_dict[model_name],model_name = model_name))
            print("smf",df_smf)
        # preprocessing models with pca results 
        elif models2process_dict[model_name] == "preprocess_results_model_pca_format":
            df_pcaf_i = preprocess_results_model_pca_format(model_path_dict[model_name], columns_to_drop = ['fold', 'sem_fold'])
            df_pcaf_i = df_pcaf_i.rename(columns={"X_pca_val": "X_pca_val_"+str(count_i)})
            df_pcaf = pd.concat([df_pcaf , df_pcaf_i ],axis=1) 
            
            count_i=+1

        else:
            print(model_name, "not processed")
        
    df_all = pd.concat([df_pcaf , df_smf.T],axis=1).T   
    ## Save model results dataframe for each sample-size
    df_all.to_csv(out_name+sample_size+".csv")
    print(df_all)
    df_sample_size[sample_size]=df_all


# 4. Get 95% CI
df_ci = calculate_and_append_ci(df_all, dof=4)  # n=5, dof=n-1=4

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
# Save the df with 95% CI  to a CSV file
df_mean_ci.to_csv(out_name+sample_size+"_95CI.csv")

print(df_mean_ci)