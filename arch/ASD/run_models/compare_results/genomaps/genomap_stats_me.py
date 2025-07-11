import os
import sys
import gc


import numpy as np
import pandas as pd

import matplotlib

from anndata import AnnData 




import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# Set Matplotlib backend to 'Agg' to avoid GUI errors
matplotlib.use('Agg')








from scMEDAL.utils.compare_results_utils import (
    get_recon_paths_df,
    get_input_paths_df
)

from scMEDAL.utils.utils import read_adata

# Add the path to the configuration file
# Path to paths_config: /MyscMEDALExpt/Experiments/ASD/paths_config.py
sys.path.append("../../../")

# Make sure you update the expt
from paths_config import (
    run_names_dict,
    results_path_dict,
    compare_models_path,
    data_base_path,
    scenario_id,
    input_base_path
)

# ----------------------------------------------------------------------------
# 1. Load Input and Reconstruction Paths
# ----------------------------------------------------------------------------

df_recon = get_recon_paths_df(results_path_dict, get_batch_recon_paths=True)
df_inputs = get_input_paths_df(input_base_path)
df = pd.merge(df_recon, df_inputs, on=["Split", "Type"], how="left")
df["recon_prefix"] = [path.split("/")[-1].split(".npy")[0] for path in df["ReconPath"]]
print("Reading paths,\ndf paths:", df.head(5))

# ----------------------------------------------------------------------------
# 2. Define Model Parameters and Paths
# ----------------------------------------------------------------------------
# Define lists of models, types, and splits. This script will only run one genomap.
models = ['scMEDAL-RE']  # Add all your models to this list
types = ['train']        # Add all types you need to iterate through
splits = [2]             # Get data from fold 2

# Select the first model, type, and split
Type = types[0]
Split = splits[0]
model_name = models[0]

# RE recon path: The counterfactual batches are stored in this directory
re_recon_path = os.path.join(results_path_dict["scMEDAL-RE"], f"splits_{Split}")

# Define experiment output directory
out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])
if not os.path.exists(out_name):
    os.makedirs(out_name)
print("Saving results to", out_name)

# Select relevant paths
recon_paths = df.loc[(df["Key"] == model_name) & (df["Split"] == Split) & (df["Type"] == Type) & (df["recon_prefix"] != "recon_train"), "ReconPath"].values
recon_prefix = df.loc[(df["Key"] == model_name) & (df["Split"] == Split) & (df["Type"] == Type) & (df["recon_prefix"] != "recon_train"), "recon_prefix"].values
print("n recon paths:", len(recon_paths))

# Inputs path
inputs_path = df.loc[(df["Key"] == model_name) & (df["Split"] == Split) & (df["Type"] == Type), "InputsPath"].values[0]
# --------------------------------------------------------------------------------------
# 1.2. Extra (optional): If extra paths are added, set add_inputs_fe = True
# Add cells from input + model reconstructions for the genomap
# --------------------------------------------------------------------------------------
add_inputs_fe = True
# extra_recon = "fe" or "all"
extra_recon = "fe"

if extra_recon == "fe":
    # n_inputs_fe = 2: One for inputs and another one for fixed effects (fe).
    # You could also add a fe classifier recon and a base autoencoder recon.
    n_inputs_fe = 2
    # Get fixed effects path: unique for "scMEDAL-FE"
    fe_ae_path = df.loc[
        (df["Key"] == "scMEDAL-FE") & (df["Split"] == Split) & (df["Type"] == Type),
        "ReconPath"
    ].values[0]
    # Define extra paths and prefix
    extra_paths = [inputs_path, fe_ae_path]
    extra_prefix = [f"input_{Type}", f"fe_ae_recon_{Type}"]

# ----------------------------------------------------------------------------
# 4. Load Gene and Cell Metadata
# ----------------------------------------------------------------------------

gene_ids_path = os.path.join(data_base_path, scenario_id, "geneids.csv")
gene_index_col = "gene_ids"
var = pd.read_csv(gene_ids_path, index_col=gene_index_col)

_, _, obs = read_adata(inputs_path, issparse=False)

# ----------------------------------------------------------------------------
# 5. Define Variables
# ----------------------------------------------------------------------------

celltype = "L2/3"
n_cells_per_batch = 300
n_batches = 31

n_genes = 2916
colNum, rowNum = 54, 54

if add_inputs_fe:
    recon_prefix = extra_prefix + recon_prefix.tolist()
    recon_paths = extra_paths + recon_paths.tolist()
    n_cells = n_cells_per_batch * (n_batches + n_inputs_fe)
else:
    n_cells = n_cells_per_batch * n_batches

# ----------------------------------------------------------------------------
# 6. Load Multibatch Count Matrix
# ----------------------------------------------------------------------------

celltype_name = celltype.replace("/", "")
cm_multibatch_name = f"CMmultibatch_{n_cells_per_batch}_cells_per_batch_{n_batches}batches_{celltype_name}"
if add_inputs_fe:
    cm_multibatch_name += f"_with_{n_inputs_fe}fe_input"
cm_multibatch_path = os.path.join(out_name, cm_multibatch_name)
print("cm_multibatch_path",cm_multibatch_path)
X_multibatch, var_multibatch, obs_multibatch = read_adata(cm_multibatch_path, issparse=False)
adata_multibatch_n_cells = AnnData(X=X_multibatch, var=var_multibatch, obs=obs_multibatch)
adata_multibatch_n_cells.obs.index = adata_multibatch_n_cells.obs.index.astype(int)

gc.collect()
print("adata_multibatch_n_cells.X", adata_multibatch_n_cells.X.shape)

# ----------------------------------------------------------------------------
# 7. Generate Genomaps paths
# ----------------------------------------------------------------------------

genomap_name = f"{n_cells_per_batch}cells_per_batch_{n_batches}batches_{Type}_{Split}"
if celltype_name:
    genomap_name += f"_{celltype_name}"
if add_inputs_fe:
    genomap_name += f"_with_{n_inputs_fe}fe_input"

path_2_genomap = os.path.join(out_name, genomap_name)
print("Genomap stored in", path_2_genomap)

# ----------------------------------------------------------------------------
# 8. Load Genomap Data
# ----------------------------------------------------------------------------

order = 'C'
statistic = 'std'

genomap_path = os.path.join(path_2_genomap, f'genomap_{genomap_name}.npy')
genomap_coordinates_path = os.path.join(path_2_genomap, f'gene_coordinates_{genomap_name}.csv')
print("Genomap path:", genomap_path)

genomap = np.load(genomap_path)
genomap_coordinates = pd.read_csv(genomap_coordinates_path)
genomap_coordinates.rename(columns={"Unnamed: 0": "gene_names"}, inplace=True)

# ----------------------------------------------------------------------------
# 9. Select Batches for Plotting
# ----------------------------------------------------------------------------

batches_provided = True

unique_combinations = obs[["diagnosis", "batch"]].drop_duplicates()

# Reset index if needed
unique_combinations = unique_combinations.reset_index(drop=True)

# Convert to dictionary where batch is the key and diagnosis is the single associated value
unique_dict = dict(zip(unique_combinations["batch"], unique_combinations["diagnosis"]))

print(unique_dict)

dict_batches = unique_dict
print("Batch dictionary:", dict_batches)

batches_to_select_from = list(dict_batches.keys())



# ----------------------------------------------------------------------------
# 10. Extract and Compare ASD vs Control
# ----------------------------------------------------------------------------

# Extract ASD and Control Keys
asd_keys = [key for key, value in dict_batches.items() if value == "ASD"]
asd_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in asd_keys]
asd = obs_multibatch.loc[(obs_multibatch["recon_prefix"].str.contains('batch')) & (obs_multibatch["recon_prefix"].isin(asd_recon_batch_list))]

control_keys = [key for key, value in dict_batches.items() if value == "Control"]
control_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in control_keys]
control = obs_multibatch.loc[(obs_multibatch["recon_prefix"].str.contains('batch')) & (obs_multibatch["recon_prefix"].isin(control_recon_batch_list))]

# Ensure matched pairs or equal shape
idx_asd = asd.index
idx_control = control.index



# ----------------------------------------------------------------------------
# 11. Mixed effects models (all pixels)
# ----------------------------------------------------------------------------


# Let's only do pixel (1,1) in a loop
i_values = list(range(0,54))
j_values = list(range(0,54))

all_pixel_results = []

for i in i_values:
    for j in j_values:
        # 1) Prepare container for cell-level data
        data_rows = []

        # 2) ASD recon_prefix
        for b in asd_recon_batch_list:
            row_inds = obs_multibatch.index[obs_multibatch['recon_prefix'] == b]
            pixel_vals = genomap[row_inds, i, j, 0]  # shape: (#cells_in_batch,)
            for val in pixel_vals:
                data_rows.append({
                    "batch": b,      
                    "group": "ASD",  
                    "value": val
                })

        # 3) Control recon_prefix
        for b in control_recon_batch_list:
            row_inds = obs_multibatch.index[obs_multibatch['recon_prefix'] == b]
            pixel_vals = genomap[row_inds, i, j, 0]
            for val in pixel_vals:
                data_rows.append({
                    "batch": b,
                    "group": "Control",
                    "value": val
                })

        # 4) Create a DataFrame for statsmodels
        df_ij = pd.DataFrame(data_rows)

        # 5) Fit Mixed-Effects Model: value ~ group + (1|batch)
        model_ij = smf.mixedlm("value ~ group", data=df_ij, groups=df_ij["batch"])
        fit_res_ij = model_ij.fit()


        # 5.1) Extract parameter estimates
        intercept = fit_res_ij.params["Intercept"]
        slope_control = fit_res_ij.params["group[T.Control]"]

        # 5.2) Extract p-values
        pval_intercept = fit_res_ij.pvalues["Intercept"]
        pval_control = fit_res_ij.pvalues["group[T.Control]"]

        # 5.3) Extract 95% confidence intervals
        ci = fit_res_ij.conf_int(alpha=0.05)
        intercept_lower_95CI, intercept_upper_95CI = ci.loc["Intercept"]
        slope_control_lower_95CI, slope_control_upper_95CI = ci.loc["group[T.Control]"]

        # 5.4) Store results
        all_pixel_results.append({
            "i": i,
            "j": j,
            "intercept": intercept,
            "slope_control": slope_control,
            "pval_intercept": pval_intercept,
            "pval_control": pval_control,
            "intercept_lower_95CI": intercept_lower_95CI,
            "intercept_upper_95CI": intercept_upper_95CI,
            "slope_control_lower_95CI": slope_control_lower_95CI,
            "slope_control_upper_95CI": slope_control_upper_95CI
        })

# ----------------------------------------------------------------------------
# 5.5. Compile results in a DataFrame
df_results_lmm = pd.DataFrame(all_pixel_results).sort_values("pval_control")
print(df_results_lmm)

# ----------------------------------------------------------------------------
# Merge coefficient, p-values, and CI columns with genomap_coordinates
# ----------------------------------------------------------------------------
genomap_coordinates = genomap_coordinates.merge(
    df_results_lmm[
        [
            "i", "j",
            "intercept", "slope_control",
            "pval_intercept", "pval_control",
            "intercept_lower_95CI", "intercept_upper_95CI",
            "slope_control_lower_95CI", "slope_control_upper_95CI",
        ]
    ],
    left_on=["pixel_i", "pixel_j"],
    right_on=["i", "j"],
    how="left"
)

# Rename the merged column to 'pval' to match the previous format
genomap_coordinates.rename(columns={'pval_control': 'pval'}, inplace=True)


# Drop the extra 'i' and 'j' columns from df_results_lmm
genomap_coordinates.drop(columns=['i', 'j'], inplace=True)
genomap_coordinates = genomap_coordinates.sort_values(by="pval")
p_threshold = 0.05
genomap_coordinates["significant"] = genomap_coordinates["pval"] < p_threshold

print("# Total Significant genes identified with mixed models:", len(genomap_coordinates[genomap_coordinates["significant"]]))


# Save the original data
genomap_coordinates.to_csv(os.path.join(out_name, "pvals_me.csv"))
