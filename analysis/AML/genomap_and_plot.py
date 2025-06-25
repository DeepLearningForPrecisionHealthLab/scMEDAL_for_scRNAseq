
import os
import sys
import gc
import shutil
import random

import pandas as pd
import numpy as np
import matplotlib

from typing import List, Optional
from scipy.stats import mannwhitneyu

# Set the Matplotlib backend to 'Agg'
matplotlib.use('Agg')

from utils.compare_results_utils import (
    get_recon_paths_df,
    get_input_paths_df
)
from utils.genomaps_utils import (
    select_cells_from_batches,
    find_intersection_batches,
    process_and_plot_genomaps_singlepath,
    create_count_matrix_multibatch,
    compute_cell_stats_acrossbatchrecon,
    plot_cell_recon_genomap
)
from utils.utils import read_adata


def genomap_and_plot(run_names_dict,
    results_path_dict,
    compare_models_path,
    data_base_path,
    scenario_id,
    input_base_path,
    path2results_file,
    scaling,
    types:Optional[List[str]]=None,
    splits:Optional[List[int]]=None,
    add_inputs_fe = False,
    extra_recon = "all", # "fe" or "all"
    seed = 42):
    
    # --------------------------------------------------------------------------------------
    # 3. Define variables
    # --------------------------------------------------------------------------------------
    celltype = ["Mono", "Mono-like"]
    n_cells_per_batch = 300
    n_batches = 19

    # Define number of genes in the genomap = colNum * rowNum
    n_genes = 2916
    colNum = 54
    rowNum = 54
    batches_to_select_from = ["AML420B", "BM5", "MUTZ3"]

    ### This is for AML
    gene_index_col = "Gene"

    # --------------------------------------------------------------------------------------
    # I run this script with Aixa_genomap env
    # --------------------------------------------------------------------------------------
    # 1. Get input paths and recon paths
    # --------------------------------------------------------------------------------------
    df_recon = get_recon_paths_df(results_path_dict, get_batch_recon_paths=True)
    df_inputs = get_input_paths_df(input_base_path)
    print(df_recon.columns, df_inputs.columns)
    print(f"splits path: {input_base_path}")
    df = pd.merge(df_recon, df_inputs, on=["Split", "Type"], how="left")
    df["recon_prefix"] = [
        recon_path.split("/")[-1].split(".npy")[0] for recon_path in df["ReconPath"]
    ]
    print("Reading paths,\ndf paths:", df.head(5))


    # --------------------------------------------------------------------------------------
    # 1.2. Base: I will run the genomap for random effects reconstructions: (scMEDAL-RE) outputs
    # --------------------------------------------------------------------------------------
    # Define lists of models, types, and splits. This script will only run one genomap.
    models = list(run_names_dict.keys())  # Add all your models to this list
    if types is None:
        types = ["train", "test", "val"]
    if splits is None:
        splits = list(range(1,6))             # Get data from fold 2

    for Type in types:
        for Split in splits:
            for model_name in [m for m in models if m != "run_name_all"]:
                # Select the first model, type, and split
                # Type = types[0]
                # Split = splits[0]
                # model_name = models[0]

                # RE recon path: The counterfactual batches are stored in this directory
                re_recon_path = os.path.join(results_path_dict[model_name], f"splits_{Split}")

                # Define experiment output directory
                out_name = os.path.join(compare_models_path, run_names_dict["run_name_all"])
                if not os.path.exists(out_name):
                    os.makedirs(out_name)
                print("Saving results to", out_name)


                # --------------------------------------------------------------------------------------
                # Copy paths_config.py and 'pipeline_CMmultibatch_genomap_and_plot_file.py' file
                # --------------------------------------------------------------------------------------
                # path2results_destination_path = os.path.join(out_name, 'paths_config.py')

                # print("\nCopying paths_config.py and pipeline_CMmultibatch_genomap_and_plot_file.py file to:", out_name)
                # pipeline_CMmultibatch_genomap_and_plot_file = os.path.abspath(__file__)
                # pipeline_CMmultibatch_destination_path = os.path.join(
                #     out_name, 'pipeline_CMmultibatch_genomap_and_plot_file.py'
                # )
                # # Copy the files
                # shutil.copy(path2results_file, path2results_destination_path)
                # shutil.copy(pipeline_CMmultibatch_genomap_and_plot_file, pipeline_CMmultibatch_destination_path)

                # --------------------------------------------------------------------------------------
                # Get batch reconstruction paths and prefix (to indicate batch)
                # --------------------------------------------------------------------------------------
                print(df.columns)
                print(df["Key"].unique())

                ####### IS this Correct ?!?!?!!?####### IS this Correct ?!?!?!!?####### IS this Correct ?!?!?!!?
                ####### IS this Correct ?!?!?!!?
                recon_paths = df.loc[
                    (df["Key"] == model_name) & (df["Split"] == Split) & 
                    (df["Type"] == Type), # & (df["recon_prefix"] != "recon_train"),
                    "ReconPath"
                ].values

                recon_prefix = df.loc[
                    (df["Key"] == model_name) & (df["Split"] == Split) & 
                    (df["Type"] == Type),# & (df["recon_prefix"] != "recon_train"),
                    "recon_prefix"
                ].values

                print("n recon paths:", len(recon_paths))

                # Get inputs path: Same for all models, split and type.
                inputs_path = df.loc[
                    (df["Key"] == model_name) & (df["Split"] == Split) & (df["Type"] == Type),
                    "InputsPath"
                ].values[0]
                ####### IS this Correct ?!?!?!!?####### IS this Correct ?!?!?!!?####### IS this Correct ?!?!?!!?
                #return df


                if extra_recon == "fe":
                    # n_inputs_fe = 2: One for inputs and another one for fixed effects (fe).
                    # You could also add a fe classifier recon and a base autoencoder recon.
                    n_inputs_fe = 2
                    # Get fixed effects path: unique for "scMEDAL-FE"
                    fe_ae_path = df.loc[
                        (df["Key"] == model_name) & (df["Split"] == Split) & (df["Type"] == Type),
                        "ReconPath"
                    ].values[0]
                    # Define extra paths and prefix
                    extra_paths = [inputs_path, fe_ae_path]
                    extra_prefix = [f"input_{Type}", f"fe_ae_recon_{Type}"]

                elif extra_recon == "all":
                    n_inputs_fe = 5
                    
                    mods = ["ae", "aec", "scmedalfe", "scmedalfec"]
                    extra_paths = []
                    for mod in mods:
                        try:
                            curr_path = df.loc[
                            (df["Key"] == mod) & (df["Split"] == Split) & (df["Type"] == Type),
                            "ReconPath"
                            ].values[0]
                            extra_paths.append(curr_path)
                        except IndexError:
                            pass
                  
                    extra_prefix = [
                        f"input_{Type}", f"ae_recon_{Type}", f"aec_recon_{Type}",
                        f"fe_ae_recon_{Type}", f"fe_aec_recon_{Type}"
                    ]

                # --------------------------------------------------------------------------------------
                # 2. Get genes and cells metadata
                # --------------------------------------------------------------------------------------
                # The splits did not store the real gene_ids. This changes on every experiment

                gene_ids_path = os.path.join(data_base_path, scenario_id, "geneids.csv")
                var = pd.read_csv(gene_ids_path, index_col=gene_index_col)

                # Get cell metadata (obs) from inputs_path (same for all models of same Type, split)
                _, _, obs = read_adata(inputs_path, issparse=True)



                if add_inputs_fe:
                    recon_prefix = extra_prefix + recon_prefix.tolist()
                    recon_paths = extra_paths + recon_paths.tolist()
                    n_cells = n_cells_per_batch * (n_batches + n_inputs_fe)
                else:
                    n_cells = n_cells_per_batch * n_batches

                # --------------------------------------------------------------------------------------
                # 4. Create multibatch count matrix for the genomap
                # --------------------------------------------------------------------------------------
                print("\nCreating count_matrix_multibatch..")

                                ############ Spliced over from MannU script 221...    
                # Determine patient group by batch
                batches_provided = True
                unique_combinations = obs[["Patient_group", "batch"]].drop_duplicates().reset_index(
                    drop=True
                )
                unique_dict = dict(zip(unique_combinations["batch"],
                                    unique_combinations["Patient_group"]))
                print(unique_dict)

                dict_batches = unique_dict
                print("Batch dictionary:", dict_batches)
                batches_to_select_from = list(dict_batches.keys())
            
                
                random.seed(seed)

                adata_multibatch_n_cells = create_count_matrix_multibatch(
                    recon_prefix,
                    recon_paths,
                    obs,
                    var,
                    n_genes=n_genes,
                    n_cells=n_cells_per_batch,
                    n_batches=n_batches,
                    out_path=out_name,
                    add_inputs_fe=n_inputs_fe if add_inputs_fe else None,
                    n_inputs_fe=n_inputs_fe,
                    celltype=celltype,
                    save_data=True,
                    scaling=scaling,
                    issparse=False,
                    seed=seed,
                    force_batches=batches_to_select_from,
                )

                adata_multibatch_n_cells.obs.index = adata_multibatch_n_cells.obs.index.astype(int)
                gc.collect()

                print("adata_multibatch_n_cells.X", adata_multibatch_n_cells.X.shape)
                print("adata_multibatch_n_cells.obs", adata_multibatch_n_cells.obs)
                print("adata_multibatch_n_cells.var", adata_multibatch_n_cells.var)

                if isinstance(celltype, str):
                    celltype_name = celltype.replace("/", "")
                elif isinstance(celltype, list):
                    celltype_name = "_".join([ct.replace("/", "") for ct in celltype])
                else:
                    celltype_name = None

                cm_multibatch_name = f"CMmultibatch_{n_cells_per_batch}_cells_per_batch_{n_batches}batches"
                if celltype_name:
                    cm_multibatch_name += f"_{celltype_name}"
                if add_inputs_fe:
                    cm_multibatch_name += f"_with_{n_inputs_fe}fe_input"

                cm_multibatch_path = os.path.join(out_name, cm_multibatch_name)
                print("cm_multibatch_path:", cm_multibatch_path)
                # --------------------------------------------------------------------------------------
                # 5. Get genomaps
                # --------------------------------------------------------------------------------------
                print("\nComputing genomap..")

                print("\nComputing genomap..")
                genomap_name = f"{n_cells_per_batch}cells_per_batch_{n_batches}batches_{Type}_{Split}"

                if celltype:
                    genomap_name += f"_{celltype_name}"
                if add_inputs_fe:
                    genomap_name += f"_with_{n_inputs_fe}fe_input"

                path_2_genomap = os.path.join(out_name, genomap_name)
                print("genomap stored in", path_2_genomap)

                gene_names = var.index[0:n_genes]

                
                try:
                    process_and_plot_genomaps_singlepath(
                        cm_multibatch_path,
                        ncells=n_cells,
                        ngenes=n_genes,
                        rowNum=rowNum,
                        colNum=colNum,
                        epsilon=0.0,
                        num_iter=100,
                        output_folder=path_2_genomap,
                        genomap_name=genomap_name,
                        gene_names=gene_names,
                    )
                    print("genomap stored in", path_2_genomap)
                    gc.collect()

                    # --------------------------------------------------------------------------------------
                    # 6. Plot genomaps
                    # --------------------------------------------------------------------------------------
                    order = "C"
                    statistic = "std"
                    if add_inputs_fe:
                        recon2plot = extra_prefix
                    else:
                        recon2plot = []

                    genomap_path = os.path.join(path_2_genomap, f"genomap_{genomap_name}.npy")
                    genomap_coordinates_path = os.path.join(
                        path_2_genomap,
                        f"gene_coordinates_{genomap_name}.csv"
                    )
                    print("genomap_path", genomap_path)

                    genomap = np.load(genomap_path)

                    genomap_coordinates = pd.read_csv(genomap_coordinates_path)
                    genomap_coordinates.rename(columns={"Unnamed: 0": "gene_names"}, inplace=True)

                    obs_multibatch = adata_multibatch_n_cells.obs
                    print("cm_multibatch_path", cm_multibatch_path)

                    cell_id_col = "Cell"
                    print("obs multibatch", obs_multibatch)

                    cell_ids_all = np.unique(obs_multibatch[cell_id_col].values)

                    cell_ids_2plot = []
                    if isinstance(celltype, list):
                        intersection_batches = find_intersection_batches(obs_multibatch, celltype)
                        print("Intersection of batches:", intersection_batches)
                        if batches_to_select_from is None:
                            batches_to_select_from = list(intersection_batches)[0:4]

                        print("selected celltypes and batches:", celltype, batches_to_select_from)
                        cell_ids_2plot = select_cells_from_batches(
                            obs_multibatch,
                            celltype,
                            batches_to_select_from,
                            seed=seed,
                            cell_id_col=cell_id_col,
                        )
                        n_batch_cols2plot = len(batches_to_select_from)
                    else:
                        n_cells_2_plot = 4
                        cell_ids_all = obs_multibatch[cell_id_col].values
                        cell_ids_2plot = random.sample(list(cell_ids_all), n_cells_2_plot)
                        n_batch_cols2plot = n_cells_2_plot

                    print("Selected cell IDs to plot:", cell_ids_2plot)

                    
                    original_batch_list = []
                    for cell_id in cell_ids_2plot:
                        original_batch = obs_multibatch.loc[
                            (obs_multibatch[cell_id_col] == cell_id)
                            & (obs_multibatch["recon_prefix"] == f"recon_{Type}"),
                            "batch",
                        ].values[0]
                        original_batch_list.append(original_batch)

                    recon2plot = recon2plot + original_batch_list

                    plot_min = -1
                    plot_max = 2

                    for cell_id in cell_ids_2plot:
                        print("cell_id", cell_id)

                        original_batch = obs_multibatch.loc[
                            (obs_multibatch[cell_id_col] == cell_id)
                           & (obs_multibatch["recon_prefix"] == f"recon_{Type}"),
                            "batch",
                        ].values[0]
                        print("original batch:", original_batch)

                        original_celltype = obs_multibatch.loc[
                            (obs_multibatch[cell_id_col] == cell_id)
                           & (obs_multibatch["recon_prefix"] == f"recon_{Type}"),
                            "celltype",
                        ].values[0]
                        print("original batch:", original_celltype)

                        cell_indexes = obs_multibatch.loc[
                            obs_multibatch[cell_id_col] == cell_id
                        ].index.values
                        cell_indexes = cell_indexes.astype(int)
                        print("n cell indexes", cell_indexes)

                        cell_indexes_batch_cf = obs_multibatch.loc[
                            (obs_multibatch[cell_id_col] == cell_id)
                            & (obs_multibatch["batch"])#.str.contains("batch"))
                            #& (obs_multibatch["recon_prefix"].str.contains("batch"))
                        ].index.values
                        cell_indexes_batch_cf = cell_indexes_batch_cf.astype(int)
                        print("n cell indexes for batch CF recon", cell_indexes_batch_cf)
                        print("obs_multibatch['recon_prefix']", obs_multibatch["recon_prefix"].values)

                        genomap_coordinates = compute_cell_stats_acrossbatchrecon(
                            genomap,
                            cell_indexes_batch_cf,
                            genomap_coordinates,
                            statistic=statistic,
                            n_top_genes=10,
                            order="C",
                            path_2_genomap=path_2_genomap,
                            file_name=cell_id,
                        )

                        print(genomap_coordinates[genomap_coordinates["Top_N"]])
                        plot_cell_recon_genomap(
                            genomap,
                            cell_indexes,
                            genomap_coordinates,
                            obs=obs_multibatch,
                            original_batch=original_batch,
                            n_top_genes=10,
                            min_val=plot_min,
                            max_val=plot_max,
                            order="C",
                            path_2_genomap=path_2_genomap,
                            file_name=f"{cell_id}_{statistic}_{original_celltype}",
                        )

                        cell_indexes_few_batches = obs_multibatch.loc[
                            (obs_multibatch[cell_id_col] == cell_id)
                            & obs_multibatch["recon_prefix"].apply(
                                lambda x: any(recon in x for recon in recon2plot)
                            )
                        ].index.values
                        # No gene labels
                        plot_cell_recon_genomap(
                            genomap,
                            cell_indexes=cell_indexes_few_batches,
                            genomap_coordinates=None,
                            obs=obs_multibatch,
                            original_batch=original_batch,
                            n_top_genes=10,
                            min_val=plot_min,
                            max_val=plot_max,
                            n_cols=n_batch_cols2plot + n_inputs_fe,
                            order="C",
                            path_2_genomap=path_2_genomap,
                            file_name=f"{cell_id}_few_batches_{statistic}_{original_celltype}",
                            remove_ticks=True,
                        )

                        plot_cell_recon_genomap(
                            genomap,
                            cell_indexes=cell_indexes_few_batches,
                            genomap_coordinates=genomap_coordinates,
                            obs=obs_multibatch,
                            original_batch=original_batch,
                            n_top_genes=10,
                            min_val=plot_min,
                            max_val=plot_max,
                            n_cols=n_batch_cols2plot + n_inputs_fe,
                            order="C",
                            path_2_genomap=path_2_genomap,
                            file_name=(
                                f"{cell_id}_few_batches_{statistic}_{original_celltype}_genelabels"
                            ),
                            remove_ticks=True,
                        )




                        # Obtain AML and control keys (and similarly for cell lines if needed)
                        aml_keys = [key for key, value in dict_batches.items() if value == "AML"]
                        aml_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in aml_keys]
                        aml = obs_multibatch.loc[
                            (obs_multibatch["recon_prefix"].str.contains('batch'))
                            & (obs_multibatch["recon_prefix"].isin(aml_recon_batch_list))
                        ]

                        control_keys = [key for key, value in dict_batches.items() if value == "control"]
                        control_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in control_keys]
                        control = obs_multibatch.loc[
                            (obs_multibatch["recon_prefix"].str.contains('batch'))
                            & (obs_multibatch["recon_prefix"].isin(control_recon_batch_list))
                        ]

                        cl_keys = [key for key, value in dict_batches.items() if value == "celline"]
                        cl_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in cl_keys]
                        cl = obs_multibatch.loc[
                            (obs_multibatch["recon_prefix"].str.contains('batch'))
                            & (obs_multibatch["recon_prefix"].isin(cl_recon_batch_list))
                        ]

                        # Matching indexes for AML and Control
                        idx_aml = aml.index
                        idx_control = control.index

                        # ----------------------------------------------------------------------------
                        # 11. Aggregate Genomaps per Batch
                        # ----------------------------------------------------------------------------
                        print("Aggregating Genomaps per Batch")
                        aml_avg_maps = []
                        for b in aml_recon_batch_list:
                            row_inds = obs_multibatch.index[obs_multibatch['recon_prefix'] == b]
                            batch_genomaps = genomap[row_inds, :, :, :]
                            avg_map = batch_genomaps.mean(axis=0)
                            aml_avg_maps.append(avg_map)

                        ctrl_avg_maps = []
                        for b in control_recon_batch_list:
                            row_inds = obs_multibatch.index[obs_multibatch['recon_prefix'] == b]
                            batch_genomaps = genomap[row_inds, :, :, :]
                            avg_map = batch_genomaps.mean(axis=0)
                            ctrl_avg_maps.append(avg_map)

                        aml_avg_maps = np.stack(aml_avg_maps, axis=0)    # shape (n_asd, 54, 54, 1)
                        ctrl_avg_maps = np.stack(ctrl_avg_maps, axis=0)  # shape (n_ctrl, 54, 54, 1)

                        print("ASD average maps shape:", aml_avg_maps.shape)
                        print("Control average maps shape:", ctrl_avg_maps.shape)

                        # ----------------------------------------------------------------------------
                        # 12. Perform Pixel-wise Mann-Whitney U Test
                        # ----------------------------------------------------------------------------
                        print("Perform Pixel-wise Mann-Whitney U Test")

                        _, height, width, _ = aml_avg_maps.shape
                        uvals = np.zeros((height, width))
                        pvals = np.zeros((height, width))

                        for i in range(height):
                            for j in range(width):
                                u_stat, p_value = mannwhitneyu(
                                    aml_avg_maps[:, i, j, 0],
                                    ctrl_avg_maps[:, i, j, 0],
                                    alternative="two-sided"
                                )
                                uvals[i, j] = u_stat
                                pvals[i, j] = p_value

                        # Process gene coordinates for significance
                        i_coords = genomap_coordinates["pixel_i"].values
                        j_coords = genomap_coordinates["pixel_j"].values
                        genomap_coordinates["pval"] = pvals[i_coords, j_coords]
                        genomap_coordinates = genomap_coordinates.sort_values(by="pval")

                        p_threshold = 0.05
                        genomap_coordinates["significant"] = (
                            genomap_coordinates["pval"] < p_threshold
                        )
                        print("\n# significant genes:",
                            len(genomap_coordinates[genomap_coordinates["significant"]]))
                        print(genomap_coordinates[genomap_coordinates["significant"]])


                        # Save final dataframe with p-values and gene annotations
                        genomap_coordinates.to_csv(
                            os.path.join(out_name, "pvals_300cellsavg_mwutest.csv")
                        )
                except: 
                    raise ValueError()
                    print("\n".join(["#"*50,"#"*50, "it broke", "#"*50,"#"*50]))