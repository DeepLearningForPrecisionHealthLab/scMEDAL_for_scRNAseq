
import os
import sys
import gc
import shutil
import random

import pandas as pd
import numpy as np
import matplotlib

from dataclasses import dataclass, field

import random

from typing import List, Optional, Dict, Tuple
from scipy.stats import mannwhitneyu

from inspect import currentframe, getframeinfo
def error_here(message:Optional[str]=None):
    frameinfo = getframeinfo(currentframe())
    print("ERROR HERE:", frameinfo.filename, frameinfo.lineno)
    print(message)

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




@dataclass
class GenomapConfig:
    # core paths
    compare_models_path: str
    data_base_path: str
    scenario_id: str
    input_base_path: str
    analysis_name: Optional[str] = None

    # tuning knobs
    celltype: Optional[List[str]] = None
    batches:  Optional[List[str]] = None
    # genomap
    n_cells_per_batch: int = 300
    n_batches: int = None
    n_genes: int = 2916
    n_col: int = 54
    n_row: int = 54
    num_iter: int = 100 

    cell_id_col : str = "Cell"
    gene_index_col: str = None
    scaling: str = "min_max"
    add_inputs_fe: bool = True
    extra_recon: str = "fe"   # "fe", "all", or "none"
    seed: int = 42
    n_cells_2_plot: int = 4
    n_top_genes : int = 10
    min_val : int = -1
    max_val : int = 2
    issparse: bool = False


    # derived
    n_inputs_fe: int = field(init=False)

    def __post_init__(self):
        self.n_inputs_fe = {"fe": 2, "all": 5}.get(self.extra_recon, 0)
        random.seed(self.seed)



class GenomapPipeline:
    """Pipeline that assembles count matrices, computes a Genomap, and plots."""

    def __init__(self, run_names_dict: Dict[str, str], results_path_dict: Dict[str, str], cfg: GenomapConfig):
        self.run_names_dict = run_names_dict
        self.results_path_dict = results_path_dict
        self.cfg = cfg
        self.df = self._load_and_merge_paths()
        print("\n\nInitialized genomap pipeline")
        print("sparse",cfg.issparse)

    #  internal helpers 

    def _load_and_merge_paths(self) -> pd.DataFrame:
        print(f"\nLooking for outputs paths for the following models:{self.results_path_dict.keys()}")
        df_recon = get_recon_paths_df(self.results_path_dict, get_batch_recon_paths=True)
        df_inputs = get_input_paths_df(self.cfg.input_base_path)
        df = pd.merge(df_recon, df_inputs, on=["Split", "Type"], how="left")
        df["recon_prefix"] = df["ReconPath"].apply(lambda p: os.path.basename(p).split(".npy")[0])
        print(f"Created df with input and recon paths\n{df}")
        return df

    def _select_recon(self, model: str, split: int, typ: str):
        mask = (self.df["Key"] == model) & (self.df["Split"] == split) & (self.df["Type"] == typ)
        return self.df.loc[mask, ["ReconPath", "recon_prefix"]].values.T  # 2×N arrays

    def _input_path(self, model: str, split: int, typ: str) -> str:
        print(f"Searching input paths in df\n{self.df}")
        m = (self.df["Key"] == model) & (self.df["Split"] == split) & (self.df["Type"] == typ)
        # Nothing matched, early warning
        if not m.any():
            print(f"[WARN] No row matches (Key, Split, Type)=({model}, {split}, '{typ}')")
            print("[INFO] Distinct triples present in df (first 20 shown):")
            print(self.df[['Key', 'Split', 'Type']].drop_duplicates().head(20))
            raise KeyError("InputsPath not found  see log above")
        inputs = self.df.loc[m, "InputsPath"].values
        # all models from same dataset have the same input
        return inputs[0]

    def _build_extra(self, split: int, typ: str):
        """Adds extra prefixes to df"""
        cfg = self.cfg
        if cfg.extra_recon == "fe":
            # Safety  check to make sure you have run scmedalfe
            assert "scmedalfe" in self.results_path_dict, (
                "extra_recon='fe' needs fixed-effects reconstructions "
                "(model key 'scmedalfe') in results_path_dict.")

            fe_mask = (
                (self.df["Key"] == "scmedalfe") & (self.df["Split"] == split) & (self.df["Type"] == typ)
            )
            fe_path = self.df.loc[fe_mask, "ReconPath"].values[0]
            return [self._input_path("scmedalfe", split, typ), fe_path], [f"input_{typ}", f"fe_ae_recon_{typ}"]

        if cfg.extra_recon == "all":
            mods = ["ae", "aec", "scmedalfe", "scmedalfec"]

            # Check every required key is present
            missing = [m for m in mods if m not in self.results_path_dict]

            if missing:                         # something is missing: fail fast
                raise KeyError(
                    "extra_recon='all' needs reconstructions for "
                    f"{', '.join(mods)}, but these are absent: {', '.join(missing)}"
                )

            extras: List[Tuple[str, str]] = []
            for m in mods:
                mask = (
                    (self.df["Key"] == m) & (self.df["Split"] == split) & (self.df["Type"] == typ)
                )
                if mask.any():
                    path = self.df.loc[mask, "ReconPath"].values[0]
                    extras.append((path, f"{m}_recon_{typ}".replace("scmedal", "")))
            extras.insert(0, (self._input_path(mods[0], split, typ), f"input_{typ}"))
            if not extras:
                return [], []
            paths, prefixes = zip(*extras)
            return list(paths), list(prefixes)
        return [], []

    def _load_meta(self, inputs_path: str):
        cfg = self.cfg
        gene_ids_path = os.path.join(self.cfg.data_base_path, self.cfg.scenario_id, "geneids.csv")
        var = pd.read_csv(gene_ids_path, index_col=self.cfg.gene_index_col)
        _, _, obs = read_adata(inputs_path, issparse=cfg.issparse)
        return var, obs

    def _build_multibatch(
            self,
            recon_paths: List[str],
            recon_prefix: List[str],
            obs: pd.DataFrame,
            var: pd.DataFrame,
            batches_to_select_from: List[str],
            out_dir: str,
        ):
        cfg = self.cfg

        # Build the concatenated matrix
        adata_mb = create_count_matrix_multibatch(
            recon_prefix,
            recon_paths,
            obs,
            var,
            n_genes     = cfg.n_genes,
            n_cells     = cfg.n_cells_per_batch,
            n_batches   = cfg.n_batches,
            out_path    = out_dir,
            add_inputs_fe = cfg.n_inputs_fe if cfg.add_inputs_fe else None,
            n_inputs_fe = cfg.n_inputs_fe,
            celltype    = cfg.celltype,
            save_data   = True,
            scaling     = cfg.scaling,
            issparse    = cfg.issparse,
            seed        = cfg.seed,
            force_batches = batches_to_select_from,
        )

        # --- NEW: make sure obs index is numeric so .iloc works safely ----------
        adata_mb.obs.index = adata_mb.obs.index.astype(int)

        return adata_mb
    def _compute_genomap(
        self,
        cm_path: str,
        genomap_name: str,
        out_dir: str,
        gene_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Build / load the Genomap for one (type, split, model) combination
        and return it as a NumPy array.
        """
        cfg = self.cfg
        # how many extra reconstructions were appended?
        extra = cfg.n_inputs_fe if cfg.add_inputs_fe else 0          # 0, 2 or 5

        # real number of cell slices in the multibatch matrix
        ncells_for_genomap = cfg.n_cells_per_batch * (cfg.n_batches + extra)

        process_and_plot_genomaps_singlepath(
            cm_path,
            ncells   = ncells_for_genomap,
            ngenes   = cfg.n_genes,
            rowNum   = cfg.n_row,
            colNum   = cfg.n_col,
            epsilon  = 0.0,
            num_iter = cfg.num_iter,
            output_folder = out_dir,
            genomap_name  = genomap_name,
            gene_names    = gene_names,   # <-- optional; pass-through
        )

        return np.load(os.path.join(out_dir, f"genomap_{genomap_name}.npy"))
    def _load_gene_coordinates(self, out_dir: str, gname: str) -> pd.DataFrame:
        """
        Load `gene_coordinates_{gname}.csv` and guarantee that the returned
        dataframe **has a column called `gene_names`** and a plain RangeIndex.
        """
        path = os.path.join(out_dir, f"gene_coordinates_{gname}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected {path} but it was not written.")

        # --- keep column, keep RangeIndex ---------------------------------
        coords = pd.read_csv(path)                       # <-- no index_col
        coords.rename(columns={"Unnamed: 0": "gene_names"}, inplace=True)
        # ------------------------------------------------------------------

        # optional sanity check
        if "gene_names" not in coords.columns:
            raise ValueError("`gene_names` column missing after load/rename.")

        return coords


    def _extra_prefixes(self, typ: str) -> List[str]:
        if not self.cfg.add_inputs_fe:
            return []
        if self.cfg.extra_recon == "fe":            # default
            return [f"input_{typ}", f"fe_ae_recon_{typ}"]
        return [
            f"input_{typ}", f"ae_recon_{typ}", f"aec_recon_{typ}",
            f"fe_ae_recon_{typ}", f"fe_aec_recon_{typ}",
        ]
    def _choose_cells_and_batches(
        self,
        adata,
        typ:   str,
        batches_to_select_from: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
         Pick cell IDs exactly like the old notebook:
               If celltype is given, keep one cell of each cell-type
                per *intersection* batch (or forced `batches_to_choose_from`)
               Otherwise pick 4 random cells.
         Return both the picked cell IDs *and* the list with the
          corresponding original batch (i.e. the batch attached to
          'input_<typ>' for that cell).  This list is later merged into
          `recon2plot` so the panels include the original recon.
        """
        cfg = self.cfg
        n_cells_2_plot = cfg.n_cells_2_plot
        obs = adata.obs
        rng = random.Random(cfg.seed)

        # ---- determine batches that contain *all* requested cell-types ----
        if cfg.celltype:
            print("batches 2 select from",batches_to_select_from)
            inter = find_intersection_batches(obs, cfg.celltype)



            if batches_to_select_from is None:
                batches_to_select_from = list(inter)[:4]
                print(f"find batches within {cfg.celltype}:{inter}")

            cell_ids = select_cells_from_batches(
                obs,
                cfg.celltype,
                batches_to_select_from,
                seed = cfg.seed,
                cell_id_col = cfg.cell_id_col,
            )
            n_batch_cols2plot = len(batches_to_select_from)
        else:
            print("n_cells_2_plot:",n_cells_2_plot)
            cell_ids_all = obs[cfg.cell_id_col].values
            cell_ids = random.sample(list(cell_ids_all), n_cells_2_plot)
            n_batch_cols2plot = n_cells_2_plot

        # ---- collect original batch per picked cell ----------------------
        original_batches = []
        for cid in cell_ids:
            b = obs.loc[
                (obs[cfg.cell_id_col] == cid)
                & (obs["recon_prefix"] == f"input_{typ}"),
                "batch",
            ].values[0]
            original_batches.append(b)
            print(f"{cid} comes from batch {b}")

        return cell_ids, original_batches, n_batch_cols2plot


    def _write_cell_stats(
        self,
        genomap: np.ndarray,
        idxs: np.ndarray,
        coords: pd.DataFrame,
        out_dir: str,
        cell_id: str,
    ):
        cfg = self.cfg
        compute_cell_stats_acrossbatchrecon(
            genomap, idxs, coords,
            statistic      = "std",
            n_top_genes    = cfg.n_top_genes,
            order          = "C",
            path_2_genomap = out_dir,
            file_name      = cell_id.replace("/",""),
        )


    # ------------------------------------------------------------------
    # Helper 2  big panel with *all* reconstructions
    # ------------------------------------------------------------------
    def _plot_big_panel(self,
        genomap: np.ndarray,
        idxs: np.ndarray,
        coords: pd.DataFrame,
        obs_df: pd.DataFrame,
        out_dir: str,
        cell_id: str,
        original_batch: str,
        original_ct: str,
    ):
        cfg = self.cfg
        filename = f"{cell_id}_std_{original_ct}"
        plot_cell_recon_genomap(
            genomap,
            idxs,
            coords,
            obs             = obs_df,
            original_batch  = original_batch,
            n_top_genes     = cfg.n_top_genes,
            min_val         = cfg.min_val,
            max_val         = cfg.max_val,
            order           = "C",
            path_2_genomap  = out_dir,
            file_name       = filename,
        )


    # ------------------------------------------------------------------
    # Helper 3  small subset panel (optionally with gene labels)
    # ------------------------------------------------------------------
    def _plot_subset_panel(self,
        genomap: np.ndarray,
        idxs_subset: np.ndarray,
        coords: Optional[pd.DataFrame] ,
        obs_df: pd.DataFrame,
        out_dir: str,
        cell_id: str,
        original_batch: str,
        original_ct: str,
        n_cols: int,
        with_labels: bool,
    ):
    
        cfg = self.cfg
        file_stub = f"{cell_id}_few_batches_std_{original_ct}"
        if with_labels:
            file_stub += "_genelabels"

        plot_cell_recon_genomap(
            genomap,
            cell_indexes        = idxs_subset,
            genomap_coordinates = coords if with_labels else None,
            obs                 = obs_df,
            original_batch      = original_batch,
            n_top_genes         = cfg.n_top_genes,
            min_val             = cfg.min_val,
            max_val             = cfg.max_val,
            n_cols              = n_cols,
            order               = "C",
            path_2_genomap      = out_dir,
            file_name           = file_stub,
            remove_ticks        = True,
        )
        matplotlib.pyplot.close()




    def _plot_panels(
        self,
        genomap: np.ndarray,
        obs_df:  pd.DataFrame,
        coords:  pd.DataFrame,
        out_dir: str,
        typ:     str,
        cell_ids: List[str],
        original_batch_list: List[str],
        n_batch_cols2plot: int,
    ):
        """Create (i) a big panel with *all* recons and
           (ii) two small subset panels per cell, exactly like the
           legacy notebook."""
        # ----------------------------------------------------
        # recon2plot  =   [extra_prefixes]  +  original_batch_list
        # (mirrors the old `recon2plot = recon2plot + original_batch_list`)
        # ----------------------------------------------------
        cfg = self.cfg

        extra_pref   = self._extra_prefixes(typ)
        recon2plot   = extra_pref + original_batch_list
        n_inputs_fe  = len(extra_pref)

        for cid in cell_ids:


            # get the all the indexes of the cm multibatch that have the same cid (belong to the same cell and are recon from diff batches)
            idxs_all = obs_df.loc[obs_df[cfg.cell_id_col] == cid].index.astype(int)
            print("n cell indexes", idxs_all)

            if len(idxs_all) == 0:
                continue

            original_batch = obs_df.loc[idxs_all[0], "batch"]
            original_ct    = obs_df.loc[idxs_all[0], "celltype"]

            # -------- statistics CSV -------------------------------------
            coords_work = coords.reset_index(drop=True)
            self._write_cell_stats(genomap, idxs_all, coords_work, out_dir, cid)
            print(f"\n\nplotting big panel for {cid} with original batch {original_batch} and celltype {original_ct}, n recons: {len(idxs_all)}")
            # -------- big panel ------------------------------------------
            self._plot_big_panel(
                genomap, idxs_all, coords_work,
                obs_df, out_dir, cid, original_batch, original_ct
            )

            # -------- subset of reconstructions --------------------------
            subset_labels = set(recon2plot)
            idxs_subset = obs_df.loc[
                (obs_df[cfg.cell_id_col] == cid)
                & obs_df["recon_prefix"].apply(
                    lambda x: any(lbl in x for lbl in subset_labels)
                )
            ].index.astype(int)
            print("n cell indexes for batch CF recon", idxs_subset)
            if len(idxs_subset) == 0:
                continue

            #n_batch_cols  = len(obs_df.loc[idxs_subset, "batch"].unique())
            n_cols_subset = max(1, n_batch_cols2plot + n_inputs_fe)

            print(f"plotting small panel for {cid} with original batch {original_batch} and celltype {original_ct}, n recons: {len(idxs_subset)}")
            # (a) without gene labels
            self._plot_subset_panel(
                genomap, idxs_subset, None,
                obs_df, out_dir, cid, original_batch, original_ct,
                n_cols_subset, with_labels=False
            )
            # (b) with gene labels
            print(f"plotting same panel but with no gene labels")
            self._plot_subset_panel(
                genomap, idxs_subset, coords_work,
                obs_df, out_dir, cid, original_batch, original_ct,
                n_cols_subset, with_labels=True
            )

    def run(
        self,
        models: Optional[List[str]] = None,
        types:  Optional[List[str]] = None,
        splits: Optional[List[int]] = None,
    ) -> List[dict]:
        """Execute full pipeline and return a summary for every model/type/split."""

        cfg     = self.cfg
        models  = models 
        types   = types  or ["train", "test", "val"]
        splits  = splits or list(range(1, 6))
        summaries: List[dict] = []

        # helper: build consistent names 
        def _make_cm_and_gnames(typ: str, split: int, out_dir: str) -> Tuple[str, str, str]:
            """Return (cm_name, cm_path, gname) exactly like the writer produces."""
            # cell-type suffix (if any)
            if isinstance(cfg.celltype, list):
                cts = "_".join(ct.replace("/", "") for ct in cfg.celltype)
            elif isinstance(cfg.celltype, str):
                cts = cfg.celltype.replace("/", "")
            else:
                cts = ""

            # CM folder name
            cm_name = (
                f"CMmultibatch_{cfg.n_cells_per_batch}_cells_per_batch_"
                f"{cfg.n_batches}batches"
            )
            if cts:
                cm_name += f"_{cts}"
            if cfg.add_inputs_fe:
                cm_name += f"_with_{cfg.n_inputs_fe}fe_input"
            cm_path = os.path.join(out_dir, cm_name)

            # Genomap prefix
            gname = (
                f"{cfg.n_cells_per_batch}cells_per_batch_{cfg.n_batches}batches_"
                f"{typ}_{split}"
            )
            if cts:
                gname += f"_{cts}"
            if cfg.add_inputs_fe:
                gname += f"_with_{cfg.n_inputs_fe}fe_input"

            return cm_name, cm_path, gname
        # calling genomap function
        print("Computing genomap for",types,splits,models)
        for typ in types:
            for split in splits:
                for model in models:

                    print(typ,split,model)
                    # 1. output directory -------------------------------------------------
                    out_dir = os.path.join(cfg.compare_models_path,
                                        cfg.analysis_name or "genomap")
                    os.makedirs(out_dir, exist_ok=True)

                    # 2. recon paths ------------------------------------------------------
                    paths, prefixes = self._select_recon(model, split, typ)
                    paths, prefixes = list(paths), list(prefixes)
                    extra_paths, extra_pref = self._build_extra(split, typ)
                    if cfg.add_inputs_fe:
                        paths    = extra_paths    + paths
                        prefixes = extra_pref     + prefixes

                    # 3. metadata ---------------------------------------------------------
                    inputs_path   = self._input_path(model, split, typ)
                    var, obs      = self._load_meta(inputs_path)
                    # ensure the column exists *before* we hand `var` to the matrix builder
                    if "gene_names" not in var.columns:        # <- safe if you rerun
                        var = var.copy()                       #   (avoid SettingWithCopy)
                        var["gene_names"] = var.index

                    # 4. batch selection --------------------------------------------------
                    batches_sel = cfg.batches or list(obs["batch"].unique())
                    print("batch_sel",batches_sel)


                    # 5. multibatch matrix ----------------------------------------------
                    adata_mb = self._build_multibatch(
                        paths, prefixes, obs, var, batches_sel, out_dir
                    )
                    adata_mb.var["gene_names"] = adata_mb.var.index 

                    # choose cells and original batches
                    cell_ids, original_batch_list, n_batch_cols2plot = self._choose_cells_and_batches(adata_mb, typ, batches_to_select_from =batches_sel)
                    print("original batch list:",original_batch_list)

                    cm_name, cm_path, gname = _make_cm_and_gnames(typ, split, out_dir)

                    # adjust ncells 
                    extra = cfg.n_inputs_fe if cfg.add_inputs_fe else 0
                    ncells_for_genomap = cfg.n_cells_per_batch * (cfg.n_batches + extra)

                    genomap = self._compute_genomap(
                        cm_path, gname, out_dir, gene_names=var.index[:cfg.n_genes]
                    )
                    coords = self._load_gene_coordinates(out_dir, gname)



                    # 8. plot panels ------------------------------------------------------
                    self._plot_panels(genomap, adata_mb.obs, coords, out_dir, typ, cell_ids, original_batch_list,n_batch_cols2plot)

                    # 9. summarise --------------------------------------------------------
                    summaries.append(
                        {
                            "model":   model,
                            "type":    typ,
                            "split":   split,
                            "genomap": genomap,
                            "adata_mb": adata_mb,
                            "inputs_path": inputs_path,
                            "batches_to_select_from": batches_sel,
                            "cell_ids_to_plot": cell_ids,
                            "original_batch_list":original_batch_list,
                            "out_dir": out_dir,
                        }
                    )

                    del adata_mb, genomap
                    gc.collect()

        return summaries
    
def genomap_and_plot(run_names_dict, results_path_dict, cfg: GenomapConfig, **kwargs):
    pipeline = GenomapPipeline(run_names_dict, results_path_dict, cfg)
    return pipeline.run(**kwargs)


# def genomap_and_plot(
#     run_names_dict,
#     results_path_dict,
#     compare_models_path,
#     data_base_path,
#     scenario_id,
#     input_base_path,
#     analysis_name:Optional[str]=None,
#     celltype:Optional[List[str]]=None,
#     batches:Optional[List[str]]=None,
#     n_cells_per_batch:int=300,
#     n_batches:int=19,
#     n_genes:int=2916,
#     n_col:int=54,
#     n_row:int=54,
#     gene_index_col:str="Gene",

#     scaling:str="min_max",
#     models:Optional[List[str]]=None,
#     types:Optional[List[str]]=None,
#     splits:Optional[List[int]]=None,
#     add_inputs_fe:bool = False,
#     extra_recon:str = "all", # "fe" or "all"
#     seed:int=42
    
#     ):
    
#     # --------------------------------------------------------------------------------------
#     # I run this script with Aixa_genomap env
#     # --------------------------------------------------------------------------------------
#     # 1. Get input paths and recon paths
#     # --------------------------------------------------------------------------------------
#     df_recon = get_recon_paths_df(results_path_dict, get_batch_recon_paths=True)
#     df_inputs = get_input_paths_df(input_base_path)
#     print(df_recon.columns, df_inputs.columns)
#     print(f"splits path: {input_base_path}")
#     df = pd.merge(df_recon, df_inputs, on=["Split", "Type"], how="left")
#     df["recon_prefix"] = [
#         recon_path.split("/")[-1].split(".npy")[0] for recon_path in df["ReconPath"]
#     ]
#     print("Reading paths,\ndf paths:", df.head(5))

#     # --------------------------------------------------------------------------------------
#     # 1.2. Base: I will run the genomap for random effects reconstructions: (scMEDAL-RE) outputs
#     # --------------------------------------------------------------------------------------
#     # Define lists of models, types, and splits. This script will only run one genomap.
#     if models is None:
#         models = list(run_names_dict.keys())  # Add all your models to this list
#     if types is None:
#         types = ["train", "test", "val"]
#     if splits is None:
#         splits = list(range(1,6))             # Get data from fold 2

#     for Type in types:
#         for Split in splits:
#             for model_name in models:
#                 # Define experiment output directory
#                 out_name = os.path.join(compare_models_path, analysis_name)
#                 if not os.path.exists(out_name):
#                     os.makedirs(out_name)
#                 print("Saving results to", out_name)

#                 # --------------------------------------------------------------------------------------
#                 # Get batch reconstruction paths and prefix (to indicate batch)
#                 # --------------------------------------------------------------------------------------
#                 print(df.columns)
#                 print(df["Key"].unique())

#                 ####### IS this Correct ?!?!?!!?####### IS this Correct ?!?!?!!?####### IS this Correct ?!?!?!!?
#                 ####### IS this Correct ?!?!?!!?
#                 recon_paths = df.loc[
#                     (df["Key"] == model_name) & (df["Split"] == Split) & 
#                     (df["Type"] == Type), # & (df["recon_prefix"] != "recon_train"),
#                     "ReconPath"
#                 ].values

#                 recon_prefix = df.loc[
#                     (df["Key"] == model_name) & (df["Split"] == Split) & 
#                     (df["Type"] == Type),# & (df["recon_prefix"] != "recon_train"),
#                     "recon_prefix"
#                 ].values

#                 print("n recon paths:", len(recon_paths))

#                 # Get inputs path: Same for all models, split and type.
#                 inputs_path = df.loc[
#                     (df["Key"] == model_name) & (df["Split"] == Split) & (df["Type"] == Type),
#                     "InputsPath"
#                 ].values[0]
#                 ####### IS this Correct ?!?!?!!?####### IS this Correct ?!?!?!!?####### IS this Correct ?!?!?!!?
#                 #return df


#                 if extra_recon == "fe":
#                     # n_inputs_fe = 2: One for inputs and another one for fixed effects (fe).
#                     # You could also add a fe classifier recon and a base autoencoder recon.
#                     n_inputs_fe = 2
#                     # Get fixed effects path: unique for "scMEDAL-FE"
#                     fe_ae_path = df.loc[
#                         (df["Key"] == model_name) & (df["Split"] == Split) & (df["Type"] == Type),
#                         "ReconPath"
#                     ].values[0]
#                     # Define extra paths and prefix
#                     extra_paths = [inputs_path, fe_ae_path]
#                     extra_prefix = [f"input_{Type}", f"fe_ae_recon_{Type}"]

#                 elif extra_recon == "all":
#                     n_inputs_fe = 5
                    
#                     mods = ["ae", "aec", "scmedalfe", "scmedalfec"]
#                     extra_paths = []
#                     for mod in mods:
#                         try:
#                             curr_path = df.loc[
#                             (df["Key"] == mod) & (df["Split"] == Split) & (df["Type"] == Type),
#                             "ReconPath"
#                             ].values[0]
#                             extra_paths.append(curr_path)
#                         except IndexError:
#                             pass
                  
#                     extra_prefix = [
#                         f"input_{Type}", f"ae_recon_{Type}", f"aec_recon_{Type}",
#                         f"fe_ae_recon_{Type}", f"fe_aec_recon_{Type}"
#                     ]

#                 # --------------------------------------------------------------------------------------
#                 # 2. Get genes and cells metadata
#                 # --------------------------------------------------------------------------------------
#                 # The splits did not store the real gene_ids. This changes on every experiment

#                 gene_ids_path = os.path.join(data_base_path, scenario_id, "geneids.csv")
#                 var = pd.read_csv(gene_ids_path, index_col=gene_index_col)

#                 # Get cell metadata (obs) from inputs_path (same for all models of same Type, split)
#                 _, _, obs = read_adata(inputs_path, issparse=True)



#                 if add_inputs_fe:
#                     recon_prefix = extra_prefix + recon_prefix.tolist()
#                     recon_paths = extra_paths + recon_paths.tolist()
#                     n_cells = n_cells_per_batch * (n_batches + n_inputs_fe)
#                 else:
#                     n_cells = n_cells_per_batch * n_batches

#                 # --------------------------------------------------------------------------------------
#                 # 4. Create multibatch count matrix for the genomap
#                 # --------------------------------------------------------------------------------------
#                 print("\nCreating count_matrix_multibatch..")

#                 ############ Spliced over from MannU script 221...    
#                 # Determine patient group by batch
#                 if batches is None:
#                     unique_combinations = obs[["Patient_group", "batch"]].drop_duplicates().reset_index(
#                         drop=True
#                     )
#                     unique_dict = dict(zip(unique_combinations["batch"],
#                                         unique_combinations["Patient_group"]))
#                     print(unique_dict)

#                     dict_batches = unique_dict
#                     print("Batch dictionary:", dict_batches)
#                     batches_to_select_from = list(dict_batches.keys())
#                 else:
#                     batches_to_select_from = batches    
                
#                 random.seed(seed)

#                 adata_multibatch_n_cells = create_count_matrix_multibatch(
#                     recon_prefix,
#                     recon_paths,
#                     obs,
#                     var,
#                     n_genes=n_genes,
#                     n_cells=n_cells_per_batch,
#                     n_batches=n_batches,
#                     out_path=out_name,
#                     add_inputs_fe=n_inputs_fe if add_inputs_fe else None,
#                     n_inputs_fe=n_inputs_fe,
#                     celltype=celltype,
#                     save_data=True,
#                     scaling=scaling,
#                     issparse=False,
#                     seed=seed,
#                     force_batches=batches_to_select_from,
#                 )

#                 adata_multibatch_n_cells.obs.index = adata_multibatch_n_cells.obs.index.astype(int)
#                 gc.collect()

#                 print("adata_multibatch_n_cells.X", adata_multibatch_n_cells.X.shape)
#                 print("adata_multibatch_n_cells.obs", adata_multibatch_n_cells.obs)
#                 print("adata_multibatch_n_cells.var", adata_multibatch_n_cells.var)

#                 if isinstance(celltype, str):
#                     celltype_name = celltype.replace("/", "")
#                 elif isinstance(celltype, list):
#                     celltype_name = "_".join([ct.replace("/", "") for ct in celltype])
#                 else:
#                     celltype_name = None

#                 cm_multibatch_name = f"CMmultibatch_{n_cells_per_batch}_cells_per_batch_{n_batches}batches"
#                 if celltype_name:
#                     cm_multibatch_name += f"_{celltype_name}"
#                 if add_inputs_fe:
#                     cm_multibatch_name += f"_with_{n_inputs_fe}fe_input"

#                 cm_multibatch_path = os.path.join(out_name, cm_multibatch_name)
#                 print("cm_multibatch_path:", cm_multibatch_path)
#                 # --------------------------------------------------------------------------------------
#                 # 5. Get genomaps
#                 # --------------------------------------------------------------------------------------
#                 print("\nComputing genomap..")

#                 print("\nComputing genomap..")
#                 genomap_name = f"{n_cells_per_batch}cells_per_batch_{n_batches}batches_{Type}_{Split}"

#                 if celltype:
#                     genomap_name += f"_{celltype_name}"
#                 if add_inputs_fe:
#                     genomap_name += f"_with_{n_inputs_fe}fe_input"

#                 path_2_genomap = os.path.join(out_name, genomap_name)
#                 print("genomap stored in", path_2_genomap)

#                 gene_names = var.index[0:n_genes]

                
#                 try:
#                     process_and_plot_genomaps_singlepath(
#                         cm_multibatch_path,
#                         ncells=n_cells,
#                         ngenes=n_genes,
#                         rowNum=n_row,
#                         colNum=n_col,
#                         epsilon=0.0,
#                         num_iter=100,
#                         output_folder=path_2_genomap,
#                         genomap_name=genomap_name,
#                         gene_names=gene_names,
#                     )
#                     print("genomap stored in", path_2_genomap)
#                     gc.collect()

#                     # --------------------------------------------------------------------------------------
#                     # 6. Plot genomaps
#                     # --------------------------------------------------------------------------------------
#                     order = "C"
#                     statistic = "std"
#                     if add_inputs_fe:
#                         recon2plot = extra_prefix
#                     else:
#                         recon2plot = []

#                     genomap_path = os.path.join(path_2_genomap, f"genomap_{genomap_name}.npy")
#                     genomap_coordinates_path = os.path.join(
#                         path_2_genomap,
#                         f"gene_coordinates_{genomap_name}.csv"
#                     )
#                     print("genomap_path", genomap_path)

#                     genomap = np.load(genomap_path)

#                     genomap_coordinates = pd.read_csv(genomap_coordinates_path)
#                     genomap_coordinates.rename(columns={"Unnamed: 0": "gene_names"}, inplace=True)

#                     obs_multibatch = adata_multibatch_n_cells.obs
#                     print("cm_multibatch_path", cm_multibatch_path)

#                     cell_id_col = "Cell"
#                     print("obs multibatch", obs_multibatch)

#                     cell_ids_all = np.unique(obs_multibatch[cell_id_col].values)

        
#                     cell_ids_2plot = []
#                     if isinstance(celltype, list):
#                         intersection_batches = find_intersection_batches(obs_multibatch, celltype)
#                         print("Intersection of batches:", intersection_batches)
#                         if batches_to_select_from is None:
#                             batches_to_select_from = list(intersection_batches)[0:4]

#                         print("selected celltypes and batches:", celltype, batches_to_select_from)
#                         cell_ids_2plot = select_cells_from_batches(
#                             obs_multibatch,
#                             celltype,
#                             batches_to_select_from,
#                             seed=seed,
#                             cell_id_col=cell_id_col,
#                         )
#                         n_batch_cols2plot = len(batches_to_select_from)
#                     else:
#                         n_cells_2_plot = 4
#                         cell_ids_all = obs_multibatch[cell_id_col].values
#                         cell_ids_2plot = random.sample(list(cell_ids_all), n_cells_2_plot)
#                         n_batch_cols2plot = n_cells_2_plot

#                     print("Selected cell IDs to plot:", cell_ids_2plot)

#                     original_batch_list = []
#                     for cell_id in cell_ids_2plot:
#                         original_batch = obs_multibatch.loc[
#                             (obs_multibatch[cell_id_col] == cell_id)
#                             & (obs_multibatch["recon_prefix"] == f"recon_{Type}"),
#                             "batch",
#                         ].values[0]
#                         original_batch_list.append(original_batch)

#                     recon2plot = recon2plot + original_batch_list

#                     plot_min = -1
#                     plot_max = 2

#                     for cell_id in cell_ids_2plot:
#                         print("cell_id", cell_id)

#                         original_batch = obs_multibatch.loc[
#                             (obs_multibatch[cell_id_col] == cell_id)
#                            & (obs_multibatch["recon_prefix"] == f"recon_{Type}"),
#                             "batch",
#                         ].values[0]
#                         print("original batch:", original_batch)

#                         original_celltype = obs_multibatch.loc[
#                             (obs_multibatch[cell_id_col] == cell_id)
#                            & (obs_multibatch["recon_prefix"] == f"recon_{Type}"),
#                             "celltype",
#                         ].values[0]
#                         print("original batch:", original_celltype)

#                         cell_indexes = obs_multibatch.loc[
#                             obs_multibatch[cell_id_col] == cell_id
#                         ].index.values
#                         cell_indexes = cell_indexes.astype(int)
#                         print("n cell indexes", cell_indexes)

#                         cell_indexes_batch_cf = obs_multibatch.loc[
#                             (obs_multibatch[cell_id_col] == cell_id)
#                             & (obs_multibatch["batch"])#.str.contains("batch"))
#                             #& (obs_multibatch["recon_prefix"].str.contains("batch"))
#                         ].index.values
#                         cell_indexes_batch_cf = cell_indexes_batch_cf.astype(int)
#                         print("n cell indexes for batch CF recon", cell_indexes_batch_cf)
#                         print("obs_multibatch['recon_prefix']", obs_multibatch["recon_prefix"].values)

#                         ###### Maybe there is also an error here??
#                         genomap_coordinates = compute_cell_stats_acrossbatchrecon(
#                             genomap,
#                             cell_indexes_batch_cf,
#                             genomap_coordinates,
#                             statistic=statistic,
#                             n_top_genes=10,
#                             order="C",
#                             path_2_genomap=path_2_genomap,
#                             file_name=cell_id,
#                         )

#                         print(genomap_coordinates[genomap_coordinates["Top_N"]])
#                         plot_cell_recon_genomap(
#                             genomap,
#                             cell_indexes,
#                             genomap_coordinates,
#                             obs=obs_multibatch,
#                             original_batch=original_batch,
#                             n_top_genes=10,
#                             min_val=plot_min,
#                             max_val=plot_max,
#                             order="C",
#                             path_2_genomap=path_2_genomap,
#                             file_name=f"{cell_id}_{statistic}_{original_celltype}",
#                         )

#                         # This is A bug. This is giving an empty array.
#                         cell_indexes_few_batches = obs_multibatch.loc[
#                             (obs_multibatch[cell_id_col] == cell_id)
#                             & obs_multibatch["recon_prefix"].apply(
#                                 lambda x: any(recon in x for recon in recon2plot)
#                             )
#                         ].index.values
#                         # No gene labels
#                         try:
#                             plot_cell_recon_genomap(
#                                 genomap,
#                                 cell_indexes=cell_indexes_few_batches,
#                                 genomap_coordinates=None,
#                                 obs=obs_multibatch,
#                                 original_batch=original_batch,
#                                 n_top_genes=10,
#                                 min_val=plot_min,
#                                 max_val=plot_max,
#                                 n_cols=n_batch_cols2plot + n_inputs_fe,
#                                 order="C",
#                                 path_2_genomap=path_2_genomap,
#                                 file_name=f"{cell_id}_few_batches_{statistic}_{original_celltype}",
#                                 remove_ticks=True,
#                             )
#                         except Exception as e: 
#                             error_here()
#                             raise e
#                         finally:
#                             return genomap, cell_indexes_few_batches, obs_multibatch, original_batch, plot_min, plot_max, n_batch_cols2plot + n_inputs_fe, path_2_genomap
                        
#                         try:
#                             plot_cell_recon_genomap(
#                                 genomap,
#                                 cell_indexes=cell_indexes_few_batches,
#                                 genomap_coordinates=genomap_coordinates,
#                                 obs=obs_multibatch,
#                                 original_batch=original_batch,
#                                 n_top_genes=10,
#                                 min_val=plot_min,
#                                 max_val=plot_max,
#                                 n_cols=n_batch_cols2plot + n_inputs_fe,
#                                 order="C",
#                                 path_2_genomap=path_2_genomap,
#                                 file_name=(
#                                     f"{cell_id}_few_batches_{statistic}_{original_celltype}_genelabels"
#                                 ),
#                                 remove_ticks=True,
#                             )
#                         except Exception as e: 
#                             error_here()
#                             raise e   

#                         # Obtain AML and control keys (and similarly for cell lines if needed)
#                         aml_keys = [key for key, value in dict_batches.items() if value == "AML"]
#                         aml_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in aml_keys]
#                         aml = obs_multibatch.loc[
#                             (obs_multibatch["recon_prefix"].str.contains('batch'))
#                             & (obs_multibatch["recon_prefix"].isin(aml_recon_batch_list))
#                         ]

#                         control_keys = [key for key, value in dict_batches.items() if value == "control"]
#                         control_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in control_keys]
#                         control = obs_multibatch.loc[
#                             (obs_multibatch["recon_prefix"].str.contains('batch'))
#                             & (obs_multibatch["recon_prefix"].isin(control_recon_batch_list))
#                         ]

#                         cl_keys = [key for key, value in dict_batches.items() if value == "celline"]
#                         cl_recon_batch_list = [f'recon_batch_{Type}_{b}' for b in cl_keys]
#                         cl = obs_multibatch.loc[
#                             (obs_multibatch["recon_prefix"].str.contains('batch'))
#                             & (obs_multibatch["recon_prefix"].isin(cl_recon_batch_list))
#                         ]

#                         # Matching indexes for AML and Control
#                         idx_aml = aml.index
#                         idx_control = control.index

#                         # ----------------------------------------------------------------------------
#                         # 11. Aggregate Genomaps per Batch
#                         # ----------------------------------------------------------------------------
#                         print("Aggregating Genomaps per Batch")
#                         aml_avg_maps = []
#                         for b in aml_recon_batch_list:
#                             row_inds = obs_multibatch.index[obs_multibatch['recon_prefix'] == b]
#                             batch_genomaps = genomap[row_inds, :, :, :]
#                             avg_map = batch_genomaps.mean(axis=0)
#                             aml_avg_maps.append(avg_map)

#                         ctrl_avg_maps = []
#                         for b in control_recon_batch_list:
#                             row_inds = obs_multibatch.index[obs_multibatch['recon_prefix'] == b]
#                             batch_genomaps = genomap[row_inds, :, :, :]
#                             avg_map = batch_genomaps.mean(axis=0)
#                             ctrl_avg_maps.append(avg_map)

#                         aml_avg_maps = np.stack(aml_avg_maps, axis=0)    # shape (n_asd, 54, 54, 1)
#                         ctrl_avg_maps = np.stack(ctrl_avg_maps, axis=0)  # shape (n_ctrl, 54, 54, 1)

#                         print("ASD average maps shape:", aml_avg_maps.shape)
#                         print("Control average maps shape:", ctrl_avg_maps.shape)

#                         # ----------------------------------------------------------------------------
#                         # 12. Perform Pixel-wise Mann-Whitney U Test
#                         # ----------------------------------------------------------------------------
#                         print("Perform Pixel-wise Mann-Whitney U Test")

#                         _, height, width, _ = aml_avg_maps.shape
#                         uvals = np.zeros((height, width))
#                         pvals = np.zeros((height, width))

#                         for i in range(height):
#                             for j in range(width):
#                                 u_stat, p_value = mannwhitneyu(
#                                     aml_avg_maps[:, i, j, 0],
#                                     ctrl_avg_maps[:, i, j, 0],
#                                     alternative="two-sided"
#                                 )
#                                 uvals[i, j] = u_stat
#                                 pvals[i, j] = p_value

#                         # Process gene coordinates for significance
#                         i_coords = genomap_coordinates["pixel_i"].values
#                         j_coords = genomap_coordinates["pixel_j"].values
#                         genomap_coordinates["pval"] = pvals[i_coords, j_coords]
#                         genomap_coordinates = genomap_coordinates.sort_values(by="pval")

#                         p_threshold = 0.05
#                         genomap_coordinates["significant"] = (
#                             genomap_coordinates["pval"] < p_threshold
#                         )
#                         print("\n# significant genes:",
#                             len(genomap_coordinates[genomap_coordinates["significant"]]))
#                         print(genomap_coordinates[genomap_coordinates["significant"]])


#                         # Save final dataframe with p-values and gene annotations
#                         genomap_coordinates.to_csv(
#                             os.path.join(out_name, "pvals_300cellsavg_mwutest.csv")
#                         )
#                 except Exception as e: 
#                     raise e
#                     print("\n".join(["#"*50,"#"*50, "it broke", "#"*50,"#"*50]))