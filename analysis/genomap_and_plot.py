
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

from anndata import AnnData

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
    extra_label_cols: Optional[List[str]] = None


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
        self.df = None   # if you have cm multibatch not needed. Before: self._load_and_merge_paths()
        # self.df = self._load_and_merge_paths()
        print("\n\nInitialized genomap pipeline ..")
        #print("sparse",cfg.issparse)

    #  internal helpers 

    def _load_and_merge_paths(self) -> pd.DataFrame:
        print(f"\nLooking for outputs paths for the following models:{self.results_path_dict.keys()}")
        df_recon = get_recon_paths_df(self.results_path_dict, get_batch_recon_paths=True)
        df_inputs = get_input_paths_df(self.cfg.input_base_path)
        df = pd.merge(df_recon, df_inputs, on=["Split", "Type"], how="left")
        df["recon_prefix"] = df["ReconPath"].apply(lambda p: os.path.basename(p).split(".npy")[0])
        print(f"Created df with input and recon paths")
        #print("f\{df}")
        return df

    def _select_recon(self, model: str, split: int, typ: str):
        mask = (self.df["Key"] == model) & (self.df["Split"] == split) & (self.df["Type"] == typ)
        return self.df.loc[mask, ["ReconPath", "recon_prefix"]].values.T  # 2×N arrays

    def _input_path(self, model: str, split: int, typ: str) -> str:
        #print(f"Searching input paths in df\n{self.df}")
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
        print("\nComputing genomap..")
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
            print("cells selected from the following batches",batches_to_select_from)
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
        statistic  = "std"
    ):
        cfg = self.cfg
        compute_cell_stats_acrossbatchrecon(
            genomap, idxs, coords,
            statistic      = statistic,
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
        statistic = 'std'
    ):
        cfg = self.cfg
        filename = f"{cell_id}_{original_ct}_{statistic}"
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
            extra_label_cols= cfg.extra_label_cols
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
        statistic = "std"
    ):
        
        cfg = self.cfg
        if statistic is not None:
            file_stub = f"screconfewbatches_{cell_id}_{original_ct}_{statistic}"
        else:
            file_stub = f"screconfewbatches_{cell_id}_{original_ct}"
        # if with_labels:
        #     file_stub += "_genelabels"

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
            extra_label_cols    =cfg.extra_label_cols
        )
        matplotlib.pyplot.close()


    def _compute_scstats_and_statplots(
        self,
        genomap,
        idxs_all,
        coords,
        out_dir,
        cid,
        statistic,
        obs_df,
        original_batch,
        original_ct,
        idxs_subset,
        n_cols_subset
    ):
        """Compute statistics for a single cell across batches. Outputs: CSVs and plot the big panel and the subset panel *with* labels."""
        # -------- statistics CSV (you need them to select relevant genes to label in the plot) -------------------------------------
        stats_dir = os.path.join(out_dir, f"singlecell_gene_{statistic}_acrossbatches")
        os.makedirs(stats_dir, exist_ok=True)
        coords_work = coords.reset_index(drop=True)
        self._write_cell_stats(genomap, idxs_all, coords_work, stats_dir, cid, statistic)

        print(f"\n\nplotting big panel for single cell {cid} from original batch {original_batch} and celltype {original_ct}, n recons: {len(idxs_all)}")
        # -------- big panel  for cell recons with gene labels------------------------------------------
        plot_dir_labels   = os.path.join(stats_dir, "genomap_plots_scallbatches_genelabels")
        os.makedirs(plot_dir_labels , exist_ok=True)
        self._plot_big_panel(
            genomap, idxs_all, coords_work,
            obs_df, plot_dir_labels, cid, original_batch, original_ct
        )


        plot_dir_labels   = os.path.join(stats_dir, "genomap_plots_scfewbatches_genelabels")
        os.makedirs(plot_dir_labels,   exist_ok=True)
        self._plot_subset_panel(
            genomap, idxs_subset, coords_work,
            obs_df, plot_dir_labels , cid, original_batch, original_ct,
            n_cols_subset, with_labels=True, statistic = statistic
        )

        return stats_dir, coords_work

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
        statistic = "std"
    ):
        """Create (i) a big panel with *all* recons and
           (ii) two small subset panels per cell."""
        # ----------------------------------------------------
        # recon2plot  =   [extra_prefixes]  +  original_batch_list
        # (mirrors the old `recon2plot = recon2plot + original_batch_list`)
        # ----------------------------------------------------
        cfg = self.cfg

        extra_pref   = self._extra_prefixes(typ)
        recon2plot   = extra_pref + original_batch_list
        n_inputs_fe  = len(extra_pref)

        out_dir   = os.path.join(out_dir, "genomap_plots")
        os.makedirs(out_dir,   exist_ok=True)
        celltypes = np.unique(obs_df["celltype"].values)
        n_celltypes = len(celltypes)

        for cid in cell_ids:


            # get the all the indexes of the cm multibatch that have the same cid (belong to the same cell and are recon from diff batches)
            idxs_all = obs_df.loc[obs_df[cfg.cell_id_col] == cid].index.astype(int)
            # print("n cell indexes", idxs_all)

            if len(idxs_all) == 0:
                continue

            original_batch = obs_df.loc[idxs_all[0], "batch"]
            original_ct    = obs_df.loc[idxs_all[0], "celltype"]
            
            if n_celltypes>1: #create subfolders for celltypes
                out_dir_ct = os.path.join(out_dir,original_ct.replace("/", ""))
                os.makedirs(out_dir_ct, exist_ok=True)
            else:
                out_dir_ct = out_dir



            # -------- subset of reconstructions (only plot cell recons for few batches) --------------------------
            subset_labels = set(recon2plot)
            idxs_subset = obs_df.loc[
                (obs_df[cfg.cell_id_col] == cid)
                & obs_df["recon_prefix"].apply(
                    lambda x: any(lbl in x for lbl in subset_labels)
                )
            ].index.astype(int)
            # print("n cell indexes for batch CF recon", idxs_subset)
            if len(idxs_subset) == 0:
                continue

            #n_batch_cols  = len(obs_df.loc[idxs_subset, "batch"].unique())
            n_cols_subset = max(1, n_batch_cols2plot + n_inputs_fe)



            print(f"plotting small panel for {cid} with original batch {original_batch} and celltype {original_ct}, n recons: {len(idxs_subset)}")
            



            # -------- statistics of a single cell across batches CSV + big panel + subset (with labels) -------------------------------------
            stats_dir, coords_work = self._compute_scstats_and_statplots(
            genomap=genomap,
            idxs_all=idxs_all,
            coords=coords,
            out_dir=out_dir_ct,
            cid=cid,
            statistic=statistic,
            obs_df=obs_df,
            original_batch=original_batch,
            original_ct=original_ct,
            idxs_subset=idxs_subset,
            n_cols_subset=n_cols_subset)

            # (b) without gene labels
            print(f"created plot with no gene labels")
            plot_dir_nolabels = os.path.join(out_dir_ct, "genomap_plots_scfewbatches_nogenelabels")
            os.makedirs(plot_dir_nolabels, exist_ok=True)
            self._plot_subset_panel(
                genomap, idxs_subset, None,
                obs_df, plot_dir_nolabels, cid, original_batch, original_ct,
                n_cols_subset, with_labels=False, statistic =None # you only need the stats to choose which  genes to label
            )

        # helper: build consistent names 
    def _make_cm_and_gnames(self,typ: str, split: int, out_dir: str) -> Tuple[str, str, str]:
        """Return (cm_name, cm_path, gname) exactly like the writer produces."""
        cfg     = self.cfg
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


    def run(
        self,
        model: Optional[str] = None, # models: Optional[List[str]] = None,
        typ: Optional[str] = None, # types:  Optional[List[str]] = None,
        split: Optional[int] = None, # splits: Optional[List[int]] = None,
    ) -> dict:
        """Execute full pipeline that creates adata multibatch, genomaps and plots and return a summary for a model/type/split."""

        cfg     = self.cfg
        typ   = typ  or "train" #test and val also possible
        split  = split or 1
        summary: dict = {}
        inputs_path = None # initialize, only filled when adata_mb is computed


        # 1. output directory -------------------------------------------------
        out_dir = os.path.join(cfg.compare_models_path,
                                        cfg.analysis_name or "analysis","genomap")
        os.makedirs(out_dir, exist_ok=True)
        print(f"genomaps saved to {out_dir} ")

        # calling genomap function
        print("Run genomap pipeline for: ",typ,split,model)
        
        # 5. multibatch matrix ----------------------------------------------
        cm_name, cm_path, gname = self._make_cm_and_gnames(typ, split, out_dir)
        if not os.path.exists(cm_path):
            if self.df is None: # loading recon paths (this is not necessary if you laready have cm_multibatch)
                print("\nLoading recon paths..")
                self.df = self._load_and_merge_paths() # loading recon paths (this is not necessary if you laready have cm_multibatch)
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
            print("Batches selected for plotting:",batches_sel)
            print ("\n\nSampling adata  multibatch from original data and recons..") 
            print("Saving to adata multibatch directory:",cm_path)     
            adata_mb = self._build_multibatch(
                paths, prefixes, obs, var, batches_sel, out_dir
            )
            print("adata_mb",adata_mb)
            adata_mb.var["gene_names"] = adata_mb.var.index 
        else:
            print ("\n\nReading adata multibatch from",cm_path)  
            
            X_mb, var_mb, obs_mb = read_adata(cm_path)
            adata_mb= AnnData(X=X_mb, obs=obs_mb, var=var_mb)                       
            adata_mb.var["gene_names"] = adata_mb.var[cfg.gene_index_col].astype(str)#.values
            adata_mb.var.index = adata_mb.var["gene_names"]
            adata_mb.obs.index = adata_mb.obs.index.astype(int)
            print("adata_mb",adata_mb)
            # print("adata_mb gene_names",adata_mb.var["gene_names"])
            # print("adata_mb gene_names",adata_mb.var["gene_names"])
            #print("adata_mb index",adata_mb.obs.index)
            
            # batch selection should come from obs in CM
            batches_sel = cfg.batches or list(adata_mb.obs["batch"].unique())
            print("Batches selected for plotting:", batches_sel)
            print (f"Please verify that the count matrix adata multibatch contains cells from the  selected batches: {batches_sel}")

        # choose cells and original batches
        cell_ids, original_batch_list, n_batch_cols2plot = self._choose_cells_and_batches(adata_mb, typ, batches_to_select_from =batches_sel)
        print("Original batch list:",original_batch_list)

        # adjust ncells 
        extra = cfg.n_inputs_fe if cfg.add_inputs_fe else 0
        ncells_for_genomap = cfg.n_cells_per_batch * (cfg.n_batches + extra)

        #genomap = self._compute_genomap(cm_path, gname, out_dir, gene_names=gene_names=var.index[:cfg.n_genes])
        gene_names = adata_mb.var.index.astype(str)[:cfg.n_genes]
        genomap = self._compute_genomap(cm_path, gname, out_dir, gene_names=gene_names)
        coords = self._load_gene_coordinates(out_dir, gname)
        # 8. plot panels ------------------------------------------------------
        self._plot_panels(genomap, adata_mb.obs, coords, out_dir, typ, cell_ids, original_batch_list,n_batch_cols2plot)
        # 9. summarise --------------------------------------------------------
        summary = {
                "model":   model,
                "typ":    typ,
                "split":   split,
                "genomap": genomap,
                "adata_mb": adata_mb,
                "inputs_path": inputs_path,
                "batches_to_select_from": batches_sel,
                "cell_ids_to_plot": cell_ids,
                "original_batch_list":original_batch_list,
                "out_dir": out_dir,
                "gname":gname,
            }
        del adata_mb, genomap
        gc.collect()

        return summary

    def compute_asd_me_stats(self,
                        genomap: np.ndarray,
                        obs_multibatch: pd.DataFrame,
                        genomap_coordinates: pd.DataFrame,
                        asd_recon_batch_list: List[str],
                        control_recon_batch_list: List[str],
                        p_threshold: float = 0.05) -> pd.DataFrame:
        """
        Compute mixed-effects model statistics for all pixels comparing ASD and Control groups.

        Args:
            genomap: Genomap data array.
            obs_multibatch: Observations DataFrame.
            genomap_coordinates: DataFrame with pixel coordinates.
            asd_recon_batch_list: List of batch prefixes for ASD group.
            control_recon_batch_list: List of batch prefixes for Control group.
            p_threshold: Significance threshold for identifying significant pixels.

        Returns:
            Updated genomap_coordinates DataFrame with mixed-effects model results.
        """
        import statsmodels.formula.api as smf
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        # Silence all ConvergenceWarning messages globally
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            category=UserWarning)


        all_pixel_results = []

        # Iterate over all pixels
        for i in range(genomap.shape[1]):
            for j in range(genomap.shape[2]):
                data_rows = []

                # ASD group
                for b in asd_recon_batch_list:
                    row_inds = obs_multibatch.index[obs_multibatch['recon_prefix'] == b]
                    pixel_vals = genomap[row_inds, i, j, 0]
                    for val in pixel_vals:
                        data_rows.append({"batch": b, "group": "ASD", "value": val})

                # Control group
                for b in control_recon_batch_list:
                    row_inds = obs_multibatch.index[obs_multibatch['recon_prefix'] == b]
                    pixel_vals = genomap[row_inds, i, j, 0]
                    for val in pixel_vals:
                        data_rows.append({"batch": b, "group": "Control", "value": val})

                # Create DataFrame for mixed-effects model
                df_ij = pd.DataFrame(data_rows)

                # Fit model: value ~ group + (1|batch)
                model_ij = smf.mixedlm("value ~ group", data=df_ij, groups=df_ij["batch"])
                fit_res_ij = model_ij.fit()

                # Extract estimates and statistics
                intercept = fit_res_ij.params["Intercept"]
                slope_control = fit_res_ij.params["group[T.Control]"]
                pval_intercept = fit_res_ij.pvalues["Intercept"]
                pval_control = fit_res_ij.pvalues["group[T.Control]"]
                ci = fit_res_ij.conf_int(alpha=0.05)
                intercept_lower_95CI, intercept_upper_95CI = ci.loc["Intercept"]
                slope_control_lower_95CI, slope_control_upper_95CI = ci.loc["group[T.Control]"]

                all_pixel_results.append({
                    "pixel_i": i,
                    "pixel_j": j,
                    "intercept": intercept,
                    "slope_control": slope_control,
                    "pval_intercept": pval_intercept,
                    "pval_control": pval_control,
                    "intercept_lower_95CI": intercept_lower_95CI,
                    "intercept_upper_95CI": intercept_upper_95CI,
                    "slope_control_lower_95CI": slope_control_lower_95CI,
                    "slope_control_upper_95CI": slope_control_upper_95CI
                })

        # Compile results
        df_results_lmm = pd.DataFrame(all_pixel_results).sort_values("pval_control")

        # Merge with genomap_coordinates
        genomap_coordinates = genomap_coordinates.merge(
            df_results_lmm,
            on=["pixel_i", "pixel_j"],
            how="left"
        )

        genomap_coordinates.rename(columns={'pval_control': 'pval'}, inplace=True)
        genomap_coordinates["significant"] = genomap_coordinates["pval"] < p_threshold

        print(f"Total significant pixels identified: {genomap_coordinates['significant'].sum()}")

        return genomap_coordinates

    def computeandsave_asd_me_stats(self, summary, p_threshold: float = 0.05):
        """Internal helper that computes ME stats for the first summary and appends
        the results (path + DataFrame) to that summary dict.

        Parameters
        ----------
        summary : dict
            returned by ``pipeline.run``.
        p_threshold : float, optional
            Significance cut off; forwarded to ``compute_me_stats``.

        """
        import os, pandas as pd
 

        if not summary:
            raise RuntimeError("Empty summary passed to _save_asd_me_stats")

        genomap = summary["genomap"]
        obs_mb = summary["adata_mb"].obs
        typ = summary["typ"]
        split = summary["split"]

        # Diagnosis  batch mapping
        dict_batches = (
            obs_mb[["diagnosis", "batch"]]
            .drop_duplicates()
            .set_index("batch")["diagnosis"].to_dict()
        )
        asd_keys     = [k for k, v in dict_batches.items() if v == "ASD"]
        control_keys = [k for k, v in dict_batches.items() if v == "Control"]

        asd_recon_batch_list     = [f"recon_batch_{typ}_{b}" for b in asd_keys]
        control_recon_batch_list = [f"recon_batch_{typ}_{b}" for b in control_keys]

        # Gene coordinate table
        #_, _, gname = self._make_cm_and_gnames(typ, split, summary["out_dir"])
        # coords_path = os.path.join(summary["out_dir"], f"gene_coordinates_{gname}.csv")
        # coords      = pd.read_csv(coords_path).rename(columns={"Unnamed: 0": "gene_names"})
        coords = self._load_gene_coordinates(summary["out_dir"],summary["gname"])

        updated_coords = self.compute_asd_me_stats(
            genomap                  = genomap,
            obs_multibatch           = obs_mb,
            genomap_coordinates      = coords,
            asd_recon_batch_list     = asd_recon_batch_list,
            control_recon_batch_list = control_recon_batch_list,
            p_threshold              = p_threshold,
        )

        me_path = os.path.join(summary["out_dir"], "asd_pvals_me.csv")
        updated_coords.to_csv(me_path, index=False)
        print("Saved mixed effects statistics ", me_path)

        summary["me_stats_path"] = me_path
        summary["me_dataframe"]  = updated_coords

        return summary


    # ------------------------------------------------------------------
    #  Mann-Whitney-U statistics: AML vs Control
    # ------------------------------------------------------------------

    def compute_aml_mwu_stats(
        self,
        genomap: np.ndarray,
        obs_multibatch: pd.DataFrame,
        genomap_coordinates: pd.DataFrame,
        aml_recon_batch_list: List[str],
        control_recon_batch_list: List[str],
        p_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """
        Compute pixel-wise Mann-Whitney U statistics comparing AML and Control
        groups.  Returns an updated ``genomap_coordinates`` dataframe that
        contains U-values, p-values and a boolean ``significant`` column.

        Parameters
        ----------
        genomap : np.ndarray
            4-D array returned by ``_compute_genomap``  
            (shape = [n_recons, 54, 54, 1]).
        obs_multibatch : pd.DataFrame
            ``adata_multibatch.obs`` holding metadata for every reconstruction.
        genomap_coordinates : pd.DataFrame
            Output of ``_load_gene_coordinates`` (must contain ``pixel_i`` and
            ``pixel_j``).
        aml_recon_batch_list, control_recon_batch_list : List[str]
            Lists with the *recon_prefix* values that correspond to each group,
            e.g. ``['recon_batch_train_B01', ...]``.
        p_threshold : float, default 0.05
            Significance threshold used to label pixels.

        Returns
        -------
        pd.DataFrame
            ``genomap_coordinates`` augmented with the U- and p-values plus a
            ``significant`` boolean column.
        """
        from scipy.stats import mannwhitneyu    # already imported at top of file
        # ------------------------------------------------------------------
        # 1 | aggregate one mean-map per *batch*
        # ------------------------------------------------------------------
        def _avg_maps(prefixes: List[str]) -> np.ndarray:
            maps = []
            for p in prefixes:
                idx = obs_multibatch.index[obs_multibatch["recon_prefix"] == p]
                if len(idx) == 0:
                    continue           # skip if batch not present
                maps.append(genomap[idx, :, :, :].mean(axis=0))
            if not maps:
                raise RuntimeError(f"No genomaps found for prefixes: {prefixes}")
            return np.stack(maps, axis=0)      # (n_batches, 54, 54, 1)

        aml_avg_maps   = _avg_maps(aml_recon_batch_list)
        ctrl_avg_maps  = _avg_maps(control_recon_batch_list)

        # ------------------------------------------------------------------
        # 2 | pixel-wise Mann-Whitney U
        # ------------------------------------------------------------------
        _, H, W, _ = aml_avg_maps.shape
        uvals  = np.zeros((H, W))
        pvals  = np.zeros((H, W))

        for i in range(H):
            for j in range(W):
                u, p = mannwhitneyu(
                    aml_avg_maps[:, i, j, 0],
                    ctrl_avg_maps[:, i, j, 0],
                    alternative="two-sided",
                )
                uvals[i, j] = u
                pvals[i, j] = p

        # ------------------------------------------------------------------
        # 3 | attach stats to genomap_coordinates
        # ------------------------------------------------------------------
        i_coords = genomap_coordinates["pixel_i"].values
        j_coords = genomap_coordinates["pixel_j"].values
        genomap_coordinates["uval"] = uvals[i_coords, j_coords]
        genomap_coordinates["pval"] = pvals[i_coords, j_coords]
        genomap_coordinates.sort_values("pval", inplace=True)
        genomap_coordinates["significant"] = genomap_coordinates["pval"] < p_threshold

        print(f"Total significant pixels identified (MWU): "
              f"{genomap_coordinates['significant'].sum()}")

        return genomap_coordinates



    def computeandsave_aml_mwu_stats(self, summary: dict, p_threshold: float = 0.05):
        """
        Convenience wrapper that  extracts inputs from *summary*,
        runs ``compute_aml_mwu_stats``, ? writes the CSV, and
        injects the results back into *summary*.

        Adds two keys to *summary*:
          * ``'mwu_stats_path'`` CSV file with all pixels
          * ``'mwu_dataframe'``  the dataframe itself
        """
        if not summary:
            raise RuntimeError("Empty summary passed to computeandsave_aml_mwu_stats")

        genomap   = summary["genomap"]
        obs_mb    = summary["adata_mb"].obs
        typ       = summary["typ"]
        split     = summary["split"]

        # --------------------------------------------------------------
        # build batch->group map from Patient_group
        # --------------------------------------------------------------
        dict_batches = (
            obs_mb[["Patient_group", "batch"]]
            .drop_duplicates()
            .set_index("batch")["Patient_group"]
            .str.lower()            # case-insensitive match
            .to_dict()
        )
        aml_keys     = [k for k, v in dict_batches.items() if v == "aml"]
        control_keys = [k for k, v in dict_batches.items() if v == "control"]

        aml_recon_batch_list     = [f"recon_batch_{typ}_{b}" for b in aml_keys]
        control_recon_batch_list = [f"recon_batch_{typ}_{b}" for b in control_keys]

        # --------------------------------------------------------------
        # load gene-coordinate table
        # --------------------------------------------------------------
        # _, _, gname = self._make_cm_and_gnames(typ, split, summary["out_dir"])
        # coords_path = os.path.join(summary["out_dir"], f"gene_coordinates_{gname}.csv")
        # coords      = pd.read_csv(coords_path).rename(columns={"Unnamed: 0": "gene_names"})
        coords = self._load_gene_coordinates(summary["out_dir"],summary["gname"])

        # --------------------------------------------------------------
        # run stats + save
        # --------------------------------------------------------------
        updated_coords = self.compute_aml_mwu_stats(
            genomap                  = genomap,
            obs_multibatch           = obs_mb,
            genomap_coordinates      = coords,
            aml_recon_batch_list     = aml_recon_batch_list,
            control_recon_batch_list = control_recon_batch_list,
            p_threshold              = p_threshold,
        )

        mwu_path = os.path.join(summary["out_dir"], "aml_pvals_mwu.csv")
        updated_coords.to_csv(mwu_path, index=False)
        print("Saved Mann-Whitney-U statistics:", mwu_path)

        # attach to caller?s summary dict
        summary["mwu_stats_path"] = mwu_path
        summary["mwu_dataframe"]  = updated_coords
        return summary
    
def genomap_and_plot(run_names_dict, results_path_dict, cfg: GenomapConfig, **kwargs):
    pipeline = GenomapPipeline(run_names_dict, results_path_dict, cfg)
    # return pipeline.run(**kwargs)
    summary  = pipeline.run(**kwargs)
    return summary
    # pipeline.computeandsave_asd_me_stats(summary)


    