import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import glob
import configs.configs as cfg
import tensorflow as tf
import pandas as pd
import numpy as np
from abc import ABC
from typing import Optional, Dict, Any
from types import SimpleNamespace
from utils.model_train_utils import generate_run_name, run_all_folds
from dataclasses import asdict, is_dataclass, replace




# ---------------------------------------------------------------------------
# Generic helpers to make all config objects (named?tuple, dataclass, dict, or
# plain class) behave uniformly.
# ---------------------------------------------------------------------------

def _update_cfg(obj, updates: Optional[Dict[str, Any]]):
    """Return a *new* config with *updates* applied, regardless of obj type."""
    if not updates:
        return obj

    # 1) typing.NamedTuple ? has _fields & _replace
    if hasattr(obj, "_fields") and hasattr(obj, "_replace"):
        return obj._replace(**{k: v for k, v in updates.items() if k in obj._fields})

    # 2) dataclass ? use dataclasses.replace
    if is_dataclass(obj):
        return replace(obj, **{k: v for k, v in updates.items() if hasattr(obj, k)})

    # 3) dict ? mutate in place (config objects created as .configs dicts)
    if isinstance(obj, dict):
        obj.update({k: v for k, v in updates.items() if k in obj})
        return obj

    # 4) plain Python class ? set attributes
    for k, v in updates.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj


def _cfg_to_dict(obj) -> Dict[str, Any]:
    """Flatten any config wrapper into a plain dict.

    * If an attribute name **starts with "configs"** (e.g. ``configs?BaseScoreConfigs``)
      its *contents* are recursively flattened so that end?users can treat the
      inner parameters as top?level keys (useful for run?name exclusion).
    """
    # 1) Already a dict ? fast?path
    if isinstance(obj, dict):
        return obj

    # 2) Unwrap common pattern where the real payload is in `.configs`
    if hasattr(obj, "configs"):
        inner = getattr(obj, "configs")
        try:
            # trust recursion to handle the shape
            return _cfg_to_dict(inner)
        except RecursionError:
            pass  # fallthrough ? unlikely but safe

    # 3) NamedTuple
    if hasattr(obj, "_asdict"):
        return obj._asdict()

    # 4) Dataclass
    if is_dataclass(obj):
        return asdict(obj)

    # 5) Plain class ? use vars(), with special handling for `configs?*` attrs
    flat: Dict[str, Any] = {}
    for k, v in vars(obj).items():
        if k.startswith("configs"):
            # Merge the *inner* dict instead of keeping the wrapper
            flat.update(_cfg_to_dict(v))
        elif not k.startswith("_") and not callable(v):
            flat[k] = v
    return flat



class Model(ABC):
    """Base class for all scMEDAL family models."""

    # ---------------------------------------------------------------------
    # 0. Allowed identifiers
    # ---------------------------------------------------------------------
    valid_models = ["ae", "aec", "scmedalfe", "scmedalfec", "scmedalre"]
    valid_named_experiment = ["AML", "ASD", "HH"]

    # ---------------------------------------------------------------------
    # 1. Constructor & config initialisation
    # ---------------------------------------------------------------------
    def __init__(self, model_name: str, **kwargs):
        assert model_name in self.valid_models, (
            f"Invalid model {model_name}. Valid options: {self.valid_models}"
        )
        self.model_name = model_name

        # --- Config groups ------------------------------------------------
        self.compile_configs = self._init_compile_configs(kwargs)
        self.model_configs = self._init_model_configs(kwargs)
        self.data_configs = self._init_data_configs(kwargs)
        self.training_configs = self._init_training_configs(kwargs)
        self.scores_configs = self._init_score_configs(kwargs)
        self.expt_design_configs = self._init_exp_design_configs(kwargs)

        # combined params dict (used for run names etc.)
        self.model_params: Dict[str, Any] = self._init_model_params()

    # ------------------------------------------------------------------
    # 1.1  Config helpers (all funnel to _update_cfg)
    # ------------------------------------------------------------------
    def _init_compile_configs(self, kwargs=None):
        cfg_obj = cfg.CompileConfigs(self.model_name).configs  # likely dict
        return _update_cfg(cfg_obj, kwargs)

    def _init_model_configs(self, kwargs=None):
        cfg_obj = cfg.ModelConfigs(self.model_name).configs
        return _update_cfg(cfg_obj, kwargs)

    def _init_data_configs(self, kwargs=None):
        cfg_obj = cfg.DataConfigs(self.model_name).configs
        return _update_cfg(cfg_obj, kwargs)

    def _init_training_configs(self, kwargs=None):
        cfg_obj = cfg.TrainingConfigs(self.model_name)
        return _update_cfg(cfg_obj, kwargs)

    def _init_score_configs(self, kwargs=None):
        cfg_obj = cfg.ScoreConfigs(self.model_name)
        return _update_cfg(cfg_obj, kwargs)

    def _init_exp_design_configs(self, kwargs=None):
        cfg_obj = cfg.ExperimentDesignConfigs()
        return _update_cfg(cfg_obj, kwargs)

    # ------------------------------------------------------------------
    # 1.2  Consolidate model parameters into a single dict
    # ------------------------------------------------------------------
    def _init_model_params(self):
        params = {
            **_cfg_to_dict(self.compile_configs),
            **_cfg_to_dict(self.model_configs),
            **_cfg_to_dict(self.data_configs),
            **_cfg_to_dict(self.training_configs),
            **_cfg_to_dict(self.scores_configs),
            **_cfg_to_dict(self.expt_design_configs),
        }
        ignore = params.pop("ignore", [])
        params["run_name"] = generate_run_name(params, ignore, name="run_crossval")
        return params

    # ------------------------------------------------------------------
    # 2. TF alias replacements (optimizer, loss, metrics)
    # ------------------------------------------------------------------
    def _materialise_compile_objects(self):
        """Convert string names in compile_configs to actual TensorFlow objects."""
        compile_cfg = self.compile_configs
        get = (lambda k: compile_cfg[k]) if isinstance(compile_cfg, dict) else (
            lambda k: getattr(compile_cfg, k)
        )

        new_cfg = {
            "optimizer": tf.keras.optimizers.get(get("optimizer")),
            "loss": [tf.keras.losses.get(l) for l in get("loss")],
            "metrics": [tf.keras.metrics.get(m) for m in get("metrics")],
        }
        self.compile_configs = _update_cfg(compile_cfg, new_cfg)

    # ------------------------------------------------------------------
    # 3. Convenience helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except Exception:
            return False

    def save_configs(self, path: str) -> None:
        """Dump model_params to JSON, handling non?serialisable objects."""
        safe = {
            k: v if self._is_jsonable(v) else ["Non?JSON", str(v)]
            for k, v in self.model_params.items()
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(safe, fh, indent=2)

    # ------------------------------------------------------------------
    # 4. Predefined experiment paths
    # ------------------------------------------------------------------
    def load_named_experiment_paths(self, experiment: str) -> Dict[str, str]:
        assert experiment in self.valid_named_experiment, (
            f"Unrecognised experiment '{experiment}'. Valid: {self.valid_named_experiment}"
        )
        base = (
            "/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments"
        )
        mapping = {
            "AML": {
                "data_path": f"{base}/data/AML_data",
                "scenario_id": "log_transformed_2916hvggenes",
                "splits_folder": "splits",
            },
            "ASD": {
                "data_path": f"{base}/data/ASD_data/reverse_norm/",
                "scenario_id": "log_transformed_2916hvggenes",
                "splits_folder": "splits",
            },
            "HH": {
                "data_path": f"{base}/data/HealthyHeart_data/",
                "scenario_id": "log_transformed_3000hvggenes",
                "splits_folder": "splits",
            },
        }
        return mapping[experiment]

    # ------------------------------------------------------------------
    # 5. Training driver (trimmed; unchanged logic)
    # ------------------------------------------------------------------
    def run_train(
        self,
        data_path: Optional[str] = None,
        outputs_path: Optional[str] = None,
        named_experiment: Optional[str] = None,
        save_model: bool = True,
        quick: bool = False,
        quick_epochs: int = 3,
        quick_folds=None,
        plotconfigs: Optional[cfg.PlotConfigs] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """quick:"quick setup"""
        quick_folds = quick_folds or [1]
        # (body unchanged ? same as previous version)


        if plotconfigs is None:
            plotconfigs = cfg.PlotConfigs() if plot_kwargs is None else cfg.PlotConfigs(**plot_kwargs)
        shape_color_dict = plotconfigs.get_shape_color_dict(self.expt_design_configs)

        if quick:
            self.training_configs._replace(epochs=3)
            self.model_params['epochs'] =  quick_epochs
            self.model_params['fold_list'] = quick_folds

        model_name = self.model_name

        if outputs_path is None:
            outputs_path = os.path.join(os.getcwd(), "outputs")

        if named_experiment is not None:
            paths = self.load_named_experiment_paths(named_experiment)
            data_path = paths.get("data_path")
            folder_name = paths.get("scenario_id")
            splits_path = os.path.join(data_path, folder_name, paths.get("splits_folder"))
            outputs_path = os.path.join(outputs_path, named_experiment)
        
        issparse, load_dense = False, False
        if named_experiment == "HH":
            issparse, load_dense = True, True

        print(f"Parent folder: {splits_path}")

        saved_models_base = os.path.join(outputs_path, "saved_models", folder_name, model_name)
        figures_base = os.path.join(outputs_path, "figures", folder_name, model_name)
        latent_space_base = os.path.join(outputs_path, "latent_space", folder_name, model_name)

        # --------------------------------------------------------------------------------------
        # 12. Define Base Paths Dictionary
        # --------------------------------------------------------------------------------------
        base_paths_dict = {
            "models": saved_models_base,
            "figures": figures_base,
            "latent": latent_space_base
        }

        print("Save model set to:", save_model)

        params=SimpleNamespace(**self.model_params)

        # --------------------------------------------------------------------------------------
        # 2. Load Metadata and Define Categories
        # --------------------------------------------------------------------------------------
        # Load metadata before splits
        metadata_all = pd.read_csv(glob.glob(os.path.join(data_path, folder_name) + "/*meta.csv")[0])

        # Convert columns to categorical types
        metadata_all['celltype'] = metadata_all['celltype'].astype('category')
        if "batch" in metadata_all.columns:
            metadata_all['batch'] = metadata_all['batch'].astype('category')
        if "sampleID" in metadata_all.columns:
            metadata_all['batch'] = metadata_all['sampleID'].astype('category')

        # Print the number of unique batches
        print("Number of batches:", len(np.unique(metadata_all[params.batch_col])))

        # Define One Hot Encoded (OHE) order for donor and celltype categories
        seen_donor_ids = np.unique(metadata_all[params.batch_col]).tolist()
        print("Ordered batches (donors):", seen_donor_ids)

        celltype_ids = np.unique(metadata_all[params.bio_col]).tolist()


        try:
        # Works if self.model_configs is already a dict-like object
            build_model_dict = {k: v for k, v in self.model_configs.items() if k != "ignore"}
        except AttributeError:
        # Fallback for NamedTuple (or any object that lacks .items())
            build_model_dict = {k: v for k, v in self.model_configs._asdict().items() if k != "ignore"}

        # --------------------------------------------------------------------------------------
        # 3. Run All Folds
        # --------------------------------------------------------------------------------------
        # Run folds and compute clustering metrics
        mean_scores = run_all_folds(
            Model=self.alg,
            input_base_path=splits_path,
            out_base_paths_dict=base_paths_dict,
            folds_list=params.fold_list,
            run_name=params.run_name,
            model_params_dict=self.model_params,
            build_model_dict=build_model_dict,
            compile_dict=self.compile_configs,
            save_model=save_model,
            batch_col=params.batch_col,
            bio_col=params.bio_col,
            batch_col_categories=seen_donor_ids,
            bio_col_categories=celltype_ids,
            model_type=self.model_name,
            issparse=issparse,
            load_dense=load_dense,
            shape_color_dict=shape_color_dict,
            sample_size=params.sample_size
        )

        # --------------------------------------------------------------------------------------
        # 4. Save Configuration File
        # --------------------------------------------------------------------------------------
        destination_path = os.path.join(saved_models_base, params.run_name, "configs.json")
        self.save_configs(destination_path)
