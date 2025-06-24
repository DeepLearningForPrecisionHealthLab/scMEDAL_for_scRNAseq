import os
import glob
import pandas as pd
import numpy as np

from typing import Optional
#from collections import namedtuple
from types import SimpleNamespace
from .base import Model
from .scMEDAL.scMEDAL import AE as ae_alg
from utils.model_train_utils import run_all_folds

from utils.defaults import AML_EXPERIMENT_NAME, DATA_DIR, AML_OUTPUTS_DIR
#data_base_path = os.path.join(DATA_DIR, "AML", "AML_data")
data_base_path = f"/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/data/AML_data/"
input_base_path= f"/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/scMEDAL_for_scRNAseq/Experiments/data/AML_data/log_transformed_2916hvggenes/splits"
scenario_id = AML_EXPERIMENT_NAME
outputs_path = AML_OUTPUTS_DIR

class AE(Model):
    def __init__(self, model:str="ae", **kwargs):
        super().__init__(model=model, **kwargs)

    def run_train(self, data_path:Optional[str]=None,save_model:bool=True):
        ########## TODO: 
        if data_path is None:
            data_path = os.path.join(data_base_path, scenario_id)
        #input_base_path = os.path.join(data_base_path, scenario_id, 'splits')
        print(f"Parent folder: {input_base_path}")

        # --------------------------------------------------------------------------------------
        # 10. Define Output Paths
        # --------------------------------------------------------------------------------------
        folder_name = scenario_id
        model_name = "AE"

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
        # Define shape-color dictionary for plotting latent spaces
        shape_color_dict = {
            f"{params.bio_col}-{params.bio_col}": {"shape_col": params.bio_col, "color_col": params.bio_col},
            f"{params.batch_col}-{params.batch_col}": {"shape_col": params.batch_col, "color_col": params.batch_col},
            # Uncomment for additional combinations if needed
            # f"{params.bio_col}-{params.batch_col}": {"shape_col": params.bio_col, "color_col": params.batch_col},
            # f"{params.batch_col}-{params.batch_col}": {"shape_col": params.batch_col, "color_col": params.batch_col},
        }

        # --------------------------------------------------------------------------------------
        # 2. Load Metadata and Define Categories
        # --------------------------------------------------------------------------------------
        # Load metadata before splits
        metadata_all = pd.read_csv(glob.glob(data_path + "/*meta.csv")[0])

        # Convert columns to categorical types
        metadata_all['celltype'] = metadata_all['celltype'].astype('category')
        metadata_all['batch'] = metadata_all['batch'].astype('category')

        # Print the number of unique batches
        print("Number of batches:", len(np.unique(metadata_all[params.batch_col])))

        # Define One Hot Encoded (OHE) order for donor and celltype categories
        seen_donor_ids = np.unique(metadata_all[params.batch_col]).tolist()
        print("Ordered batches (donors):", seen_donor_ids)

        celltype_ids = np.unique(metadata_all[params.bio_col]).tolist()

        # --------------------------------------------------------------------------------------
        # 3. Run All Folds
        # --------------------------------------------------------------------------------------
        # Run folds and compute clustering metrics
        mean_scores = run_all_folds(
            Model=ae_alg,
            input_base_path=input_base_path,
            out_base_paths_dict=base_paths_dict,
            folds_list=params.fold_list,
            run_name=params.run_name,
            model_params_dict=self.model_params,
            build_model_dict={k:v for k, v in self.model_configs._asdict().items() if k != "ignore"},
            compile_dict=self.compile_configs._asdict(),
            save_model=save_model,
            batch_col=params.batch_col,
            bio_col=params.bio_col,
            batch_col_categories=seen_donor_ids,
            bio_col_categories=celltype_ids,
            model_type="ae",
            issparse=False,
            load_dense=False,
            shape_color_dict=shape_color_dict,
            sample_size=params.sample_size
        )

        # --------------------------------------------------------------------------------------
        # 4. Save Configuration File
        # --------------------------------------------------------------------------------------
        # Define the destination path for the configuration file
        destination_path = os.path.join(saved_models_base, run_name, 'model_config.py')

        print("\nCopying config.py file to:", destination_path)

        # Copy the configuration file to the destination
        # TODO: JSON DUMPS the configs
        #shutil.copy(source_file, destination_path)
