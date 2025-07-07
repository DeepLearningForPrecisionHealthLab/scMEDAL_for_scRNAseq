import os
import glob
import pandas as pd
import numpy as np
import configs.configs as cfg
from types import SimpleNamespace
from .base import Model
from .scMEDAL.scMEDAL import MixedEffectsModel as me_alg
from utils.model_train_utils import run_model_pipeline_LatentClassifier_v2_PCA, ModelManager, calculate_metrics_with_ci
from utils.compare_results_utils import get_latent_paths_df, get_input_paths_df, create_latent_dict_from_df

class MEC(Model):
    def __init__(self, **kwargs):
        super().__init__(model_name="mec", **kwargs)
        self.alg = me_alg

    def run_train(self, results_path_dict, data_path=None, outputs_path=None, named_experiment=None, save_model=True, quick=False, plotconfigs=None, plot_kwargs=None):
        """
        Quick sets epochs to 3.
        """
        
        if plotconfigs is None:
            plotconfigs = cfg.PlotConfigs() if plot_kwargs is None else cfg.PlotConfigs(**plot_kwargs)
        shape_color_dict = plotconfigs.get_shape_color_dict(self.expt_design_configs)

        if quick:
            self.training_configs._replace(epochs=10)
            self.model_params['epochs'] = 10
            self.model_params['fold_list'] = [1]

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

        mod = {"Model": self.alg}
        params=SimpleNamespace(**self.model_params, **mod)

        # --------------------------------------------------------------------------------------
        # 2. Load Metadata and Define Categories
        # --------------------------------------------------------------------------------------
        # Load metadata before splits
        metadata_all = pd.read_csv(glob.glob(os.path.join(data_path, folder_name) + "/*meta.csv")[0])

        # Convert columns to categorical types
        metadata_all[params.bio_col] = metadata_all[params.bio_col].astype('category')
        if params.batch_col in metadata_all.columns:
            metadata_all[params.batch_col] = metadata_all[params.batch_col].astype('category')
        
        ## This is ugly and could break things.
        if "sampleID" in metadata_all.columns:
            metadata_all['batch'] = metadata_all['sampleID'].astype('category')

        # Print the number of unique batches
        print("Number of batches:", len(np.unique(metadata_all[params.batch_col])))

        # Define One Hot Encoded (OHE) order for donor and celltype categories
        params.batch_col_categories = np.unique(metadata_all[params.batch_col]).tolist() ## V2": batch_col_categories
        print("Ordered batches (donors):", params.batch_col_categories)

        params.bio_col_categories = np.unique(metadata_all[params.bio_col]).tolist() ## V2 : bio_col_categories


        # --------------------------------------------------------------------------------------
        # 1. Get Input Paths and Latent Paths
        # --------------------------------------------------------------------------------------
        # df_latent = get_latent_paths_df(results_path_dict, k_folds=params.fold_list[-1]) # <---------------- not sure this is the correct path
        # df_inputs = get_input_paths_df(splits_path, k_folds=params.fold_list[-1], eval_test=True)
        df_latent = get_latent_paths_df(results_path_dict) # <---------------- not sure this is the correct path
        df_inputs = get_input_paths_df(splits_path, eval_test=True)

        # Merge latent and input paths
        df = pd.merge(df_latent, df_inputs, on=["Split", "Type"], how="left")
        print("Reading paths,\ndf paths:\n", df.head(5))
    
        # Create latent path dictionary
        params.latent_path_dict = create_latent_dict_from_df(df_latent)
        params.save_model = save_model
        params.base_path = splits_path
        # --------------------------------------------------------------------------------------
        # 4. Run the Classifier for All Folds Latent Space
        # --------------------------------------------------------------------------------------
        all_folds_metrics_df = pd.DataFrame()


        for fold in params.fold_list:
            print("fold", fold)

            # Initialize model manager
            model_manager = ModelManager(
                self.model_params,
                base_paths_dict,
                params.run_name,
                save_model=save_model,
                use_kfolds=True,
                kfold=fold
            )

            # Update LatentClassifier config
            params.model_params = model_manager.params
            pipeline_LatentClassifier_config = vars(params)#{**load_latent_spaces_dict, **LatentClassifier_config}
            print(params)
            #pipeline_LatentClassifier_config['build_model_dict'] = None
            # Run pipeline
            results = run_model_pipeline_LatentClassifier_v2_PCA(
                 fold = fold, 
                 compile_dict=self.compile_configs,
                 build_model_dict={k:v for k, v in self.model_configs._asdict().items() if k not in [
                     "ignore", 'latent_keys_config', "return_metrics", "return_adata_dict", "return_trained_model", 
                     "model_type", "seed", "latent_path_dict", "model_params", "base_path", "fold", "models_list", 
                     "batch_col_categories", "bio_col_categories", 
                     ]},
                **pipeline_LatentClassifier_config,
                
                )
            results["metrics"]["fold"] = fold

            # Append metrics
            all_folds_metrics_df = pd.concat([all_folds_metrics_df, results["metrics"]], ignore_index=True)

            # # Clear memory
            # gc.collect()

        # --------------------------------------------------------------------------------------
        # 5. Save All Folds Metrics Results
        # --------------------------------------------------------------------------------------
        # output_path = os.path.join(
        #     load_latent_spaces_dict["model_params"].latent_path_main,
        #     "metrics_allfolds.csv"
        # )
        all_folds_metrics_df.to_csv(os.path.join(latent_space_base,params.run_name, "metrics_allfolds.csv"))
        # print("\nall_folds_metrics_df:", all_folds_metrics_df)

        # # --------------------------------------------------------------------------------------
        # # 6. Calculate and Save 95% CI
        # # --------------------------------------------------------------------------------------
        results_df = calculate_metrics_with_ci(all_folds_metrics_df)
        # output_path = os.path.join(
        #     load_latent_spaces_dict["model_params"].latent_path_main,
        #     "metrics_allfolds_95CI.csv"
        # )
        results_df.to_csv(os.path.join(latent_space_base,params.run_name,"metrics_allfolds_95CI.csv"))
        print("\nresults_df:", results_df)

        pass