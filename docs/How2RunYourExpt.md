# How to Run Your Experiment

This guide provides an overview of the folder structure, data organization, and the steps to run your experiments, including data preprocessing, creating splits for cross-validation, and training models.

## Folder Structure

```markdown
Experiments/
|-- data/
|   |-- <datasetname>_data/
|-- <datasetname>/                        # e.g., HealthyHeart, AML, ASD
|   |-- paths_config.py
|   |-- preprocessing/
|       |-- 5fold_cross_val/
|           |-- config_split_paths.py    # Configures input/output paths for splits
|           |-- create_splits.ipynb      # Splits the data into train/val/test
|           |-- check_splits.ipynb       # Checks for data leakage in splits
|       |-- preprocess_<datasetname>.py  # Main preprocessing script (Python)
|       |-- batch_preprocess_<datasetname>.sh # Optional SLURM script for batch preprocessing
|       |-- preprocess_<datasetname>.ipynb   # Optional preprocessing notebook (Jupyter)
|-- run_models/
|   |-- <modelname>/                     # Model-specific scripts (e.g., AE, AEC, etc.)
|       |-- model_config.py              # Model hyperparameters & output settings
|       |-- run_<modelname>_allfolds.py  # Runs the model pipeline for all folds
|       |-- sbatch_run_<modelname>.py    # SLURM script to run the model
|-- outputs/
|   |-- <datasetname>_outputs/
```

### Special case: Mixed Effect classifier (MEC)

```markdown
Experiments/
|-- run_models/
    |-- <MEC>/                          # Mixed Effects Classifier
    |-- <target_type>_target/           # Target type for the model
    |-- <latent_space_combo>_latent/    # Combination of latent spaces
        |-- model_config.py             # Model hyperparameters & output settings
        |-- run_<modelname>_allfolds.py # Runs the model pipeline for all folds
        |-- sbatch_run_<modelname>.py   # SLURM script to run the model

```


where <target_type> in celltype, dx
and <latent_space_combo> in scMEDAL-FE, scMEDAL-FEandscMEDAL-RE, PCA

### Data Files

The count matrices for your datasets are stored as follows:

- `exprMatrix.npy`
- `geneids.csv`
- `meta.csv`

### Data Organization

Within `data/<datasetname>/`, your data might be structured like this:

```markdown
data/
|-- <datasetname>/
    |-- <countmatrixname>/                # Count matrix before preprocessing
        |-- exprMatrix.npy or exprMatrix.tsv
        |-- geneids.csv or geneids.tsv
        |-- meta.csv or meta.tsv
    |-- <scenario_id>/                    # Output of preprocessing scripts or notebooks
        |-- exprMatrix.npy
        |-- geneids.csv
        |-- meta.csv
        |-- splits/                       # Created by create_splits.ipynb
            |-- split_1/
                |-- test/                 # Each folder contains the count matrices for that split
                |-- train/
                |-- val/
            |-- split_2/
            |-- split_3/
            |-- split_4/
            |-- split_5/
```

Each `split_X/` directory contains train, val, and test subsets of the data.

---
## Running the Experiment

### Step 1. Setup you experiment
See instructions in **[How2SetupYourExpt](How2SetupYourExpt.md)**.

---

### Step 2: Preprocess Your Data

1. **Option A (Notebook)**:  
   Run the `preprocess_<datasetname>.ipynb` notebook (suitable for datasets like HealthyHeart).

2. **Option B (Python Script)**:  
   For other datasets (e.g., ASD, AML), use the Python script:  
   ```bash
   python preprocess_<datasetname>.py
   ```

3. **Option C (SLURM)**:  
   If you have a SLURM environment, run the batch preprocessing script:  
   ```bash
   sbatch batch_preprocess_<datasetname>.sh
   ```

**Environment**: Use the `preprocess_and_plot_umaps_env` environment for preprocessing.

**Note**: If you use SLURM, make sure to update the environment name in the SLURM script.

---

### Step 3: Create 5-Fold Cross-Validation Splits

1. Open and run `create_splits.ipynb` to generate the 5-fold splits.
2. Run `check_splits.ipynb` to verify that there is no data leakage.

**Environment**: Use the `run_models_env` environment for this step.

---

### Step 4: Run the Model


Each model has its own directory under `run_models/`. For example, `<modelname>/` might contain:

- `model_config.py`: Configure model hyperparameters, output paths, plotting parameters, and other settings. It also generates a unique `run_name` (with a timestamp) needed for analyzing outputs.
- `run_<modelname>_allfolds.py`: Executes the entire training pipeline for all 5 folds.
- `sbatch_run_<modelname>.py`: SLURM script to run the model training on a cluster.

**Customization Options**:

- Modify `model_config.py` to change hyperparameters (e.g., layer units, latent dimensions, epochs, how the model is loaded).
- If you only want to run a single fold (e.g., fold 1), adjust `run_<modelname>_allfolds.py` accordingly.

Once configured, run:

```bash
python run_<modelname>_allfolds.py
```

or submit via SLURM:

```bash
sbatch sbatch_run_<modelname>.py

```

The AE, AEC, scMEDAL-FE, scMEDAL-FEC, or scMEDAL-RE models can be run independently.

The PCA model can be run simultaneusly with another model. Just set "get_pca": True in config.py

```python
get_scores_dict = {

    "get_pca": True

}
```

The  The MEC model require `latent_space` the outputs from the AE, AEC, scMEDAL-FE, scMEDAL-FEC, or scMEDAL-RE models. It cannot run without them.  


**Environment Note**: Run the models using the `run_models_env`.

---
### Step 5: Check the Model's Outputs
Check more details about the outputs in **[ExperimentOutputs](ExperimentOutputs.md)**

---

**Summary**:

1. **Preprocessing**: Prepares data and may produce processed datasets and scenarios.
2. **Splitting Data**: Creates train/val/test subsets for cross-validation.
3. **Running Models**: Trains models on the prepared splits and produces output results for further analysis. Each model's configuration and runtime settings can be adjusted to your needs.

Once the model run is complete, you'll have timestamped run names and corresponding outputs ready for downstream analysis (such as generating confidence intervals, UMAP visualizations, and Genomaps). See more instructions in **[How2AnalyzeYourModelOutputs](How2AnalyzeYourModelOutputs.md)**.




