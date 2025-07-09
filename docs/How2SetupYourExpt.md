

# How to Set Up Your Experiment

This document describes how to organize and configure experiments, including how to preprocess data, adjust paths, and update run names for post-analysis tasks (e.g., generating 95% confidence intervals, UMAPs, and Genomaps).

## General Folder Structure

``` markdown

Experiments/
|-- data/
|   |-- <datasetname>_data/
|-- <datasetname>/                       # Replace <datasetname> with your chosen experiment name (e.g., HealthyHeart, AML, ASD)
|   |-- paths_config.py
|   |-- preprocessing/
|   |-- run_models/
|       |-- AE/                          # Autoencoder model scripts/results
|       |-- AEC/                         # Autoencoder Classifier
|       |-- scMEDAL-FEC/                 # Autoencoder Classifier with Fixed Effects
|       |-- scMEDAL-FE/                  # Autoencoder with Fixed Effects
|       |-- scMEDAL-RE/                  # Autoencoder with Random Effects
|       |-- compare_results/
|           |-- clustering_scores/       # Scripts and data for clustering evaluation
|           |-- genomaps/                # Scripts and data for genomap generation
|           |-- umap_plots/              # Scripts for UMAP visualization
|-- outputs/
|   |-- <datasetname>_outputs/
```

### Key Directories

Make sure you have downloaded and setup your data folders in **`/Experiments/data`**.  
   - If the required subfolders do not exist, create them before saving the datasets.

- **`data/`**: Holds your datasets. For example:
  - `data/HealthyHeart_data` for the HealthyHeart dataset.
     - Source: [Figshare](https://figshare.com/articles/dataset/Batch_Alignment_of_single-cell_transcriptomics_data_using_Deep_Metric_Learning/20499630/2) 
  - `data/ASD_data` for the AML dataset.
      - Source: [Autism Cell Atlas](https://autism.cells.ucsc.edu)  
  - `data/AML_data` for the ASD dataset.
    - Source: [GEO: GSE116256](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256)  

- **`<datasetname>/`**: Contains experiment-specific configurations and code:
  - **`paths_config.py`**: Defines paths to data, outputs, and scenario identifiers.
  - **`preprocessing/`**: Houses scripts and notebooks for preparing and cleaning your dataset.
  - **`run_models/`**: Contains scripts for training and evaluating various models.

- **`outputs/`**: Stores output results. For example:
  - `outputs/HealthyHeart_outputs` for the HealthyHeart results.
  - `outputs/AML_outputs` for AML results.
  - `outputs/ASD_outputs` for ASD results.

## Managing Relative Paths and Imports

In some scripts, especially those located in nested directories, you might see lines like:

```python
import sys
sys.path.append("../../")
```

or

```python
sys.path.append("../../../")
```

These adjustments ensure Python can locate shared modules or `paths_config.py` files located higher in the directory structure. If you change the folder layout or move scripts around, you must adjust these relative paths accordingly. For instance:

- Moving a script from `run_models/compare_results/umap_plots/` one level up might allow you to change `sys.path.append("../../../")` to `sys.path.append("../../")`.

Always verify that the paths align with your current directory structure.

## Configuring `paths_config.py`

The `paths_config.py` file is crucial for setting up correct paths to your data and outputs. It also defines scenario identifiers, run names, and other configuration details. Below is an example configuration for the HealthyHeart dataset:

```python
import os

# Get the directory of the current file (paths_config.py)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the data base path relative to the current file
data_base_path = os.path.join(base_dir, "../data/HealthyHeart_data")
print("data_base_path:", data_base_path)

# Specify the scenario_id to consistently represent a particular preprocessing setup
scenario_id = "log_transformed_3000hvggenes"
input_base_path = os.path.join(data_base_path, scenario_id, 'splits')

# Define the output paths for saving results
outputs_path = os.path.join(base_dir, "../outputs/HealthyHeart_outputs")
print("outputs_path:", outputs_path)
```


**What to Consider When Modifying Paths:**

- **`data_base_path`**: Points to your main data directory. If you rename or move your data folder, update this line accordingly.
- **`scenario_id`**: Specifies a particular scenario for your experiments (e.g., using a specific preprocessing or feature selection method). Change this to match your scenario directory structure inside `data_base_path`.
- **`outputs_path`**: Points to where your experiment results, model checkpoints, and figures will be saved. Adjust this if you move or rename the `outputs` folder.

If you add another level to your directory structure or move `paths_config.py` deeper into subdirectories, you might need to update `data_base_path` and `outputs_path` to ensure correct relative paths. For example, if you move `paths_config.py` one level deeper, you may need to add another `..` to the paths.

## Defining Unique Run Names for Experiments

Once you have run your models, each run is often associated with a unique timestamped name. To generate tables with 95% confidence intervals, UMAPs, or Genomaps, you need to identify the run name and update the `expt` section of `paths_config.py`.

For example:

```python
expt = "expt_test"

if expt == "expt_test":
    scaling = "min_max"

    # Unique run names with timestamps should be provided here
    run_names_dict = {
        "scMEDAL-RE": "scMEDAL-RE_run_name",
        "run_name_all": "DefineGeneralname4yourexpt"
    }

    # Set True if you plan to calculate clustering scores
    calculate_clustering_scores = True

    # If calculating clustering scores, add other models
    if calculate_clustering_scores:
        run_names_dict.update({
            "AE": "AE_run_name",
            "AEC": "AEC_run_name",
            "scMEDAL-FEC": "scMEDAL-FEC_run_name",
            "scMEDAL-FE": "scMEDAL-FE_run_name"
        })
```

**Steps to Update:**

1. **Set the Experiment Identifier (`expt`)**:  
   Choose a descriptive name for your experiment, e.g., `"expt_healthyheart_v1"`.

2. **Run Names**:  
   Replace the placeholder run names (e.g., `"scMEDAL-RE_run_name"`) with the actual run names generated during model training. These run names are typically created automatically by your training scripts and often include a timestamp.

3. **Add or Remove Models**:  
   If you run additional models (e.g., `scMEDAL-FE` or `AEC`), add their unique run names. If you choose not to calculate clustering scores, set `calculate_clustering_scores = False` and remove or omit the extra model entries.

4. **Regenerating Results**:  
   After updating `run_names_dict` with the correct run names, rerun your result generation scripts (e.g., scripts in `compare_results/clustering_scores/`, `umap_plots/`, or `genomaps/`), and they will use these updated run names to locate and process the correct results.

## Summary

- **Folder Structure**: Keep a consistent hierarchy and remember that some scripts rely on `sys.path.append` to import modules. If you move scripts around, adjust the relative paths.
- **Configuring Paths**: Update `paths_config.py` whenever you change data or output directories. This ensures all scripts know where to find input data and save outputs.
- **Updating Run Names**: After training models, update `run_names_dict` in `paths_config.py` with the correct unique run names. This is essential for generating summary tables, visualizations, and analysis plots.
