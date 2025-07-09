# How to Analyze Your Model Outputs



## Compare Models Outputs

The outputs of the models's analysis are stored in the *compare_results* output directory. We compare the outputs of different models through: clustering scores, umaps and genomaps.

**File Structure:**
```markdown
<datasetname>/
|-- run_models/
    |-- compare_results/
        |-- clustering_scores/              # Scripts and data for clustering evaluation
            |-- compare_clustering_scores.py
        |-- genomaps/                       # Scripts and data for genomap generation
            |-- pipeline_CMmultibatch_genomap_and_plot_multicell.py
            |-- sbatch_get_genomaps.sh
        |-- umap_plots/                     # Scripts for UMAP visualization
            |-- compare_results_umap_v1.py
            |-- sbatch_compare_results_umap.sh
```

**Important Note on Paths:**  
Before running these scripts, update your `paths_config.py` with the actual run names for each model. After training each model, check the output folder to find the directory with the unique timestamped run name. Use that timestamped name in the configuration.

An example configuration snippet might look like this:

```python
expt = "expt_test"

if expt == "expt_test":
    scaling = "min_max"

    # Replace with your actual run names obtained from your output directories
    run_names_dict = {
        "scMEDAL-RE": "scMEDAL-RE_run_name",
        "run_name_all": "DefineGeneralNameForYourExpt"
    }

    # Set this to True if you plan to calculate clustering scores
    calculate_clustering_scores = True

    # If calculating clustering scores, include other models as needed
    if calculate_clustering_scores:
        run_names_dict.update({
            "AE": "AE_run_name",
            "AEC": "AEC_run_name",
            "scMEDAL-FEC": "scMEDAL-FEC_run_name",
            "scMEDAL-FE": "scMEDAL-FE_run_name"
        })
```

**Default Dataset Type:**  
By default, `dataset_type = 'test'`.

---

## 1. Comparing Clustering Scores

After your models have finished training, you may want to consolidate clustering score results to compare them more easily.

**Run Locally:**
```bash
python compare_clustering_scores.py
```

**Submit via SLURM:**
```bash
sbatch sbatch_run_<modelname>.py
```

**Environment Note:**  
These scripts are run using the `run_models_env`.

---

## 2. Computing Genomaps

In `pipeline_CMmultibatch_genomap_and_plot_multicell.py`, update the following parameters as needed:

```python
# Define models, types, and splits for the genomap script
models = ['scMEDAL-RE']   # Add all your models to this list
types = ['train']         # Add all data splits you need to iterate through
splits = [2]              # Example: using fold 2

# Set to true if you want to use input data and scMEDAL-FE reconstructions to generate the genomap.
add_inputs_fe = True
extra_recon = "fe"  # options: "fe" or "all"

# Specify cell types
celltype = ["Ventricular_Cardiomyocyte", "Endothelial", "Fibroblast", "Pericytes"]
n_cells_per_batch = 300
n_batches = 147

# Define the genomap dimensions and number of genes
n_genes = 2916
colNum = 54
rowNum = 54

# Optionally specify batches if needed
batches_to_select_from = ["H0037_Apex", "HCAHeart7836681", "HCAHeart8102861", "H0015_septum"]
```

**Adjusting Genomap Iterations:**  

In the function `process_and_plot_genomaps_singlepath`, you can the number of iterations to generate Genomap transform. For our experiments, we used 100.

```python
process_and_plot_genomaps_singlepath(
    cm_multibatch_path,
    ncells=n_cells,
    ngenes=n_genes,
    rowNum=rowNum,
    colNum=colNum,
    epsilon=0.0,
    num_iter=100,  # Modify as needed. Iterations for genomap optimization.
    output_folder=path_2_genomap,
    genomap_name=genomap_name,
    gene_names=gene_names
)
```



To run the genomap pipeline:

**Run Locally:**
```bash
python pipeline_CMmultibatch_genomap_and_plot_multicell.py
```

**Submit via SLURM:**
```bash
sbatch sbatch_get_genomaps.sh
```

## 3. Computing UMAPs

 
In `compare_results_umap.py`, update the following parameters as needed:
```python
# Specify folds (splits) for filtering the data you want to plot. Default 2 for all models.
filter_folds = {
    "AE": 2,
    "AEC": 2,
    "scMEDAL-FEC": 2,
    "scMEDAL-FE": 2,
    "scMEDAL-RE": 2
}

# Filter data to include only specific models and splits for "train" data
filtered_df = filter_models_by_type_and_split(df, filter_folds, Type='train')



# Initialize dimensionality reduction processor
processor = DimensionalityReductionProcessor(
    filtered_df,
    umap_path,
    plot_params,
    sample_size=None, # Take a smaller sample size if you want faster results. Default = None. Takes all the cells.
    n_neighbors=15, # Change if needed.
    scaling="min_max", # Load data
    n_batches_sample=20, # Change according to the number of batches you want to plot
    batch_col="batch",
    plot_tsne=False, # Add tsne plots
    n_pca_components=2, # make sure this match the latent dimensions of you model. Default = 2.
    min_dist=0.5 # UMAP parameter
)

```


To run the umap pipeline:

**Run Locally:**
```bash
python compare_results_umap.py
```

**Submit via SLURM:**
```bash
sbatch sbatch_compare_results_umap.sh
```