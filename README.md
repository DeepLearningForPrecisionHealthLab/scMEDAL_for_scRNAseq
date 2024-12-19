# **scMEDAL: Mixed Effects Deep Autoencoder Learning Framework**
### For details on how to reproduce our experiments, see: **[ExperimentsReproducibility](./docs/ExperimentsReproducibility.md)**

---
## **Overview**
The single-cell Mixed Effects Deep Autoencoder Learning (**scMEDAL**) framework provides a robust approach to analyze single-cell RNA sequencing (scRNA-seq) data. By disentangling batch-invariant from batch-specific signals, scMEDAL offers a more interpretable representation of complex datasets.



![scMEDAL Diagram](./docs/images/scMEDAL.png)

---

## **1. Framework Overview**

### **Fixed Effects Subnetwork (scMEDAL-FE)**
- Captures features that remain consistent across batches.
- Uses adversarial learning to minimize batch label predictability, ensuring batch-invariant latent representations.

### **Random Effects Subnetwork (scMEDAL-RE)**
- Models batch-specific variability using variational inference.
- Regularizes the latent space to accurately represent batch-specific patterns without overfitting.

---

## **2. scMEDAL Setup and Installation**

General structure of the repository:

```markdown
scMEDAL_for_scRNAseq/
|-- Experiments/               # Scripts and notebooks for experiments
|-- scMEDAL/                   # Main package
|   |-- __init__.py
|   |-- models/                # Model definitions
|   |    |-- __init__.py
|   |    |-- scMEDAL/
|   |    |-- models/
|   |-- utils/                 # Utilities for preprocessing, training, etc.
|   |    |-- __init__.py
|
|-- scMEDAL_env/               # Environment YAML files
|-- setup.py                   # Package setup
```

### **Installing `scMEDAL`**
1. **Activate Your Environment**  
   ```bash
   conda activate your_env_name
   ```

2. **Install in Editable Mode**  
   Navigate to the `scMEDAL_for_scRNAseq` directory and install:
   ```bash
   cd /path/to/scMEDAL_for_scRNAseq
   pip install -e .
   ```

3. **Verify Installation**
   ```python
   from scMEDAL.utils import your_function
   print("scMEDAL is ready to use!")
   ```

---

## **3. Execution Environments**

To handle dependency conflicts, `scMEDAL` uses three separate Conda environments:

1. **`genomaps_env`**: For generating Genomaps.
2. **`preprocess_and_plot_umaps_env`**: For data preprocessing and UMAP visualization.
3. **`run_models_env`**: For data splitting and running models.

### **Setting Up the Environments**
1. Navigate to the `scMEDAL_env` directory:
   ```bash
   cd /path/to/scMEDAL_for_scRNAseq/scMEDAL_env
   ```

2. Create each environment:
   ```bash
   conda env create -f genomaps_env.yaml
   conda env create -f preprocess_and_plot_umaps_env.yaml
   conda env create -f run_models_env.yaml
   ```

3. Activate the desired environment:
   ```bash
   conda activate genomaps_env

   ```
   or 
   ```bash
   conda activate preprocess_and_plot_umaps_env
   ```
   or
   ```bash
   conda activate run_models_env
   ```

### **Switching Environments**
- Use the environment that matches the script or task you are running.
- Ensure that all environments have the `scMEDAL` package installed (Step 2 above).
- Configure Slurm scripts to load the correct Conda environment before execution.

---

## **4. scMEDAL Utilities and Modules**

### **Utilities**
- **[utils.py](./scMEDAL/utils/utils.py):** Provides data I/O, plotting, and clustering score functions.
- **[model_train_utils.py](./scMEDAL/utils/model_train_utils.py):** Functions for training and loading models.
- **[splitter.py](./scMEDAL/utils/splitter.py):** Utility for k-fold cross-validation splitting.
- **[callbacks.py](./scMEDAL/utils/callbacks.py):** Tracks clustering metrics during training.
- **[compare_results_utils.py](./scMEDAL/utils/compare_results_utils.py):** Combines clustering results from multiple models.
- **[genomaps_utils.py](./scMEDAL/utils/genomaps_utils.py):** Custom Genomap generation functions.
- **[preprocessing_utils.py](./scMEDAL/utils/preprocessing_utils.py):** Preprocessing routines for datasets.
- **[utils_load_model.py](./scMEDAL/utils/utils_load_model.py):** Utilities for loading trained models.

### **Models**
- **[scMEDAL.py](./scMEDAL/models/scMEDAL.py):** Implements AEC, DA_AE, and DomainEnhancingAutoencoderClassifier models.
- **[random_effects.py](./scMEDAL/models/random_effects.py):** Bayesian layers and utilities for random effects modeling.

---

## **5. Experiment Setup**
This setup will allow you to run our models in the Healthy Heart, ASD and AML datasets.
**Experiment Folder Structure**: Each dataset-specific experiment follows a standard directory layout:

```markdown

scMEDAL_for_scRNAseq/
|-- Experiments/ 
   |--  data/ # Download and Setup your data folders
   |-- outputs 
   |--  <dataset_name>/
      |-- preprocessing/
      |   |-- 5fold_cross_val/
      |   |   |-- create_splits.ipynb
      |   |   |-- check_splits.ipynb
      |   |   |-- config_split_paths.py
      |   |-- preprocess_datasetname.py
      |   |-- batch_preprocess_dataset.sh
      |   |-- preprocess_datasetname.ipynb
      |-- run_models/
      |   |-- AE/
      |   |-- AEC/
      |   |-- scMEDAL-FEC/
      |   |-- scMEDAL-FE/
      |   |-- scMEDAL-RE/
      |   |-- compare_results/
      |   |   |-- clustering_scores/
      |   |   |-- genomaps/
      |   |   |-- umap_plots/
      |   |-- MEC/
      |       |-- target/
      |           |-- scMEDAL-FEandscMEDAL-RE_latent/
      |           |-- scMEDAL-FE/
      |           |-- PCA_latent/
      |-- paths_config.py
   
```

- **`data/`**  
   - *(Download and set up your data folders here.)*
- **`outputs/`**  
   - *(This folder will be created automatically when running: `import outputs_path` from `paths_config`.)*
- **`datasetname/`**  
   - Folder with scripts to preprocess and run models.


For instructions on setting up experiments, see **[How2SetupYourExpt](./docs/How2SetupYourExpt.md)**.

### **Model Configuration**
Each model directory contains a `model_config.py` file that specifies settings and paths. For example:  
- [Healthy Heart AE Model Configuration](./Experiments/HealthyHeart/run_models/AE/model_config.py)

---

## **6. Dataset-Specific Instructions**

To set up the datasets for your experiments, follow these steps:

1. **Download the datasets** from the provided sources.  
2. **Save them in the appropriate directories** under the main folder: **`/Experiments/data`**.  
   - If the required subfolders do not exist, create them before saving the datasets.

### **Datasets and Sources**

- **Healthy Human Heart**  
  - Source: [Figshare](https://figshare.com/articles/dataset/Batch_Alignment_of_single-cell_transcriptomics_data_using_Deep_Metric_Learning/20499630/2)  
  - Save the dataset in: `/Experiments/data/HealthyHeart_data`  
  - *Note: Create the folder `HealthyHeart_data` if it does not already exist.*

- **Autism Spectrum Disorder (ASD)**  
  - Source: [Autism Cell Atlas](https://autism.cells.ucsc.edu)  
  - Save the dataset in: `/Experiments/data/ASD_data`  
  - *Note: Create the folder `ASD_data` if it does not already exist.*

- **Acute Myeloid Leukemia (AML)**  
  - Source: [GEO: GSE116256](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256)  
  - Save the dataset in: `/Experiments/data/AML_data`  
  - *Note: Create the folder `AML_data` if it does not already exist.*

---

## **7. Running Models and Experiments**

You can run AE, AEC, scMEDAL-FE, scMEDAL-FEC, or scMEDAL-RE independently. PCA can be generated simultaneously by setting `"get_pca": True` in `config.py`.

The MEC model requires latent outputs from one of the above models; it cannot run independently.

### **Steps to Run Models**
1. **Run All Folds Locally:**
   ```bash
   python run_modelname_allfolds.py
   ```

2. **Submit Jobs via Slurm:**
   ```bash
   sbatch sbatch_run_modelname.sh
   ```

For detailed instructions, see **[How2RunYourExpt](./docs/How2RunYourExpt.md)**.

### **Important Notes**
- Always activate the correct Conda environment before running scripts.

---

## **8. Experiment Outputs**

For more information about output files and their contents, refer to [ExperimentOutputs](./docs/ExperimentOutputs.md).

---

## **9. Analyzing Your Model Outputs**

For guidance on analyzing and interpreting model outputs, see [How2AnalyzeYourModelOutputs](./docs/How2AnalyzeYourModelOutputs.md).

---

## **10. References**
Nguyen KP, et al., 2023. *Adversarially-Regularized Mixed Effects Deep Learning Models Improve Interpretability*. [IEEE TPAMI](https://doi.org/10.1109/TPAMI.2023.3234291).
Litvinukova, M. et al. Cells of the adult human heart. Nature 588, 466?472 (2020).
van Galen, P. et al. Single-Cell RNA-Seq Reveals AML Hierarchies Relevant to Disease Progression and Immunity. Cell 176, 1265?1281.e24 (2019).
Velmeshev, D. et al. Single-cell genomics identifies cell type-specific molecular changes in autism. Science 364, 685?689 (2019).
Speir, M. L. et al. UCSC Cell Browser: visualize your single-cell data. Bioinformatics 37, 4578?4580 (2021).
Yu, X., Xu, X., Zhang, J., & Li, X. Batch alignment of single-cell transcriptomics data using deep metric learning. Nat Commun 14, 960 (2023).
Yu, X., Xu, X., Zhang, J., & Li, X. Batch alignment of single-cell transcriptomics data using deep metric learning. figshare https://doi.org/10.6084/m9.figshare.20499630.v2 (2023).
