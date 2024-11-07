# Mixed Effects Deep Learning (MEDL) Autoencoder for Interpretable Analysis of Single-cell RNA Sequencing (scRNA-seq) Data

Updated README description and experiments on 11/7/2024

## Description

The Mixed Effects Deep Learning (MEDL) framework is designed to extract meaningful latent representations from single-cell RNA sequencing (scRNA-seq) data while accounting for batch effects, which are common confounders that can obscure true biological signals. By extending linear mixed-effects models into a nonlinear context, MEDL effectively addresses the inherent non-linearity in confounded scRNA-seq datasets.

**Key Features:**
- **Inputs:** Gene expression count matrix $X \in \mathbb{R}^{n \times m}$, where $n$ is the number of cells and $m$ is the number of genes.

- **Outputs:** Reconstructed gene expression matrix $\hat{X}$ and latent space representations.

- **Latent Space:** Reduced feature space of size $n \times p$.

## MEDL Framework Overview

The MEDL framework consists of two parallel subnetworks that jointly model fixed and random effects within gene expression data:

1. **Fixed Effects Subnetwork (MEDL-AE-FE):** This subnetwork captures batch-invariant features by suppressing batch effects. It employs an autoencoder with weight tying, dense hidden layers, and an adversarial classifier that learns to predict batch labels. The loss function balances reconstruction error and adversarial loss to mitigate batch-specific variations.

2. **Random Effects Subnetwork:** This subnetwork models batch-specific variations using variational inference. It approximates true batch distributions with optimized surrogate posteriors and includes a classifier for batch label prediction. By maximizing the Evidence Lower Bound (ELBO), the model ensures that the latent space encodes batch-specific information while regularizing to prevent overfitting.

![MEDL Diagram](./images/MEDL.png)

### Subnetwork Details


### Fixed Effects Loss Function
The fixed effects loss function combines reconstruction accuracy with suppression of batch effects to capture fixed effects:
$$
L_{\text{FE}} = \lambda_{\text{MSE}} \cdot L_{\text{MSE}}(X, \hat{X}) - \lambda_{\text{A}} \cdot L_{\text{CCE}}(z, \hat{z})
$$

- **$L_{\text{MSE}}(X, \hat{X})$**: Mean Squared Error (MSE) to ensure accurate reconstruction of the original gene expression matrix $X$ by minimizing the difference from $\hat{X}$, the reconstructed matrix.
- **$\lambda_{\text{MSE}}$**: Weight for reconstruction accuracy.
- **$L_{\text{CCE}}(z, \hat{z})$**: Categorical Cross-Entropy (CCE) loss for the adversarial classifier, discouraging batch label predictability in the latent space.
- **$\lambda_{\text{A}}$**: Weight for the CCE term, controlling the model's emphasis on batch effect suppression.

### Random Effects Loss Function
The random effects loss function incorporates reconstruction, batch classification, and regularization to capture batch-specific variations:
$$
L_{\text{RE}} = \lambda_{\text{MSE}} \cdot L_{\text{MSE}}(X, \hat{X}') + \lambda_{\text{CCE}} \cdot L_{\text{CCE}}(z, \hat{z}') + \lambda_{\text{KL}} \cdot D_{\text{KL}}(q(U) \parallel p(U))
$$

- **$L_{\text{MSE}}(X, \hat{X}')$**: Mean Squared Error to ensure accurate reconstruction in the random effects subnetwork.
- **$\lambda_{\text{MSE}}$**: Weight for the reconstruction term.
- **$L_{\text{CCE}}(z, \hat{z}')$**: Categorical Cross-Entropy (CCE) to encourage batch-specific feature encoding by training the classifier to predict batch labels.
- **$\lambda_{\text{CCE}}$**: Weight for the CCE term, balancing emphasis on batch classification.
- **$D_{\text{KL}}(q(U) \parallel p(U))$**: Kullback-Leibler (KL) divergence, regularizing the model by aligning the learned distribution $q(U)$ with the prior distribution $p(U)$.
- **$\lambda_{\text{KL}}$**: Weight for the KL divergence term, controlling the degree of regularization to prevent overfitting.


## Usage

To implement the MEDL framework:

1. **Data Preparation:** Ensure your dataset includes a gene expression count matrix and corresponding batch labels (e.g., donor IDs, experimental batches).

2. **Model Training:** Use the provided MEDL-AE code to train on your data. The model outputs reconstructed gene expression matrices and latent space representations with reduced batch effects.

3. **Downstream Analysis:** Utilize the latent representations for tasks like clustering, visualization, or differential expression analysis to uncover true biological signals.



By leveraging the MEDL framework, researchers can achieve more accurate and interpretable analyses of scRNA-seq data, effectively separating true biological variability from technical noise due to batch effects.

### Models
* [AE_v4.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/models/AE_v4.py): Contains models including the simple AEC (Autoencoder Classifier), DA_AE (Domain Adversarial Autoencoder for fixed effects), and the DomainEnhancingAutoencoderClassifier (for Random Effects).
* [random_effects.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/models/random_effects.py): Implements random effects classes, originally developed by Kevin Nguyen for the ARMED paper.

### Utilities
* [utils.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/utils/utils.py)
  * Functions for reading and saving data, plotting, and calculating clustering scores.
* [model_train_utils.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/utils/model_train_utils.py)
  * Functions to load data, build, and train models using configuration information.
* [splitter.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/utils/splitter.py)
  * Function to split data into training, validation, and test sets. Includes support for 5-fold cross-validation.
* [utils_load_model.py](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/utils/utils_load_model.py)
  * Functions to load previously saved models.

# Experiment Files

## Healthy Heart dataset
Access the Healthy Human Heart dataset utilized in this experiment from [here](https://figshare.com/articles/dataset/Batch_Alignment_of_single-cell_transcriptomics_data_using_Deep_Metric_Learning/20499630/2) (Yu et al., 2023).

### 5-Fold Cross-Validation
- **[5-Fold Cross-Validation Directory](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/preprocessing/5fold_cross_val)**
  - [Create Splits Notebook](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/heart_data/preprocessing/5fold_cross_val/create_splits.ipynb): A notebook for splitting data using 5-fold cross-validation.
  - [Config Split Paths Script](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/heart_data/preprocessing/5fold_cross_val/config_split_paths.py): Manages paths for input data and data splits.
  - [Check Splits Notebook](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/heart_data/preprocessing/5fold_cross_val/check_splits.ipynb): Ensures no data leakage between training, testing, and validation sets.

Explore the models used in the Heart Data dataset:
### Experiment 1: Characterize the fixed and random effect latent spaces and visualizing learned batches
- **[Run Models Directory](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models)**
  - [Autoencoder (AE)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AE)
  - [Autoencoder Classifier (AEC)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AEC)
  - [Random Effects Subnetwork (AE_RE)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AE_RE)

### Experiment 2: Evaluate the complementary nature of the FE and RE latent representations for enhancing prediction performance
- **[Run Models Directory](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models)**
  - [Mixed Effects Classifier (MEC)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/MEC/celltype_target)

### Experiment 3: Quantify the impact of a cell type classifier on batch and cell type separability
- **[Run Models Directory](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models)**
  - [MEDL Autoencoder Classifier (MEDL-AEC)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AEC)
  - [Fixed Effects Subnetwork (MEDL-AEC-FE)](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AEC_DA)


### Scripts to compare models
Explore the models used in the Heart Data dataset:
- **[Compare Results Directory](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/compare_results)**
  - [Clustering scores](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/compare_results/clustering_scores)
  - [Generate Genomaps](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/compare_results/genomaps)
  - [Generate UMAPs](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/tree/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/compare_results/umap_plots)




# Experiment Configuration

### Model Configuration Script

- **Model Configuration**
  - Each model has its own `model_config.py` file. For an example, see the [Health Heart AE model configuration](https://git.biohpc.swmed.edu/s437576/armed_genomics_git/-/blob/main/heart_data/run_models/Healthy_human_heart/log_transformed_3000hvggenes/AE/model_config.py).
  - This script updates model settings and defines output paths.

Make sure you update your data paths in `model_config.py`:

```python
data_base_path = "path to base path with all variants of the experiment"
scenario_id = "path to specific pre-processing of your experiment"
```

## Construct the Data Paths

- **Path to the data from the experiment you want to run:**

```python
data_path = os.path.join(data_base_path, scenario_id)
```

- **Path to the folder that contains the specific splits (for k-fold cross validation you are running):**

```python
data_seen = os.path.join(data_base_path, scenario_id, 'splits')
print(f"Parent folder: {data_seen}")
```

## Base Output Path

Update the following variable in your script:

```python
outputs_path = "/path/to/outputs"
```

## Folder and Model Naming

Define how you want to name your experiment and the model folder name:

```python
folder_name = "how you want to name your expt"
model_name = "AE_RE"  # the folder name of your output
```

# Script Execution

- **Run Model Across All Folds**
  - Command: `python run_modelname_allfolds.py`
- **Slurm Submission Script**
  - Command: `sbatch sbatch_run_modelname.sh`

### Execution Environment

- The experiments are executed within the ARMED_Aixa_v2 environment.

### Notes

Make sure to change the path to the utils folder in each of the files of the type: `run_modelname_allfolds.py`.


