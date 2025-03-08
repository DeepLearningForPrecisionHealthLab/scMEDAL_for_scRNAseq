# Experiment Reproducibility Guide

This guide provides instructions for reproducing the experiments described in our paper. It maps each section of the paper to the corresponding code and datasets, ensuring that you can run the models and scripts as described.

We recommend using the **Acute Myeloid Leukemia (AML)** dataset as a **DEMO** for running our model, as it is the smallest dataset used in our study. To adjust the number of training epochs, modify the `model_config.py` file. In our experiments, we used **500 epochs** with early stopping, which may halt training earlier.

- [AML Model Configuration for **scMEDAL-FE**](./Experiments/AML/run_models/scMEDAL-FE/model_config.py)  
- [AML Model Configuration for **scMEDAL-RE**](./Experiments/AML/run_models/scMEDAL-RE/model_config.py)  

Once the 5-fold splits are set up, the estimated runtime for all folds with **500 epochs and early stopping** is approximately **8 minutes for scMEDAL-FE** and **24 minutes for scMEDAL-RE** on an **Nvidia Tesla P4 GPU with 8 GB of memory**.


Our documentation also includes:  
- **Instructions on setting up experiments:** See **[How to Set Up Your Experiment](./docs/How2SetupYourExpt.md)**.  
- **Guidance on analyzing and interpreting model outputs:** See **[How to Analyze Your Model Outputs](./docs/How2AnalyzeYourModelOutputs.md)**.  
- **Step-by-step instructions for running experiments:** See **[How to Run Your Experiment](./docs/How2RunYourExpt.md)**.  
- **Details on experiment outputs:** See **[Experiment Outputs](./docs/ExperimentOutputs.md)**.  

---
## Experiments overview

We conducted experiments on three datasets: **Healthy Heart**, **Autism Spectrum Disorder (ASD)**, and **Acute Myeloid Leukemia (AML)**. For each dataset, we provide:

### Scripts  

- Preprocessing scripts and 5-fold cross-validation scripts to split the data in train, val and test.
- Directories and scripts to run various models (e.g., Autoencoder, scMEDAL variants, Mixed Effects Classifier).
- Scripts to generate and compare results (e.g., clustering scores, Genomaps, and UMAP plots).
- Example configuration files (`model_config.py`) for each model, along with a list of variables and hyperparameters needed to reproduce our experiments. You can review these details [here](scMEDAL_user_variables.pdf).


### Data  

We have included:  
- The 5 cross-validation splits metadata  
- The highly variable genes (HVGs) selected for each experiment 

These are available in:  

```
./Experiments/<dataset>_data/<scenario_id>
```
We also provide links to access the public datasets used in this study.
You can either run the 5-fold cross-validation scripts or use the cell ids provided to generate splits.


**Note:** Due to variability in TensorFlow, model outputs may differ slightly across runs. To account for this, we report 95% confidence intervals (CI) as an estimate of variability.

---

## Healthy Heart Dataset

**Data source:**  
The Healthy Heart dataset is available from Yu et al. (2023) at [figshare](https://doi.org/10.6084/m9.figshare.20499630.v2).

### Preprocessing
- [Preprocessing Notebook](../Experiments/HealthyHeart/preprocessing/preprocessing_HealthyHeart.ipynb)
- [5-Fold Cross-Validation Setup](../Experiments/HealthyHeart/preprocessing/5fold_cross_val)

### Models and scripts to reproduce results sections (RS)
---

**RS 2.2: scMEDAL subnetworks create complementary batch-invariant and batch-specific latent spaces in the Healthy Heart dataset**
- [Autoencoder (AE)](../Experiments/HealthyHeart/run_models/AE)
- [Fixed Effects Subnetwork (scMEDAL-FE)](../Experiments/HealthyHeart/run_models/scMEDAL-FE)
- [Random Effects Subnetwork (scMEDAL-RE)](../Experiments/HealthyHeart/run_models/scMEDAL-RE)

**RS 2.6: Improved cell classification accuracy using complementary latent spaces of scMEDAL**
- [Mixed Effects Classifier (MEC)](../Experiments/HealthyHeart/run_models/MEC)

**RS 2.7: The AE classifier, scMEDAL-FEC, enhances cell type preservation**
- [Autoencoder Classifier (AEC)](../Experiments/HealthyHeart/run_models/AEC)
- [Fixed Effects Subnetwork with Cell Type Classifier (scMEDAL-FEC)](../Experiments/HealthyHeart/run_models/scMEDAL-FEC)

### Comparison scripts
- RS 2.2 and 2.7 [Clustering scores](../Experiments/HealthyHeart/run_models/compare_results/clustering_scores)
- RS 2.5 [Generate genomaps](../Experiments/HealthyHeart/run_models/compare_results/genomaps)
- RS 2.2 and 2.7 [Generate UMAPs](../Experiments/HealthyHeart/run_models/compare_results/umap_plots)

For details on setting input and output paths for the Healthy Heart dataset, please refer to the [path setup instructions].

---

## Autism Spectrum Disorder (ASD) dataset

**Data source:**  
The ASD dataset can be accessed via the UCSC Cell Browser: [https://autism.cells.ucsc.edu](https://autism.cells.ucsc.edu)  
*(Speir et al., 2021; Velmeshev et al., 2019)*

### Preprocessing
- [Preprocessing Script](../Experiments/ASD/preprocessing/preprocess_ASD.py)
- [5-Fold Cross-Validation Setup](../Experiments/ASD/preprocessing/5fold_cross_val)

### Models and scripts to reproduce results sections

**RS 2.3: scMEDAL's components reflect disease-associated neuronal patterns in ASD**
- [Autoencoder (AE)](../Experiments/ASD/run_models/AE)
- [Fixed Effects Subnetwork (scMEDAL-FE)](../Experiments/ASD/run_models/scMEDAL-FE)
- [Random Effects Subnetwork (scMEDAL-RE)](../Experiments/ASD/run_models/scMEDAL-RE)

**RS 2.6: Improved cell classification accuracy using complementary latent spaces of scMEDAL**
- [Mixed Effects Classifier (MEC)](../Experiments/ASD/run_models/MEC)

**RS 2.7: The AE classifier, scMEDAL-FEC, enhances cell type preservation**
- [Autoencoder Classifier (AEC)](../Experiments/HealthyHeart/run_models/AEC)  
  *(Note: This link points to the Healthy Heart directory. Please ensure the correct path for ASD models.)*
- [Fixed Effects Subnetwork with Cell Type Classifier (scMEDAL-FEC)](../Experiments/HealthyHeart/run_models/scMEDAL-FEC)  
  *(Note: Similarly, ensure the correct ASD directory is used.)*

### Comparison scripts
- RS 2.3 and 2.7 [Clustering scores](../Experiments/ASD/run_models/compare_results/clustering_scores)
- RS 2.5 [Generate genomaps and compute genomap statistics](../Experiments/ASD/run_models/compare_results/genomaps)
- RS 2.3 and 2.7 [Generate UMAPs](../Experiments/ASD/run_models/compare_results/umap_plots)

---

## Acute Myeloid Leukemia (AML) dataset

**Data source:**  
The AML dataset is available at the Gene Expression Omnibus (GEO) under accession number [GSE116256](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256) (van Galen et al., 2019).

### Preprocessing
- [AML data reader Notebook](../Experiments/AML/preprocessing/AML_data_reader.ipynb)
- [Preprocessing Script](../Experiments/AML/preprocessing/preprocess_AML.py)
- [5-Fold Cross-Validation Setup](../Experiments/AML/preprocessing/5fold_cross_val)

### Models and Scripts to Reproduce Results Sections (RS)

**RS 2.4 scMEDAL balances the trade-off between batch correction and cell type information preservation in leukemia**
- [Autoencoder (AE)](../Experiments/AML/run_models/AE)
- [Fixed Effects Subnetwork (scMEDAL-FE)](../Experiments/AML/run_models/scMEDAL-FE)
- [Random Effects Subnetwork (scMEDAL-RE)](../Experiments/AML/run_models/scMEDAL-RE)

**RS 2.6: Improved cell classification accuracy using complementary latent spaces of scMEDAL**
- [Mixed Effects Classifier (MEC)](../Experiments/AML/run_models/MEC)
  - [Cell Type Target](../Experiments/AML/run_models/MEC/celltype_target)
  - [Patient Group Target](../Experiments/AML/run_models/MEC/dx_target)

**RS 2.7: The AE classifier, scMEDAL-FEC, enhances cell type preservation**
- [Autoencoder Classifier (AEC)](../Experiments/AML/run_models/AEC)
- [Fixed Effects Subnetwork with Cell Type Classifier (scMEDAL-FEC)](../Experiments/AML/run_models/scMEDAL-FEC)

### Comparison scripts
- RS 2.4 and 2.6 [Clustering scores](../Experiments/AML/run_models/compare_results/clustering_scores)
- RS 2.5 [Generate genomaps and compute genomap statistics](../Experiments/AML/run_models/compare_results/genomaps)
- RS 2.4 and 2.6[Generate UMAPs](../Experiments/AML/run_models/compare_results/umap_plots)

---

## Note on variability

Due to the inherent variability in TensorFlow, results may differ slightly each time you train the models. To account for this, we have computed 95% confidence intervals to provide an estimate of the variability in model performance.

---

## References

- Litvinukova, M. et al. Cells of the adult human heart. Nature 588, 466-472 (2020).
- van Galen, P. et al. *Single-Cell RNA-Seq Reveals AML Hierarchies Relevant to Disease Progression and Immunity.* Cell 176, 1265?1281.e24 (2019).
- Velmeshev, D. et al. *Single-cell genomics identifies cell type-specific molecular changes in autism.* Science 364, 685?689 (2019).
- Speir, M. L. et al. *UCSC Cell Browser: visualize your single-cell data.* Bioinformatics 37, 4578?4580 (2021).
- Yu, X., Xu, X., Zhang, J., & Li, X. *Batch alignment of single-cell transcriptomics data using deep metric learning.* Nat Commun 14, 960 (2023).  
- Yu, X., Xu, X., Zhang, J., & Li, X. *Batch alignment of single-cell transcriptomics data using deep metric learning.* figshare [https://doi.org/10.6084/m9.figshare.20499630.v2](https://doi.org/10.6084/m9.figshare.20499630.v2) (2023).