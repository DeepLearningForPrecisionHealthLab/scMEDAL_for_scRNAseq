# Experiment Reproducibility Guide

This guide provides instructions for reproducing the experiments described in our paper. It links each section of our paper to the corresponding code and datasets, ensuring that you can run the models and scripts as described.

## Overview

We conducted experiments on three datasets: **Healthy Heart**, **Autism Spectrum Disorder (ASD)**, and **Acute Myeloid Leukemia (AML)**. For each dataset, we provide:

- Preprocessing scripts and 5-fold cross-validation setups.
- Directories and scripts to run various models (e.g., Autoencoder, scMEDAL variants, Mixed Effects Classifier).
- Scripts to generate and compare results (e.g., clustering scores, Genomaps, and UMAP plots).

**Note:** Due to variability in TensorFlow, model outputs may differ slightly across runs. To account for this, we report 95% confidence intervals (CI) as an estimate of variability.

---

## Healthy Heart Dataset

**Data Source:**  
The Healthy Heart dataset is available from Yu et al. (2023) at [figshare](https://doi.org/10.6084/m9.figshare.20499630.v2).

### Preprocessing
- [Preprocessing Notebook](../Experiments/HealthyHeart/preprocessing/preprocessing_HealthyHeart.ipynb)
- [5-Fold Cross-Validation Setup](../Experiments/HealthyHeart/preprocessing/5fold_cross_val)

### Models and Scripts to Reproduce Results Sections (RS)
---

**RS: scMEDAL subnetworks create complementary batch-invariant and batch-specific latent spaces:**
- [Autoencoder (AE)](../Experiments/HealthyHeart/run_models/AE)
- [Fixed Effects Subnetwork (scMEDAL-FE)](../Experiments/HealthyHeart/run_models/scMEDAL-FE)
- [Random Effects Subnetwork (scMEDAL-RE)](../Experiments/HealthyHeart/run_models/scMEDAL-RE)

**RS: scMEDAL complementary latent spaces improve prediction accuracy at the cellular level:**
- [Mixed Effects Classifier (MEC)](../Experiments/HealthyHeart/run_models/MEC)

**RS: scMEDAL-FEC enhances cell type preservation in latent spaces:**
- [Autoencoder Classifier (AEC)](../Experiments/HealthyHeart/run_models/AEC)
- [Fixed Effects Subnetwork with Cell Type Classifier (scMEDAL-FEC)](../Experiments/HealthyHeart/run_models/scMEDAL-FEC)

### Comparison Scripts
- [Clustering Scores](../Experiments/HealthyHeart/run_models/compare_results/clustering_scores)
- [Generate Genomaps](../Experiments/HealthyHeart/run_models/compare_results/genomaps)
- [Generate UMAPs](../Experiments/HealthyHeart/run_models/compare_results/umap_plots)

For details on setting input and output paths for the Healthy Heart dataset, please refer to the [path setup instructions].

---

## Autism Spectrum Disorder (ASD) Dataset

**Data Source:**  
The ASD dataset can be accessed via the UCSC Cell Browser: [https://autism.cells.ucsc.edu](https://autism.cells.ucsc.edu)  
*(Speir et al., 2021; Velmeshev et al., 2019)*

### Preprocessing
- [Preprocessing Script](../Experiments/ASD/preprocessing/preprocess_ASD.py)
- [5-Fold Cross-Validation Setup](../Experiments/ASD/preprocessing/5fold_cross_val)

### Models and Scripts to Reproduce Results Sections (RS)

**RS: scMEDAL disentangles donor effects to highlight disease-associated neuronal patterns:**
- [Autoencoder (AE)](../Experiments/ASD/run_models/AE)
- [Fixed Effects Subnetwork (scMEDAL-FE)](../Experiments/ASD/run_models/scMEDAL-FE)
- [Random Effects Subnetwork (scMEDAL-RE)](../Experiments/ASD/run_models/scMEDAL-RE)

**RS: scMEDAL complementary latent spaces improve prediction accuracy at the cellular level:**
- [Mixed Effects Classifier (MEC)](../Experiments/ASD/run_models/MEC)

**RS: scMEDAL-FEC enhances cell type preservation in latent spaces:**
- [Autoencoder Classifier (AEC)](../Experiments/HealthyHeart/run_models/AEC)  
  *(Note: This link points to the Healthy Heart directory. Please ensure the correct path for ASD models.)*
- [Fixed Effects Subnetwork with Cell Type Classifier (scMEDAL-FEC)](../Experiments/HealthyHeart/run_models/scMEDAL-FEC)  
  *(Note: Similarly, ensure the correct ASD directory is used.)*

### Comparison Scripts
- [Clustering Scores](../Experiments/ASD/run_models/compare_results/clustering_scores)
- [Generate Genomaps](../Experiments/ASD/run_models/compare_results/genomaps)
- [Generate UMAPs](../Experiments/ASD/run_models/compare_results/umap_plots)

---

## Acute Myeloid Leukemia (AML) Dataset

**Data Source:**  
The AML dataset is available at the Gene Expression Omnibus (GEO) under accession number [GSE116256](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256) (van Galen et al., 2019).

### Preprocessing
- [Preprocessing Script](../Experiments/AML/preprocessing/preprocess_AML.py)
- [5-Fold Cross-Validation Setup](../Experiments/AML/preprocessing/5fold_cross_val)

### Models and Scripts to Reproduce Results Sections (RS)

**RS: scMEDAL navigates the trade-off between batch correction and cell type preservation:**
- [Autoencoder (AE)](../Experiments/AML/run_models/AE)
- [Fixed Effects Subnetwork (scMEDAL-FE)](../Experiments/AML/run_models/scMEDAL-FE)
- [Random Effects Subnetwork (scMEDAL-RE)](../Experiments/AML/run_models/scMEDAL-RE)

**RS: scMEDAL complementary latent spaces improve prediction accuracy at the cellular level:**
- [Mixed Effects Classifier (MEC)](../Experiments/AML/run_models/MEC)
  - [Cell Type Target](../Experiments/AML/run_models/MEC/celltype_target)
  - [Patient Group Target](../Experiments/AML/run_models/MEC/dx_target)

**RS: scMEDAL-FEC enhances cell type preservation in latent spaces:**
- [Autoencoder Classifier (AEC)](../Experiments/AML/run_models/AEC)
- [Fixed Effects Subnetwork with Cell Type Classifier (scMEDAL-FEC)](../Experiments/AML/run_models/scMEDAL-FEC)

### Comparison Scripts
- [Clustering Scores](../Experiments/AML/run_models/compare_results/clustering_scores)
- [Generate Genomaps](../Experiments/AML/run_models/compare_results/genomaps)
- [Generate UMAPs](../Experiments/AML/run_models/compare_results/umap_plots)

---

## Note on Variability

Due to the inherent variability in TensorFlow, results may differ slightly each time you train the models. To account for this, we have computed 95% confidence intervals to provide an estimate of the variability in model performance.

---

## References

- Litvinukova, M. et al. *Cells of the adult human heart.* Nature 588, 466?472 (2020).
- van Galen, P. et al. *Single-Cell RNA-Seq Reveals AML Hierarchies Relevant to Disease Progression and Immunity.* Cell 176, 1265?1281.e24 (2019).
- Velmeshev, D. et al. *Single-cell genomics identifies cell type-specific molecular changes in autism.* Science 364, 685?689 (2019).
- Speir, M. L. et al. *UCSC Cell Browser: visualize your single-cell data.* Bioinformatics 37, 4578?4580 (2021).
- Yu, X., Xu, X., Zhang, J., & Li, X. *Batch alignment of single-cell transcriptomics data using deep metric learning.* Nat Commun 14, 960 (2023).  
- Yu, X., Xu, X., Zhang, J., & Li, X. *Batch alignment of single-cell transcriptomics data using deep metric learning.* figshare [https://doi.org/10.6084/m9.figshare.20499630.v2](https://doi.org/10.6084/m9.figshare.20499630.v2) (2023).
