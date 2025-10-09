# Setup the Data

Follow the instructions to download the **AML** dataset and set up your data as described in the **[Datasets](../README.md#datasets)** section. That section explains how to download the raw data, preprocess it, and create train/validation/test splits.

---

# Running the Demos (10 epochs)

In our **paper experiments**, models are trained for **up to 500 epochs** with **early stopping** (training may halt earlier).
By contrast, the **demo notebooks** use **10 epochs** to provide a quick, end-to-end tutorial.

Use the demos to learn how to train **scMEDAL** and make visualizations:

* **[`demo/demo_aml.ipynb`](../demo/demo_aml.ipynb)**
* **[`demo/demo_hh.ipynb`](../demo/demo_hh.ipynb)**
* **[`demo/demo_asd.ipynb`](../demo/demo_asd.ipynb)**

> Runtime varies by dataset. **AML** is the fastest because it has fewer cells.

---

# Running the AML Dataset for 500 Epochs with Early Stopping (5 folds)

* Set `quick=False` in `train_kwargs` to run the **full training** (**5-fold × 500 epochs**) with early stopping.
* In the paper experiments, we **use `quick=False`**.

---

# Experiment Reproducibility Guide

This guide maps each section of the paper to the corresponding **code**, **datasets**, and **model outputs**, so you can reproduce the results as described.

## Experiments Overview

We conducted experiments on three datasets: **Healthy Heart (HH)**, **Autism Spectrum Disorder (ASD)**, and **Acute Myeloid Leukemia (AML)**. For each dataset, we provide:

* **Preprocessing and 5-fold cross-validation scripts** to split the data into train/val/test.
  Alternatively, we provide the **precomputed data splits** via Figshare. See **[Datasets](../README.md#datasets)**.
* **Training scripts** for our API models (e.g., Autoencoder, **scMEDAL** variants, Mixed Effects Classifier) and **comparable models**: **scVI**, **scANVI**, **Harmony**, **Scanorama**, and **SAUCIE**.
* [**Analysis notebooks**](#analysis-notebooks) for post-hoc evaluation (e.g., clustering scores, Genomaps, UMAP plots). These notebooks guide you through generating the paper results once latent spaces and reconstruction outputs are available.

We also provide in the Figshare:

* **Latent space outputs**  and a **sample count matrix**  with 300-cell projections so you can reproduce the **Genomaps** exactly as in the paper.

You have two options:

1. **Reproduce analyses from provided model outputs**
   Download our latent spaces and the 300-cell sample matrix, place them in the specified folders (see below), and run the analysis notebooks to regenerate paper figures.

2. **Run models from scratch**
   Train the models yourself and then run the analysis notebooks.

To run all models from our API (AE,AEC,scMEDAL-FE,scMEDAL-RE and scMEDAL-FEC), use [1-run_scMEDAL_alldatasets.py](../scripts/1-run_scMEDAL_alldatasets.py) and make sure your **scMEDAL** environment is activated. See **[scMEDAL Installation](../README.md#Installation)**.

```bash
python 1-run_scMEDAL_alldatasets.py
```


---
## Comparable Models (per-dataset configs)

Each comparable model has its own script and environment requirements. Configurations vary by dataset. Below we list the **AML** entry points; for **HH** and **ASD**, use the corresponding dataset paths/scripts.

> All comparable-model scripts produce the **same output file structure** as our API models. See **[Outputs & analysis folders](../README.md#outputs-and-analysis-folders)**.
#### SAUCIE

* Example Script (AML): [`comparables/AML/run_SAUCIE.py`](../comparables/AML/run_SAUCIE.py)
  
  Run:

  ```bash
  python run_SAUCIE.py
  ```
- **Requirements:** See [`comparables/SAUCIE/requirements.txt`](../comparables/SAUCIE/requirements.txt)
- Use [`SAUCIE.yaml`](../comparables/comparables_env/SAUCIE.yaml) tp create conda environment.

#### scVI

* Example Script (AML): [`comparables/AML/run_scVI.py`](../comparables/AML/run_scVI.py)
  
  Run:

  ```bash
  python run_scVI.py
  ```
- **Docs:** [https://scvi-tools.org/](https://scvi-tools.org/)


#### scanorama

* Example Script (AML): [`comparables/AML/run_scanorama.py`](../comparables/AML/run_scanorama.py)
  
  Run:

  ```bash
  python run_scanorama.py
  ```
- **Docs:** [https://scanpy.readthedocs.io/en/stable/external/preprocessing.html](https://scanpy.readthedocs.io/en/stable/external/preprocessing.html)


#### harmony

* Example Script (AML): [`comparables/run_harmony.py`](../comparables/AML/run_harmony.py)
  
  Run:

  ```bash
  python run_harmony.py
  ```
- **Docs:** [https://scanpy.readthedocs.io/en/stable/external/preprocessing.html](https://scanpy.readthedocs.io/en/stable/external/preprocessing.html)


#### scANVI

* Example Script (AML) : [`comparables/run_scANVI.py`](../comparables/AML/run_scANVI.py)
  
  Run:

  ```bash
  python run_scANVI.py
  ```
- **Docs:** [https://scvi-tools.org/](https://scvi-tools.org/)

Note:  Use **Comparables env for scVI, scANVI, harmony, scanorama and SAUCIE, use [comparables_env.yaml](../comparables/comparables_env/comparables_env.yaml) to create conda environment.
---

## Using Provided Model Outputs (example for AML Genomaps)

1. **Place the AML latent space** under:

   ```
   /outputs/AML/latent_space/log_transformed_2916hvggenes
   ```
2. **Place the 300-cell sample matrix** used for Genomaps under:

   ```
   outputs/AML/compare_models/log_transformed_2916hvggenes/AML_default/genomap/CMmultibatch_300_cells_per_batch_19batches_Mono_Mono-like_with_2fe_input
   ```

> After these paths are set, run the analysis notebooks to reproduce the Genomaps section for AML.


---

## Using provided model outputs (example for AML Genomaps)

1. **Place the AML latent space** under:

```
/outputs/AML/latent_space/log_transformed_2916hvggenes
```

2. **Place the 300-cell sample matrix** used for Genomaps under:

```
outputs/AML/compare_models/log_transformed_2916hvggenes/AML_default/genomap/CMmultibatch_300_cells_per_batch_19batches_Mono_Mono-like_with_2fe_input
```

> After these paths are set, run the analysis notebooks to reproduce the Genomaps section for AML.

---

## Running Models from Scratch

If you train models yourself, you may need to update paths to your run folders. For example:

```python
model_folder_dict = {
    # "ae": "",
    # "aec": "",
    "scmedalfe": "run_crossval_loss_gen_weight-1_loss_recon_weight-1000_loss_class_weight-1_n_latent_dims-50_layer_units-512-132_scaling-min_max_model_type-scmedalfe_batch_size-512_epochs-500_patience-30_sample_size-10000_2025-10-03_14-27",
    # "scmedalfec": "",
    "scmedalre": "run_crossval_loss_recon_weight-110_loss_latent_cluster_weight-0.1_n_latent_dims-50_layer_units-512-132_scaling-min_max_batch_size-512_epochs-500_patience-30_sample_size-10000_2025-10-03_14-27",
}
```

Adjust these entries to match your own output directories.

---

## MEC Classifier Scripts

After running the representation models, you can train the **Mixed Effects Classifier (MEC)**:

**ASD**

* [2-mec_asd_dx.py](../scripts/2-mec_asd_dx.py)
* [2-mec_asd_celltype.py](../scripts/2-mec_asd_celltype.py)

**AML**

* [2-mec_aml_celltype.py](../scripts/2-mec_aml_celltype.py)
* [2-mec_aml_patientgroup.py](../scripts/2-mec_aml_patientgroup.py)

**Healthy Heart**

* [2-mec_hh_tissue.py](../scripts/2-mec_hh_tissue.py)
* [2-mec_hh_celltype.py](../scripts/2-mec_hh_celltype.py)

---

## Analysis Notebooks

Use the following notebooks to reproduce the post-hoc analyses and figures:

* [3-analysis_aml.ipynb](../scripts/3-analysis_aml.ipynb)
* [3-analysis_asd.ipynb`](../scripts/3-analysis_asd.ipynb)
* [3-analysis_hh.ipynb](../scripts/3-analysis_hh.ipynb)

> Download/setup the required model outputs (or train from scratch), update paths where necessary, and run these notebooks to regenerate the paper results.



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
The AML dataset is available at the Gene Expression Omnibus (GEO) under accession number [GSE116256](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi-acc=GSE116256) (van Galen et al., 2019).


**Data splits** are available in [AML_data.zip](../Experiments/AML_data.zip). We have included the 5 cross-validation splits metadata  and the highly variable genes (HVGs) selected for this experiment.

You can either run the 5-fold cross-validation scripts or use the cell ids provided to generate splits.

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
- van Galen, P. et al. *Single-Cell RNA-Seq Reveals AML Hierarchies Relevant to Disease Progression and Immunity.* Cell 176, 1265-1281.e24 (2019).
- Velmeshev, D. et al. *Single-cell genomics identifies cell type-specific molecular changes in autism.* Science 364, 685-689 (2019).
- Speir, M. L. et al. *UCSC Cell Browser: visualize your single-cell data.* Bioinformatics 37, 4578-4580 (2021).
- Yu, X., Xu, X., Zhang, J., & Li, X. *Batch alignment of single-cell transcriptomics data using deep metric learning.* Nat Commun 14, 960 (2023).  
- Yu, X., Xu, X., Zhang, J., & Li, X. *Batch alignment of single-cell transcriptomics data using deep metric learning.* figshare [https://doi.org/10.6084/m9.figshare.20499630.v2](https://doi.org/10.6084/m9.figshare.20499630.v2) (2023).