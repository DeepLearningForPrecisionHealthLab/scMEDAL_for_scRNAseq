For instructions on setting up experiments, see **[How2SetupYourExpt](./docs/How2SetupYourExpt.md)**.

### **Model Configuration**
Each model directory contains a `model_config.py` file that specifies settings and paths. For example:  
- [Healthy Heart AE Model Configuration](./Experiments/HealthyHeart/run_models/AE/model_config.py)


**Note:** You can update the number of epochs you want to run by modifying the `epochs` parameter in the dictionary:

```python
train_model_dict = {
    "epochs": 2,        # For testing; for full experiments, use a larger value (e.g., 500)
    # "epochs": 500,     # Number of training epochs used in our experiments
}
```
---


For detailed instructions, see **[How2RunYourExpt](./docs/How2RunYourExpt.md)**.

### **Important Notes**
- Always activate the correct Conda environment before running scripts.




### **Steps to Run Models**
1. **Run All Folds Locally:**
   ```bash
   python run_modelname_allfolds.py
   ```

2. **Submit Jobs via Slurm:**
   ```bash
   sbatch sbatch_run_modelname.sh
   ```


---



## **9. Analyzing Your Model Outputs**

For guidance on analyzing and interpreting model outputs, see [How2AnalyzeYourModelOutputs](./docs/How2AnalyzeYourModelOutputs.md).




# Setup the Data

Follow the instructions to download the **AML** dataset and set up your data as described in **[Datasets](../README.md#datasets)**. That section explains how to download the raw data, preprocess it, and create train/validation/test splits.

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
* In the paper?s experiments, we **use `quick=False`**.

---

# Experiment Reproducibility Guide

This guide maps each section of the paper to the corresponding **code**, **datasets**, and **model outputs**, so you can reproduce the results as described.

## Experiments Overview

We conducted experiments on three datasets: **Healthy Heart (HH)**, **Autism Spectrum Disorder (ASD)**, and **Acute Myeloid Leukemia (AML)**. For each dataset, we provide:

* **Preprocessing and 5-fold cross-validation scripts** to split the data into train/val/test.
  Alternatively, we provide the **precomputed data splits** via Figshare. See **[Datasets](../README.md#datasets)**.
* **Training scripts** for our API models (e.g., Autoencoder, **scMEDAL** variants, Mixed Effects Classifier) and **comparable models**: **scVI**, **scANVI**, **Harmony**, **Scanorama**, and **SAUCIE**.
* **[Analysis notebooks](#analysis-notebooks)** for post-hoc evaluation (e.g., clustering scores, Genomaps, UMAP plots). These notebooks guide you through generating the paper?s results once latent spaces and reconstruction outputs are available.

We also provide on Figshare:

* **Latent space outputs** and a **sample count matrix** with 300-cell projections so you can reproduce the **Genomaps** exactly as in the paper.

You have two options:

1. **Reproduce analyses from provided model outputs**
   Download our latent spaces and the 300-cell sample matrix, place them in the specified folders (see below), and run the analysis notebooks to regenerate paper figures.

2. **Run models from scratch**
   Train the models yourself and then run the analysis notebooks.

To run all models from our API programmatically (**AE**, **AEC**, **scMEDAL-FE**, **scMEDAL-RE**, **scMEDAL-FEC**), use **[`scripts/1-run_scMEDAL_alldatasets.py`](../scripts/1-run_scMEDAL_alldatasets.py)**:

```bash
python 1-run_scMEDAL_alldatasets.py
```

Make sure your **scMEDAL** environment is activated. See **[Installation](../README.md#installation)**.

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
* **Requirements:** See [`comparables/SAUCIE/requirements.txt`](../comparables/SAUCIE/requirements.txt)
  **SAUCIE.yaml:** [`comparables/comparables_env/SAUCIE.yaml`](../comparables/comparables_env/SAUCIE.yaml)

#### scVI

* Example Script (AML): [`comparables/AML/run_scVI.py`](../comparables/AML/run_scVI.py)
  
  Run:

  ```bash
  python run_scVI.py
  ```
* **Docs:** [https://scvi-tools.org/](https://scvi-tools.org/)
  **[comparables_env.yaml (shared comparables)]**(../comparables/comparables_env/comparables_env.yaml)

#### scanorama

* Example Script (AML): [`comparables/AML/run_scanorama.py`](../comparables/AML/run_scanorama.py)
  
  Run:

  ```bash
  python run_scanorama.py
  ```
* **Preprocessing reference:** [https://scanpy.readthedocs.io/en/stable/external/preprocessing.html](https://scanpy.readthedocs.io/en/stable/external/preprocessing.html)
   **[comparables_env.yaml:]**(../comparables/comparables_env/comparables_env.yaml)

#### harmony

* Example Script (AML): [`comparables/run_harmony.py`](../comparables/AML/run_harmony.py)
  
  Run:

  ```bash
  python run_harmony.py
  ```
* **Preprocessing reference:** [https://scanpy.readthedocs.io/en/stable/external/preprocessing.html](https://scanpy.readthedocs.io/en/stable/external/preprocessing.html)
* **[comparables_env.yaml:]**(../comparables/comparables_env/comparables_env.yaml)

#### scANVI

* Example Script (AML) : [`comparables/run_scANVI.py`](../comparables/AML/run_scANVI.py)
  
  Run:

  ```bash
  python run_scANVI.py
  ```
* **Docs:** [https://scvi-tools.org/](https://scvi-tools.org/)
  **[comparables_env.yaml:]**(../comparables/comparables_env/comparables_env.yaml)

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

* `2-mec_asd_dx.py`
* `2-mec_asd_celltype.py`

**AML**

* `2-mec_aml_celltype.py`
* `2-mec_aml_patientgroup.py`

**Healthy Heart**

* `2-mec_hh_tissue.py`
* `2-mec_hh_celltype.py`

---

## Analysis Notebooks

Use the following notebooks to reproduce the post-hoc analyses and figures:

* `3-analysis_aml.ipynb`
* `3-analysis_asd.ipynb`
* `3-analysis_hh.ipynb`

> Download/setup the required model outputs (or train from scratch), update paths where necessary, and run these notebooks to regenerate the paper?s results.

---

**Note:** Due to stochasticity (e.g., TensorFlow backends), outputs may differ slightly across runs. To account for this, we report **95% confidence intervals (CI)** as an estimate of variability.
