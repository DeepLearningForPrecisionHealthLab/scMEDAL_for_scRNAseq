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