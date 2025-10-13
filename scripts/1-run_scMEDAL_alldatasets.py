import sys, os
# Set up your project path here
#os.chdir("/archive/bioinformatics/DLLab/AixaAndrade/src/gitfront/dev2/scMEDAL_for_scRNAseq")
sys.path.insert(0, os.getcwd())          # <-- add this right after the chdir

# The working dir should be scMEDAL_for_scRNAseq dir
print("working dir:",os.getcwd())
from models.models import train_model_on_named_experiment

# Run models quick test (see demo)
# scmedalfe_hh = train_model_on_named_experiment("scMEDAL-FE", "HH", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})
# scmedalre_hh = train_model_on_named_experiment("scMEDAL-RE", "HH", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})

# scmedalfe_asd = train_model_on_named_experiment("scMEDAL-FE", "ASD", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})
# scmedalre_asd = train_model_on_named_experiment("scMEDAL-RE", "ASD", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})

# scmedalfe_aml= train_model_on_named_experiment("scMEDAL-FE", "AML", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})
# scmedalre_aml = train_model_on_named_experiment("scMEDAL-RE", "AML", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})

# The following commands run each of the implemented models in the "HH",ASD and AML datasets
scmedalfe_hh = train_model_on_named_experiment("scMEDAL-FE", "HH", model_kwargs={"n_latent_dims":50})
scmedalre_hh = train_model_on_named_experiment("scMEDAL-RE", "HH", model_kwargs={"n_latent_dims":50})
ae_hh = train_model_on_named_experiment("AE", "HH", model_kwargs={"n_latent_dims":50})
aec_hh = train_model_on_named_experiment("AEC", "HH", model_kwargs={"n_latent_dims":50})
scmedalfec_hh = train_model_on_named_experiment("scMEDAL-FEC", "HH", model_kwargs={"n_latent_dims":50})

scmedalfe_asd = train_model_on_named_experiment("scMEDAL-FE", "ASD", model_kwargs={"n_latent_dims":50})
scmedalre_asd = train_model_on_named_experiment("scMEDAL-RE", "ASD", model_kwargs={"n_latent_dims":50})
ae_asd = train_model_on_named_experiment("AE", "ASD", model_kwargs={"n_latent_dims":50})
aec_asd = train_model_on_named_experiment("AEC", "ASD", model_kwargs={"n_latent_dims":50})
scmedalfec_asd = train_model_on_named_experiment("scMEDAL-FEC", "ASD", model_kwargs={"n_latent_dims":50})


scmedalfe_aml= train_model_on_named_experiment("scMEDAL-FE", "AML", model_kwargs={"n_latent_dims":50})
scmedalre_aml = train_model_on_named_experiment("scMEDAL-RE", "AML", model_kwargs={"n_latent_dims":50})
ae_aml = train_model_on_named_experiment("AE", "AML", model_kwargs={"n_latent_dims":50})
aec_aml = train_model_on_named_experiment("AEC", "AML", model_kwargs={"n_latent_dims":50})
scmedalfec_aml = train_model_on_named_experiment("scMEDAL-FEC", "AML", model_kwargs={"n_latent_dims":50})