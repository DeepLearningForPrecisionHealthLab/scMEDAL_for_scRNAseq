import sys
sys.path.append("../")
from models.models import train_model_on_named_experiment


scmedalfe_hh = train_model_on_named_experiment("scMEDAL-FE", "HH", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})
scmedalre_hh = train_model_on_named_experiment("scMEDAL-RE", "HH", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})

# scmedalfe_asd = train_model_on_named_experiment("scMEDAL-FE", "ASD", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})
# scmedalre_asd = train_model_on_named_experiment("scMEDAL-RE", "ASD", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})

# scmedalfe_aml= train_model_on_named_experiment("scMEDAL-FE", "AML", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})
# scmedalre_aml = train_model_on_named_experiment("scMEDAL-RE", "AML", model_kwargs={"n_latent_dims":50}, train_kwargs={"quick":True})