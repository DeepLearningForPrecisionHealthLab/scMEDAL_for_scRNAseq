from .base import Model
from .scMEDAL.scMEDAL import AE as ae_alg
class AE(Model):
    def __init__(self, model_name:str="ae", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.alg = ae_alg
