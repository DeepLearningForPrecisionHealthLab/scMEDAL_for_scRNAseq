from .base import Model
from .scMEDAL.scMEDAL import AEC as aec_alg

class AEC(Model):
    def __init__(self, model_name:str="aec", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.alg = aec_alg
