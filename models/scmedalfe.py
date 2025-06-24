from .base import Model
from .scMEDAL.scMEDAL import DomainAdversarialAE as scMEDALFE_alg

class scMEDALFE(Model):
    def __init__(self, **kwargs):
        super().__init__(model_name="scmedalfe", **kwargs)
        self.alg = scMEDALFE_alg
