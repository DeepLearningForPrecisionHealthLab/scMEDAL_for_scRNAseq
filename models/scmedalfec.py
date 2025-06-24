from .base import Model
from .scMEDAL.scMEDAL import DomainAdversarialAE as scMEDALFEC_alg

class scMEDALFEC(Model):
    def __init__(self, **kwargs):
        super().__init__(model_name="scmedalfec", **kwargs)
        self.alg = scMEDALFEC_alg
