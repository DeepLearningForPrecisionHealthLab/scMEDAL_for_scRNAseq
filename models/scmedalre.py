from .base import Model
from .scMEDAL.scMEDAL import DomainEnhancingAutoencoderClassifier as scMEDALRE_alg

class scMEDALRE(Model):
    def __init__(self, **kwargs):
        super().__init__(model_name="scmedalre", **kwargs)
        self.alg = scMEDALRE_alg
