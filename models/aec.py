from .base import Model
from .scMEDAL.scMEDAL import AEC as aec_alg

class AEC(Model):
    def __init__(self, **kwargs):
        super().__init__(model_name="aec", **kwargs)
        self.alg = aec_alg
