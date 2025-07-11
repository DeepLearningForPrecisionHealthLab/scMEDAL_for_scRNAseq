from .base import Model
from .scMEDAL.scMEDAL import AE as ae_alg
class AE(Model):
    def __init__(self, **kwargs):
        super().__init__(model_name="ae", **kwargs)
        self.alg = ae_alg
