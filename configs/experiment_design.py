from typing import NamedTuple

class ExperimentDesignConfigs(NamedTuple):
    batch_col:str="batch"
    bio_col:str="celltype"