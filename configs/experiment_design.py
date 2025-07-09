from typing import NamedTuple, Optional

class ExperimentDesignConfigs(NamedTuple):
    batch_col:str="batch"
    bio_col:str="celltype"
    donor_col:Optional[str]='DonorID'