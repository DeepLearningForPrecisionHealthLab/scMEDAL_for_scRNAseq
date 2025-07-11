from typing import NamedTuple, Optional

class ExperimentDesignConfigs(NamedTuple):
    batch_col:str="batch"
    bio_col:str="celltype"
<<<<<<< HEAD
    donor_col:Optional[str]='DonorID'
=======
    donor_col:Optional[str]='DonorID'
    # when you want to make predictions with MEC of another column, you do 
    # bio_col:str="diagnosis" for  ASD
    # bio_col:str="TissueDetail" for HH
    # bio_col:str="Patient_group" for AML
>>>>>>> bc7d766fb90c6d45c716908e51471d864b7ebff1
