from typing import NamedTuple, List

class DataConfigs(NamedTuple):
    eval_test:bool=True
    use_z:bool=False
    get_pred:bool=False
    scaling:str="min_max"
    fold_list:List[int]=list(range(1,6))
    