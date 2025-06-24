from typing import NamedTuple, List, Union, Optional, Dict
from .experiment_design import ExperimentDesignConfigs

class PlotConfigs(NamedTuple):
    #shape_col:str="celltype"
    #color_col:str="donor"
    markers:List[Union[str, int]]= ["x", "+", "<", "h", "s", ".", 'o', 's', '^', '*', '1', '8', 'p', 'P', 'D', '|',0, ',', 'd', 2],
    showplot:bool=False
    save_fig:bool=True
    output:Optional[str]=None
    #
    use_bio_bio:bool=True
    use_batch_batch:bool=True
    use_bio_batch:bool=False
    use_batch_bio:bool=False
    
    def get_shape_color_dict(self, expconfigs:ExperimentDesignConfigs) -> Dict[str, Dict[str,str]]:
        shape_color_dict = {}
        if self.use_bio_bio:
            shape_color_dict[f"{expconfigs.bio_col}-{expconfigs.bio_col}"] = {"shape_col": expconfigs.bio_col, "color_col": expconfigs.bio_col}
        if self.use_batch_batch:
            shape_color_dict[f"{expconfigs.batch_col}-{expconfigs.batch_col}"] = {"shape_col": expconfigs.batch_col, "color_col": expconfigs.batch_col}
        if self.use_bio_batch:
            shape_color_dict[f"{expconfigs.bio_col}-{expconfigs.batch_col}"] = {"shape_col": expconfigs.bio_col, "color_col": expconfigs.batch_col}
        if self.use_batch_bio:
            shape_color_dict[f"{expconfigs.batch_col}-{expconfigs.bio_col}"] = {"shape_col": expconfigs.batch_col, "color_col": expconfigs.bio_col}
        return shape_color_dict
