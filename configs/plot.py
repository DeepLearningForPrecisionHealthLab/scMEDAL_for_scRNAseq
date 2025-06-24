from typing import NamedTuple, List, Union, Optional

class PlotConfigs(NamedTuple):
    shape_col:str="celltype"
    color_col:str="donor"
    markers:List[Union[str, int]]= ["x", "+", "<", "h", "s", ".", 'o', 's', '^', '*', '1', '8', 'p', 'P', 'D', '|',0, ',', 'd', 2],
    showplot:bool=False
    save_fig:bool=True
    output:Optional[str]=None