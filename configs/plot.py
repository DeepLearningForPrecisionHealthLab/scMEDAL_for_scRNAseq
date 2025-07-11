from typing import NamedTuple, List, Union, Optional, Dict
from .experiment_design import ExperimentDesignConfigs

class PlotConfigs(NamedTuple):
    markers:List[Union[str, int]]= ["x", "+", "<", "h", "s", ".", 'o', 's', '^', '*', '1', '8', 'p', 'P', 'D', '|',0, ',', 'd', 2]
    palette_choice:List[str] = [
        '#e6194b',  # Red
        '#3cb44b',  # Green
        '#ffe119',  # Yellow
        '#4363d8',  # Blue
        '#f58231',  # Orange
        '#911eb4',  # Purple
        '#46f0f0',  # Cyan
        '#f032e6',  # Magenta
        '#000000',  # Black
        '#fabebe',  # Light pink
        '#008080',  # Teal
        '#e6beff',  # Lavender
        '#9a6324',  # Brown
        '#d2f53c',  # Lime
        #'#ff69b4',  # Hot pink
<<<<<<< HEAD
        "#7fff00",  # Chartreuse
=======
#        '#00ff7f'  # Spring Green
        '#7fff00',  # Chartreuse
>>>>>>> bc7d766fb90c6d45c716908e51471d864b7ebff1
        '#000080',  # Navy
        '#800000',  # Maroon
        '#808000',  # Olive
        '#800080',  # Dark purple
        '#808080',  # Gray
<<<<<<< HEAD
        '#ffd700'   # Gold
=======
        '#ffd700',  # Gold
        '#ff4500',  # Orange Red
        '#00ff7f',  # Spring Green
        '#dda0dd',  # Plum
        '#87ceeb',  # Sky Blue
        '#00ced1',  # Dark Turquoise
        '#9400d3',  # Dark Violet
        '#ff6347',  # Tomato
        '#4682b4',  # Steel Blue
        '#6b8e23',  # Olive Drab
        '#a52a2a'   # Brick Red
>>>>>>> bc7d766fb90c6d45c716908e51471d864b7ebff1
    ]
    showplot:bool=False
    save_fig:bool=True
    outpath:Optional[str]=None
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
