
#Utils to load model weights. 
class Namespace:
    """
    A simple namespace wrapper around a dictionary. Allows dictionary keys
    to be accessed as attributes for convenience.

    Parameters:
    -----------
    adict : dict
        Dictionary to be converted to a namespace.

    Example:
    --------
    >>> d = {'key': 'value'}
    >>> ns = Namespace(d)
    >>> print(ns.key)
    'value'
    """
    def __init__(self, adict):
        self.__dict__.update(adict)


def get_last_checkpoints_path(checkpoints_path):
    """
    Fetches the last checkpoint's path from a given checkpoints directory.

    Parameters:
    -----------
    checkpoints_path : str
        Path to the directory containing checkpoint files.

    Returns:
    --------
    full_checkpoint_path : str
        Full path to the last saved model checkpoint.

    Example:
    --------
    >>> path = "/path/to/checkpoints"
    >>> print(get_last_checkpoints_path(path))
    "/path/to/checkpoints/cp-0005.ckpt"
    """
    # Read the checkpoint file
    with open(checkpoints_path + "/checkpoint", 'r') as file:
        lines = file.readlines()

    # Parse the checkpoint file to get the model_checkpoint_path
    for line in lines:
        if "model_checkpoint_path" in line:
            model_checkpoint_path = line.split(': ')[1].strip('"\n')

    # Full path
    full_checkpoint_path = checkpoints_path + "/" + model_checkpoint_path
    return full_checkpoint_path

def get_latest_checkpoint(checkpoint_dir):
    import re
    from compare_results_utils import glob_like
    """
    Retrieves the latest checkpoint file from the specified directory.

    Args:
        checkpoint_dir (str): The directory where checkpoint files are stored.

    Returns:
        str: The filename of the latest checkpoint file if available, otherwise None.
    
    Notes:
        The function searches for checkpoint files matching the pattern 'cp-*.ckpt.data-00000-of-00001'
        and returns the one with the highest numeric value in its name.
    """
    # Pattern to match the checkpoint files
    checkpoint_pattern = 'cp-*.ckpt.data-00000-of-00001'
    
    # List all matching checkpoint files using glob_like
    checkpoints = glob_like(checkpoint_dir, checkpoint_pattern)
    
    # Extract the numbers from the filenames and sort the checkpoints
    checkpoints.sort(key=lambda x: int(re.findall(r'cp-(\d+)', x)[0]))
    
    # Get the latest checkpoint
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        return latest_checkpoint
    else:
        return None


def read_model_params(checkpoints_path):
    """
    Reads the model parameters from a checkpoint directory.

    Parameters:
    -----------
    checkpoints_path : str
        Path to the directory containing checkpoint files and model parameters.

    Returns:
    --------
    model_params : Namespace
        Namespace object containing all the model parameters.

    Example:
    --------
    >>> path = "/path/to/checkpoints"
    >>> params = read_model_params(path)
    >>> print(params.layer_size)
    128
    """

    import yaml
    import glob
    model_params_path = glob.glob(checkpoints_path+"/model_params*")[0]   

    # Open and read the YAML file
    with open(model_params_path, 'r') as file:
        model_params_dict = yaml.safe_load(file)

    model_params = Namespace(model_params_dict)
    return model_params

def parse_value(value):
    import ast
    try:
        # Attempt to parse the value as a Python literal
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If it fails, return the value as a str
        return str(value)

def read_model_params_frompath(model_params_path,replacements):
    """
    Reads the model parameters from a checkpoint directory.

    Parameters:
    -----------
    model_params_path : str
        Path to the yaml file containing checkpoint files and model parameters.

    Returns:
    --------
    model_params : Namespace/dict
        Namespace object/dict containing all the model parameters.

    Example:
    --------
    >>> path = "/path/to/checkpoints"
    >>> params = read_model_params(path)
    >>> print(params.layer_size)
    128
    """

    import yaml
    import glob
    # model_params_path = glob.glob(checkpoints_path+"/model_params*")[0]   

    # Open and read the YAML file
    with open(model_params_path, 'r') as file:
        model_params_dict = yaml.safe_load(file)

    try:
        model_params = Namespace(model_params_dict)
        return model_params
    except:
        print("Returning dict, fix tensorflow inputs")

        model_params_dict = {k: parse_value(v) for item in model_params_dict for k, v in [item.split(':', 1)]}
        
        for key, value in model_params_dict.items():
            if isinstance(value, str) and key in replacements:
                print(key,"replaced by",value)
                model_params_dict[key] = replacements[key]
        return model_params_dict