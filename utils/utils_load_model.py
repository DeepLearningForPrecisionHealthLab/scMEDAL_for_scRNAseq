
#Utils to load model weights. Docstring generated with chatgot 4
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
