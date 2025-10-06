import yaml

# Initialize the environment dictionary
environment = {
    'name': 'scMEDAL',
    'channels': ['conda-forge', 'defaults'],
    'dependencies': []
}

# Function to add a package to dependencies
def add_dependency(package_name, version, pip_package=False):
    if pip_package:
        # Check if 'pip' is already in dependencies
        pip_index = next((index for (index, d) in enumerate(environment['dependencies']) if isinstance(d, dict) and 'pip' in d), None)
        if pip_index is None:
            # Add a new pip section
            environment['dependencies'].append({'pip': [f"{package_name}=={version}"]})
        else:
            # Append to existing pip section
            environment['dependencies'][pip_index]['pip'].append(f"{package_name}=={version}")
    else:
        environment['dependencies'].append(f"{package_name}={version}")

# Add Python version
import sys
python_version = sys.version.split()[0]
environment['dependencies'].append(f"python={python_version}")

# Add packages
try:
    import numpy as np
    add_dependency('numpy', np.__version__)
except ImportError:
    pass

try:
    import pandas as pd
    add_dependency('pandas', pd.__version__)
except ImportError:
    pass



try:
    import scanpy as sc
    add_dependency('scanpy', sc.__version__)
except ImportError:
    pass

try:
    import anndata
    add_dependency('anndata', anndata.__version__)
except ImportError:
    pass

try:
    import scipy
    add_dependency('scipy', scipy.__version__)
except ImportError:
    pass

try:
    import sklearn
    add_dependency('scikit-learn', sklearn.__version__)
except ImportError:
    pass

try:
    import matplotlib
    add_dependency('matplotlib', matplotlib.__version__)
except ImportError:
    pass

try:
    import seaborn as sns
    add_dependency('seaborn', sns.__version__)
except ImportError:
    pass

try:
    import h5py
    add_dependency('h5py', h5py.__version__)
except ImportError:
    pass

try:
    import yaml as pyyaml
    add_dependency('pyyaml', pyyaml.__version__)
except ImportError:
    pass


try:
    import tensorflow as tf
    add_dependency('tensorflow', tf.__version__)
except ImportError:
    pass

try:
    import tensorflow_probability as tfp
    add_dependency('tensorflow-probability', tfp.__version__)
except ImportError:
    pass

try:
    import tensorflow_addons as tfa
    # Assuming tensorflow-addons is installed via pip
    add_dependency('tensorflow-addons', tfa.__version__, pip_package=True)
except ImportError:
    pass

try:
    import tensorflow as tf
    from tf.keras import __version__ as keras_version
    add_dependency('keras', keras_version)
except ImportError:
    pass

try:
    import scanpy as sc
    add_dependency('scanpy', sc.__version__)
except ImportError:
    pass

try:
    import statsmodels.api as sm
    import statsmodels
    add_dependency('statsmodels', statsmodels.__version__)
except ImportError:
    pass



try:
    import genomap
    #This is a manual check because genomap has no genomap.__version__
    add_dependency('genomap', '1.3.6')
except ImportError:
    pass

# Write the environment to a YAML file
with open('scMEDAL.yaml', 'w') as file:
    
    yaml.dump(environment, file)