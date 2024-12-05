import yaml

# Initialize the environment dictionary
environment = {
    'name': 'Aixa_scDML',
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

# Write the environment to a YAML file
with open('preprocess_and_plot_umaps_env.yaml', 'w') as file:
    
    yaml.dump(environment, file)