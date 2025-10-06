import yaml

# Initialize the environment dictionary
environment = {
    'name':'SAUCIE',
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
    import tensorflow as tf
    add_dependency('tensorflow-gpu', tf.__version__, pip_package=True)
except ImportError as e:
    import traceback
    print(f"ImportError while importing TensorFlow: {e}")
    traceback.print_exc()



try:
    import fcswrite
    add_dependency('fcswrite', fcswrite.__version__, pip_package=True)
except ImportError:
    pass

try:
    import fcsparser
    add_dependency('fcsparser', fcsparser.__version__, pip_package=True)
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
    import sklearn
    add_dependency('scikit-learn', sklearn.__version__)
except ImportError:
    pass

try:
    import matplotlib
    add_dependency('matplotlib', matplotlib.__version__)
except ImportError:
    pass

# Write the environment to a YAML file
with open('SAUCIE.yaml', 'w') as file:
    yaml.dump(environment, file)