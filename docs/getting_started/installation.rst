:topic: Installation

.. toctree::
   :caption: List of tutorials
   :maxdepth: 2
   :titlesonly:

Installation
############

With pip
********

The materials-toolkits python packages can be install as follows:

.. code-block:: bash

    pip install materials-toolkits

Virtual environement (optional)
*******************************

Create a virtual environment and activate it:

.. code-block:: bash

    python -m venv ./venv_path
    source ./venv_path/bin/activate
    pip install materials-toolkits

Torch, PyG and CUDA (optional)
******************************

.. warning::

    The command `pip install materials-toolkit` will install the default version of pytorch and torch-geometric. To install a specific version of these packages to match your cuda version you should install them manually before installing Materials toolkit.

Installation of Torch and PyG with a specific cuda version (can be cpu):

.. code-block:: bash

    export CUDA_VERSION=cu117 # the valid options are cu117, cu118 and cpu
    pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/$CUDA_VERSION
    pip install torch_geometric==2.3.1
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+$CUDA_VERSION.html


Lastest version
***************

To install the lastest version:

.. code-block:: bash

    pip install git+https://github.com/materials-toolkits/pymaterials.git

