Materials toolkit
=================

<img src="https://materials-toolkits.github.io/pymaterials/_static/images/coverage.svg">[![Unit testing](https://github.com/materials-toolkits/pymaterials/actions/workflows/tests.yml/badge.svg)](https://github.com/materials-toolkits/pymaterials/actions/workflows/tests.yml)

Materials-toolkit is a python package created to facilitate the development of machine learning models in materials science.

Doc: https://materials-toolkits.github.io/pymaterials/

# Dependancies

* torch==2.0.1
* torch_geometric==2.3.1
* pymatgen
* spgrep

# Installation

## Virtual environement (optional)

Create a virtual environment and activate it:
```bash
python -m venv ./venv_path
source ./venv_path/bin/activate
```

## Torch, PyG and CUDA (optional)

> Warning, the command `pip install materials-toolkit` will install the default version of pytorch and torch-geometric. To install a specific version of these packages to match your cuda version you should install them manually before installing Materials toolkit.

Installation of Torch and PyG with a specific cuda version (can be cpu):
```bash
export CUDA_VERSION=cu117 # the valid options are cu117, cu118 and cpu
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/$CUDA_VERSION
pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+$CUDA_VERSION.html
```

## Materials toolkit

Install stable version:
```bash
pip install materials-toolkit
```

Install lastest version:
```bash
pip install git+https://github.com/materials-toolkits/pymaterials.git
```
