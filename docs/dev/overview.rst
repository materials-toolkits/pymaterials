Poetry
======

Materials toolkit uses the `poetry <https://python-poetry.org/>`_ package to manage dependencies, the development environment and the `PyPI <https://pypi.org/>`_ repository. During setup, poetry automatically creates a virtual environment that can be used during development.

Setup
-----

Poetry can be installed using the `pip` command as follows:

.. code-block:: bash

   pip install poetry

There are a few additional dependencies needed for unit testing and generating documentation. They are configured under the `docs` group and the `test` group in `pyproject.toml` and can be installed automatically with poetry. The development environment can be built by cloning the materials-toolkits repository and installing dependencies on a virtual environment using poetry.

.. code-block:: bash

   git clone git@github.com:materials-toolkits/pymaterials.git
   cd pymaterials
   poetry install --with docs,test

Development environment
-----------------------

A shell with the appropriate virtual environment can then be opened with the following command:

.. code-block:: bash

   poetry shell
