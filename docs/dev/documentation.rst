Documentation
=============

The Materials toolkit documentation is generated using Sphinx. This documentation uses the Read the Docs theme for Sphinx and the `sphinx.ext.doctest` extension to perform unit tests on code snippets.

Installation of the dependencies
--------------------------------

Sphinx related dependencies can be installed with poetry. They are configured in the `docs` group of `pyproject.toml`. Dependancies can be installed from a cloned repository as follows:

.. code-block:: bash

   poetry install --with docs,test

Build documentation
-------------------

A static html build of the documentation can be generated using the `sphinx-build` command as follows:

.. code-block:: bash

   poetry run sphinx-build -M html docs docs/_build

The `docs/_build` directory contains the resulting static build.

Unit testing on code snippets 
-----------------------------

Unit test can be performed on code snippets using the `sphinx.ext.doctest` extension. 

.. code-block:: RST

   .. testcode::

      import materials_toolkit.data as data

      structure = data.StructureData()
      print(structure)

   .. testoutput::
      :hide:

      StructureData(periodic=[1], num_atoms=[1], batch_atoms=[0], num_edges=[1], batch_edges=[0])

This code is rendered as foolow in the documentation:

.. testcode::

   import materials_toolkit.data as data

   structure = data.StructureData()
   print(structure)

.. testoutput::
   :hide:

   StructureData(periodic=[1], num_atoms=[1], batch_atoms=[0], num_edges=[1], batch_edges=[0])

To go further into unit testing, please see :ref:`Unit testing`
