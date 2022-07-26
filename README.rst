========
Overview
========

# Spatial Overlay Operations


## Package Structure:
	The library has been created in two main packages.

	## First package: categorical_overlay_operations

		As its name says, it overlay operations for categorical data


	## Second package: numerical_overlay_operations

	As its name says, it overlay operations for numerical data.
	


* Free software: MIT license

Installation
============

::

    pip install spatialOverlayOperations

You can also install the in-development version with::

    pip install git+ssh://git@https://github.com/PhilipeRLeal/Spatial_overlay_operations.git/PhilipeRLeal/spatialOverlayOperations.git@master

Documentation
=============


https://spatialOverlayOperations.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
