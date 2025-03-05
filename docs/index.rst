VTL Engine Documentation
########################

The VTL Engine is a Python library that allows you to validate and run VTL scripts.
It is a Python-based library around the `VTL Language 2.0 <http://sdmx.org/?page_id=5096>`_

Useful links
************

- `MeaningfulData: who we are <https://www.meaningfuldata.eu>`_
- `Source Code <https://github.com/Meaningful-Data/vtlengine>`_
- `PyPI link <https://pypi.org/project/vtlengine>`_
- `Bug Tracker <https://github.com/Meaningful-Data/vtlengine/issues?q=is%3Aopen+is%3Aissue+label%3Abug>`_
- `New features Tracker <https://github.com/Meaningful-Data/vtlengine/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement>`_

Installation
************

Requirements
============

The VTL Engine requires Python 3.10 or higher.


Install with pip
================

To install the VTL Engine on any Operating System, you can download it from `pip <https://pypi.org/project/vtlengine>`_:

.. code-block:: bash

    pip install vtlengine

.. important::
    It is recommended to install any python package in a virtual environment.
    Please follow `these steps <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_

S3 Extra
========

If you would like to use the S3 storage, you need to install the `s3` extra:

.. code-block:: bash

    pip install vtlengine[s3]

The S3 extra is based on the pandas[aws] package, which requires to set up some environment variables. Please check the `boto3 documentation <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables>`_

.. toctree::

    index
    walkthrough
    api


