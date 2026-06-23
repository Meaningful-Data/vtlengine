###
API
###
The ``API`` package contains all the methods to handle VTL scripts and perform validations,
execute over data and other functionalities.

Overview
********

The VTL Engine API provides the following methods:

* :meth:`~vtlengine.semantic_analysis` — Validate a VTL script and compute
  the data structures of the datasets it creates. Supports VTL JSON,
  SDMX structure files, and ``pysdmx`` objects.
* :meth:`~vtlengine.run` — Execute a VTL script over input datapoints.
  Accepts VTL JSON, SDMX-ML / SDMX-JSON structure files, and ``pysdmx``
  structure objects for structures; pandas DataFrames, plain CSV, or
  SDMX-ML / SDMX-CSV data files for datapoints (SDMX-JSON is supported
  for structures, not data).
* :meth:`~vtlengine.run_sdmx` — Run a VTL script against ``pysdmx``
  ``PandasDataset`` objects, mapping each ``Schema`` to a VTL data
  structure. Internally calls :meth:`~vtlengine.run`.
* :meth:`~vtlengine.generate_sdmx` — Generate a ``pysdmx``
  ``TransformationScheme`` object from a VTL script.
* :meth:`~vtlengine.prettify` — Format a VTL script to make it more
  readable.
* :meth:`~vtlengine.validate_dataset` — Validate input datapoints against
  the provided data structures.
* :meth:`~vtlengine.validate_value_domain` — Validate input value domains
  using a JSON Schema.
* :meth:`~vtlengine.validate_external_routine` — Validate external
  routines using both JSON Schema and SQLGlot.

Reference
*********

.. automodule:: vtlengine
    :members:
    :undoc-members:
    :exclude-members: create_ast, check_parser

