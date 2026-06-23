Pandas Engine
#############

The pandas engine is the **default** execution backend used by :meth:`vtlengine.run` and
:meth:`vtlengine.run_sdmx`. It runs every VTL operation against in-memory
`pandas <https://pandas.pydata.org/>`_ DataFrames and is selected whenever ``use_duckdb``
is left at its default value of ``False``.

.. note::
    Execution engines only apply to :meth:`vtlengine.run` and :meth:`vtlengine.run_sdmx`.
    :meth:`vtlengine.semantic_analysis` performs validation only and does not execute
    operators against data, so it is engine-agnostic.

Overview
********

* **Default**: nothing to opt into; calls to :meth:`vtlengine.run` and
  :meth:`vtlengine.run_sdmx` use it out of the box.
* **In-memory**: every dataset is materialised as a ``pandas.DataFrame`` and operators
  apply DataFrame transformations, joins, and groupbys.
* **Stable surface**: the result of every operation is itself a DataFrame and can be
  inspected, debugged, or post-processed with the full pandas API.

When to use it
**************

* Datasets fit comfortably in RAM (single-node, no spill-to-disk requirements).
* You want full interoperability with the pandas ecosystem — pass DataFrames in,
  receive DataFrames back, plug into pandas-based pipelines downstream.
* You are running smaller scripts or interactive exploration where startup time
  matters more than raw throughput.

Limitations
***********

* The entire dataset is held in memory; very large inputs (multi-GB) can exhaust
  available RAM since there is no spill-to-disk.
* No native support for S3 URIs in ``datapoints`` or ``output_folder`` — pass local
  paths or DataFrames, or switch to the DuckDB engine.
* **Always single-threaded**: VTL operators run sequentially on a single thread.
  Whatever vectorisation pandas or NumPy expose internally is the only available
  parallelism. Use the DuckDB engine when multi-threaded query execution matters.

Configuration
*************

The pandas engine respects the number-handling environment variables shared with the
rest of the engine:

* :ref:`output_number_significant_digits` — precision for arithmetic and CSV output.
* ``COMPARISON_ABSOLUTE_THRESHOLD`` — tolerance for Number comparison operators.

See :doc:`environment_variables` for full details.

When to switch to DuckDB
************************

Consider :doc:`duckdb_engine` if you need any of the following:

* Datasets that approach or exceed available RAM (DuckDB can spill to disk via
  ``VTL_TEMP_DIRECTORY`` or run on a file-backed database via ``VTL_USE_IN_MEMORY_DB=0``).
* Reading from or writing to S3 URIs (``s3://bucket/key``).
* Multi-threaded query execution (``VTL_THREADS``); the pandas engine is always
  single-threaded.
