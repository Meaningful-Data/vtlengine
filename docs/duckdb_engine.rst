DuckDB Engine
#############

The DuckDB engine transpiles VTL scripts to SQL and executes them on
`DuckDB <https://duckdb.org/>`_, an in-process analytical database. It is opt-in via the
``use_duckdb=True`` flag on :meth:`vtlengine.run` and :meth:`vtlengine.run_sdmx`.

.. note::
    Execution engines only apply to :meth:`vtlengine.run` and :meth:`vtlengine.run_sdmx`.
    :meth:`vtlengine.semantic_analysis` performs validation only and does not execute
    operators against data, so it is engine-agnostic.

Overview
********

* **Multi-threaded execution**: DuckDB runs queries across multiple worker threads
  (controlled by ``VTL_THREADS``). This is a key advantage over the pandas engine,
  which is always single-threaded.
* **Throughput**: vectorised SQL execution that beats per-row pandas operations on
  large inputs.
* **Memory headroom**: spill-to-disk and an optional file-backed database mean datasets
  larger than RAM can be processed.
* **S3 native**: reads from and writes to ``s3://`` URIs through DuckDB's
  `httpfs <https://duckdb.org/docs/extensions/httpfs/s3api.html>`_ extension.
* **Same return shape as pandas backend**: when ``output_folder`` is not provided,
  ``run`` materialises each result as a ``pandas.DataFrame`` so downstream code is
  identical to the pandas engine. When ``output_folder`` is set, results are written
  there and the returned datasets carry no in-memory data.

When to prefer it over :doc:`pandas_engine`
*******************************************

* Datasets are large (close to or larger than available RAM).
* The script runs over hundreds of millions of rows where vectorised SQL beats
  per-row pandas operations.
* Inputs or outputs live in S3-compatible object storage.
* You want to control how many CPU threads are used by the engine.

Enabling
********

Pass ``use_duckdb=True`` to ``run`` (or ``run_sdmx``). The recommended pattern is to
provide an ``output_folder`` so each result is streamed to a CSV file rather than
materialised in memory:

.. code-block:: python

    from pathlib import Path

    import pandas as pd

    from vtlengine import run

    script = "DS_A <- DS_1 * 10;"

    data_structures = {
        "datasets": [
            {"name": "DS_1",
             "DataStructure": [
                 {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                 {"name": "Me_1", "type": "Number",  "role": "Measure",    "nullable": True},
             ]}
        ]
    }

    datapoints = {"DS_1": pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10, 20, 30]})}

    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        output_folder=Path("./vtl-output"),
        use_duckdb=True,
    )

When no ``output_folder`` is provided, the dataset values returned in ``run_result.data``
are ``pandas.DataFrame`` objects materialised from DuckDB — the return shape matches the
pandas engine, so calling code does not need to change.

.. warning::
    Running on large datasets without an ``output_folder`` forces the engine to
    materialise every result fully into memory as a ``pandas.DataFrame``. This negates
    most of the throughput and memory-headroom advantages of the DuckDB backend and
    can drop performance significantly. If any individual output dataset is larger
    than available memory, the run will raise an out-of-memory error, since pandas
    requires the complete object to be materialised in memory. For anything beyond
    small/exploratory inputs, set ``output_folder``.

In-memory vs. file-backed database
**********************************

By default the engine creates an in-memory DuckDB database (``:memory:``). For datasets
that approach available RAM, switch to a file-backed database under
``VTL_TEMP_DIRECTORY``:

.. code-block:: bash

    export VTL_USE_IN_MEMORY_DB=0
    export VTL_TEMP_DIRECTORY=/var/lib/vtlengine/duckdb-spill

The engine creates a unique session sub-directory and removes it when the connection
closes.

Configuration
*************

The DuckDB backend reads the following environment variables. See
:doc:`environment_variables` for full descriptions, valid values, and defaults.

* ``VTL_MEMORY_LIMIT`` — memory budget for DuckDB (``"80%"``, ``"16GB"``, …).
* ``VTL_THREADS`` — worker thread count.
* ``VTL_TEMP_DIRECTORY`` — spill-to-disk and file-backed database location.
* ``VTL_MAX_TEMP_DIRECTORY_SIZE`` — cap on spill disk usage.
* ``VTL_USE_IN_MEMORY_DB`` — toggle in-memory vs. file-backed database.
* :ref:`vtl_duckdb_decimal_width` — DECIMAL precision (paired with
  :ref:`output_number_significant_digits` for the scale).
* ``VTL_SKIP_LOAD_VALIDATION`` — skip post-load validation (benchmarking only).

S3 URI support
**************

When ``use_duckdb=True`` you may pass S3 URIs as ``datapoints`` and as ``output_folder``:

.. code-block:: python

    run(
        script="DS_r <- DS_1;",
        data_structures=data_structures,
        datapoints="s3://my-bucket/input/DS_1.csv",
        output_folder="s3://my-bucket/output/",
        use_duckdb=True,
    )

Authentication uses the standard AWS environment variables (``AWS_ACCESS_KEY_ID``,
``AWS_SECRET_ACCESS_KEY``, ``AWS_DEFAULT_REGION``, optionally ``AWS_SESSION_TOKEN`` and
``AWS_ENDPOINT_URL`` for S3-compatible services). See the
:doc:`S3 Configuration <environment_variables>` section for the full list.

Behavioural notes
*****************

* **Decimal precision**: numbers are stored using the DuckDB ``DECIMAL(width, scale)``
  type. ``width`` and ``scale`` come from
  :ref:`vtl_duckdb_decimal_width` and :ref:`output_number_significant_digits`
  respectively.
* **Post-load validation**: after each table is created the engine runs no-duplicates,
  temporal column format, and DWI cardinality checks. Disable them only for
  benchmarking via ``VTL_SKIP_LOAD_VALIDATION``.
* **Storage compatibility**: the engine pins DuckDB's storage format to ``v1.4.0`` so
  file-backed databases remain readable by the supported DuckDB version range.
