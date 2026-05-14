########################
10 minutes to VTL Engine
########################

The VTL Engine validates and executes
`VTL 2.1 <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf>`_
scripts against tabular data. This page walks through the two main entry
points:

* :ref:`run <run-section>` — execute a script against pandas DataFrames or
  CSV files, with structures described in VTL JSON.
* :ref:`run_sdmx <run-sdmx-section>` — execute a script against ``pysdmx``
  ``PandasDataset`` objects, where the SDMX structure travels with the data.

Both methods accept the same script. They differ in **how the data
structures and datapoints are supplied**, plus a couple of related
arguments (see *Optional* below).

For semantic validation without execution, see :ref:`semantic_analysis` below.
For the full API — including :meth:`~vtlengine.generate_sdmx`,
:meth:`~vtlengine.validate_dataset`,
:meth:`~vtlengine.validate_value_domain`, and
:meth:`~vtlengine.validate_external_routine` — see :doc:`api`.

*************
What you need
*************

Every run takes three required inputs and a handful of optional ones.

**Required**

* **VTL script** — the transformations to apply. A ``str``, a ``Path`` to a
  ``.vtl`` file, or a ``pysdmx`` ``TransformationScheme`` (accepted by both
  methods).
* **Data structures** — the schema of every input dataset (components, types,
  roles). Accepted formats depend on the method (see below).
* **Datapoints** — the actual data. Accepted formats depend on the method
  (see below).

**Optional**

* ``value_domains`` — codelists referenced by the ``in`` operator.
  See :doc:`extra_inputs`.
* ``external_routines`` — SQL snippets used inside ``eval``.
  See :doc:`extra_inputs`.
* ``scalar_values`` — scalar inputs (``run`` only). See Example 2 below.
* ``output_folder`` — write results to disk instead of returning them in memory.
* ``sdmx_mappings`` — map an SDMX short-URN to the dataset name used in
  the script. Exposed as ``mappings`` in ``run_sdmx``.
  See :doc:`sdmx_inputs`.


.. _run-section:

***
Run
***

Use :meth:`vtlengine.run` when you have data in pandas DataFrames or CSV
files, or when you want to mix VTL JSON and SDMX inputs.

.. note::
    ``run`` natively reads SDMX structure files (``.xml`` / ``.json``) and
    ``pysdmx`` structure objects (``Schema``, ``DataStructureDefinition``,
    ``Dataflow``), so passing SDMX inputs does **not** require ``run_sdmx``.
    Use :ref:`run_sdmx <run-sdmx-section>` only when structure and data are
    already bundled together inside ``pysdmx`` ``PandasDataset`` objects
    (typically obtained from ``pysdmx.io.get_datasets``) — it's a convenience
    wrapper that unpacks them into the two arguments ``run`` expects.

**Data structures** — a VTL JSON ``dict``, or a ``Path`` to a ``.json``
file. For SDMX structure files or ``pysdmx`` objects, see
:doc:`sdmx_inputs`.

**Datapoints** — a ``dict`` keyed by dataset name, where each value is a
``pandas.DataFrame`` or a ``Path`` to a plain CSV file. For SDMX data
files (SDMX-ML, SDMX-JSON, SDMX-CSV), see :doc:`sdmx_inputs`.

The method returns a dictionary containing all generated datasets. If the
``output_folder`` parameter is set, results are written to that folder
instead of being kept in memory.

Two validations are performed before execution, either of which may raise an error:

* **Semantic analysis**: equivalent to running :meth:`vtlengine.semantic_analysis`.
* **Data load analysis**: a basic check of data structure names and types.

.. seealso::

    * :doc:`duckdb_engine` — execute on DuckDB by passing ``use_duckdb=True``.
    * :doc:`sdmx_inputs` — SDMX file inputs, ``pysdmx`` structure objects,
      and the shared-Dataflow pattern.
    * :doc:`extra_inputs` — value domains, external routines, and using
      ``Path`` objects for all inputs.

=====================
Example 1: Simple run
=====================

.. code-block:: python

    from vtlengine import run
    import pandas as pd

    script = """
        DS_A <- DS_1 * 10;
    """

    data_structures = {
        'datasets': [
            {'name': 'DS_1',
             'DataStructure': [
                 {'name': 'Id_1',
                  'type':
                      'Integer',
                  'role': 'Identifier',
                  'nullable': False},
                 {'name': 'Me_1',
                  'type': 'Number',
                  'role': 'Measure',
                  'nullable': True}
             ]
             }
        ]
    }

    data_df = pd.DataFrame(
        {"Id_1": [1, 2, 3],
         "Me_1": [10, 20, 30]})

    datapoints = {"DS_1": data_df}

    run_result = run(script=script, data_structures=data_structures,
                     datapoints=datapoints)

    print(run_result["DS_A"].data)



.. csv-table:: Returns:
    :file: _static/DS_A_run.csv
    :header-rows: 1

.. note::
    ``run`` returns a ``dict`` keyed by output name. Each value is a
    :class:`Dataset <vtlengine.Model.Dataset>` (with ``.data`` as a
    ``pandas.DataFrame``) or a :class:`Scalar <vtlengine.Model.Scalar>`
    (with ``.value``). The example above accesses
    ``run_result["DS_A"].data`` to get the DataFrame.


=================================
Example 2: Run with Scalar Values
=================================

``run`` supports scalar inputs to the script via the ``scalar_values``
parameter. When an ``output_folder`` is provided, the engine generates
CSV files containing the results of the script execution. Scalar results
are saved as a CSV file containing all resulting scalar values.

.. code-block:: python

    from vtlengine import run
    import pandas as pd

    script = """
        DS_r <- DS_1[filter Me_1 = Sc_1];
        Sc_r <- Sc_1 + 10;
        Sc_r2 <- Sc_r * 2;
        Sc_r3 <- null;
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ],
        "scalars": [
            {
                "name": "Sc_1",
                "type": "Number",
            }
        ],
    }

    data_df = pd.DataFrame({"Id_1": [1, 2, 3], "Me_1": [10, 20, 30]})
    datapoints = {"DS_1": data_df}
    scalars = {"Sc_1": 20}

    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        scalar_values=scalars,
        return_only_persistent=True
    )

    print(run_result)


Returns:

.. csv-table::
    :file: _static/DS_r_run_with_scalars.csv
    :header-rows: 1

.. csv-table::
    :file: _static/Sc_r_run_with_scalars.csv
    :header-rows: 1


.. _run-sdmx-section:

********
Run SDMX
********

Use :meth:`vtlengine.run_sdmx` when you already have ``pysdmx``
``PandasDataset`` objects — typically obtained via
``pysdmx.io.get_datasets``. Internally, ``run_sdmx`` converts the SDMX
inputs and delegates execution to :meth:`vtlengine.run`.

Unlike ``run``, **structures and datapoints travel together** inside each
``PandasDataset``; you pass a single ``datasets`` argument instead of two.

By default, each ``PandasDataset``'s ``Schema`` ID is used as the VTL
dataset name (e.g. ``DataStructure=MD:TEST_DS(1.0)`` → ``TEST_DS``). If
the script refers to a dataset by a different name, pass a ``mappings``
argument — a ``dict`` mapping the SDMX short-URN to the VTL alias, or a
``pysdmx`` ``VtlDataflowMapping`` object.

For details on reading and writing SDMX datasets, see the
`pysdmx documentation <https://py.sdmx.io/howto/vtl_handling.html>`_.

.. important::
    The short-urn is the meaningful part of the URN. The format is:
    SDMX_type=Agency:ID(Version).

    Example:

    Dataflow=MD:TEST_DF(1.0) is the short-urn for
    urn:sdmx:org.sdmx.infomodel.datastructure.Dataflow=MD:TEST_DF(1.0)

.. seealso::

    :doc:`sdmx_inputs` — using a ``TransformationScheme`` as the script,
    ``VtlDataflowMapping`` for SDMX-to-VTL aliasing, and the shared-Dataflow
    pattern.

================================
Example 3: Run from SDMX Dataset
================================

.. code-block:: python

    from pathlib import Path

    from pysdmx.io import get_datasets

    from vtlengine import run_sdmx

    data = Path("docs/_static/data.xml")
    structure = Path("docs/_static/metadata.xml")
    datasets = get_datasets(data, structure)
    script = "DS_r <- DS_1 [calc Me_4 := OBS_VALUE];"
    print(run_sdmx(script, datasets)['DS_r'].data)


.. csv-table:: Returns:
    :file: _static/DS_r_run_sdmx.csv
    :header-rows: 1

Files used in the example:

- :download:`data.xml <_static/data.xml>`
- :download:`metadata.xml <_static/metadata.xml>`


.. _semantic_analysis:

*****************
Semantic Analysis
*****************

The :meth:`vtlengine.semantic_analysis` method validates the correctness of a VTL script and computes the data structures of
the datasets generated by the script itself (a prerequisite for semantic analysis).

* If the VTL script is correct, the method returns a dictionary containing the data structures of all datasets generated by the script.
* If the VTL script is incorrect, a `SemanticError` is raised.

The ``data_structures`` parameter accepts multiple formats:

- **VTL JSON format**: Dictionaries or paths to ``.json`` files
- **SDMX structure files**: Paths to SDMX-ML (``.xml``) or SDMX-JSON (``.json``) files
- **pysdmx objects**: ``Schema``, ``DataStructureDefinition``, or ``Dataflow`` objects
- **Mixed lists**: Any combination of the above formats

======================
Example 4: Correct VTL
======================

.. code-block:: python

    from vtlengine import semantic_analysis

    script = """
        DS_A <- DS_1 * 10;
    """

    data_structures = {
        'datasets': [
            {'name': 'DS_1',
             'DataStructure': [
                 {'name': 'Id_1',
                  'type':
                      'Integer',
                  'role': 'Identifier',
                  'nullable': False},
                 {'name': 'Me_1',
                  'type': 'Number',
                  'role': 'Measure',
                  'nullable': True}
             ]
             }
        ]
    }

    sa_result = semantic_analysis(script=script, data_structures=data_structures)

    print(sa_result)

Returns:

.. code-block:: python

    {'DS_A': Dataset(name='DS_A', components={'Id_1': Component(name='Id_1', data_type="Integer", role="Identifier", nullable=False), 'Me_1': Component(name='Me_1', data_type="Number", role="Measure", nullable=True)}, data=None)}

========================
Example 5: Incorrect VTL
========================

Compared to Example 4, the only difference is that `Me_1` uses a `String` data type instead of `Number`.

.. code-block:: python

    from vtlengine import semantic_analysis

    script = """
        DS_A <- DS_1 * 10;
    """

    data_structures = {
        'datasets': [
            {'name': 'DS_1',
             'DataStructure': [
                 {'name': 'Id_1',
                  'type':
                      'Integer',
                  'role': 'Identifier',
                  'nullable': False},
                 {'name': 'Me_1',
                  'type': 'String',
                  'role': 'Measure',
                  'nullable': True}
             ]
             }
        ]
    }

    sa_result = semantic_analysis(script=script, data_structures=data_structures)

    print(sa_result)


Raises the following Error:

.. code-block:: python

    raise SemanticError(code="1-1-1-2",
    vtlengine.Exceptions.SemanticError: ('Invalid implicit cast from String and Integer to Number.', '1-1-1-2')


====================================
Example 6: Using SDMX Structures
====================================

The ``semantic_analysis`` function can also accept SDMX structure files or pysdmx objects:

.. code-block:: python

    from pathlib import Path

    from vtlengine import semantic_analysis

    script = """
        DS_A <- DS_1 * 10;
    """

    # Using an SDMX-ML structure file
    sdmx_structure = Path("path/to/structure.xml")

    sa_result = semantic_analysis(script=script, data_structures=sdmx_structure)

    print(sa_result)

Using pysdmx objects directly:

.. code-block:: python

    from pathlib import Path

    from pysdmx.io import read_sdmx

    from vtlengine import semantic_analysis

    script = """
        DS_A <- DS_1 * 10;
    """

    # Load structure using pysdmx
    msg = read_sdmx(Path("path/to/structure.xml"))
    dsds = msg.get_data_structure_definitions()

    sa_result = semantic_analysis(script=script, data_structures=dsds)

    print(sa_result)


********
Prettify
********

The :meth:`vtlengine.prettify` method formats a VTL script to improve readability.

.. code-block:: python

    from vtlengine import prettify
    script = """
        define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

        DS_r <- check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY dataset);
        """
    prettified_script = prettify(script)
    print(prettified_script)

Returns:

.. code-block:: text


    define hierarchical ruleset accountingEntry(variable rule ACCOUNTING_ENTRY) is
        B = C - D
        errorcode "Balance (credit-debit)"
        errorlevel 4;

        N = A - L
        errorcode "Net (assets-liabilities)"
        errorlevel 4
    end hierarchical ruleset;

    DS_r <-
        check_hierarchy(
            BOP,
            accountingEntry,
            rule ACCOUNTING_ENTRY);


For more information on usage, please refer to the `API documentation <https://docs.vtlengine.meaningfuldata.eu/api.html>`_
