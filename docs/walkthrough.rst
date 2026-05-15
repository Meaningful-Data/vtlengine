########################
10 minutes to VTL Engine
########################

The VTL Engine is a Python library that runs
`VTL 2.1 <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf>`_
scripts against tabular data. You bring three things — a script, a
description of your data, and the data itself — and the engine parses,
type-checks, and executes the whole thing.

There are two main ways to call it, depending on where your data
comes from:

* :ref:`run <run-section>` — for plain tabular workflows: pandas
  DataFrames or CSV files, with structures described in VTL JSON.
* :ref:`run_sdmx <run-sdmx-section>` — **the SDMX path**: load your
  SDMX files through ``pysdmx`` and execute a VTL script against them
  in a single call. This is the recommended path if you're working
  with SDMX data.

.. tip::
    **Already working with SDMX data?** Skip to the
    :ref:`Run SDMX <run-sdmx-section>` section below for a complete
    "load an SDMX file → run a VTL transformation" example. For
    additional SDMX patterns (file inputs, dataflow mappings,
    ``TransformationScheme`` as the script), see :doc:`sdmx_inputs`.

Both methods accept exactly the same script. They differ in **how the
data structures and datapoints are supplied** — plus a couple of related
arguments you can read about under *Optional* below.

If you just want to validate a script without running it, see
:ref:`semantic_analysis` later on this page. The rest of the API —
:meth:`~vtlengine.generate_sdmx`, :meth:`~vtlengine.validate_dataset`,
:meth:`~vtlengine.validate_value_domain`, and
:meth:`~vtlengine.validate_external_routine` — is documented in
:doc:`api`.

*************
What you need
*************

Every run takes the same three required inputs, in the same conceptual
roles, regardless of which method you use:

* The **VTL script** — the transformations you want to apply. A Python
  ``str``, a ``Path`` to a ``.vtl`` file, or a ``pysdmx``
  ``TransformationScheme``. See :doc:`vtl_scripts`.
* The **data structures** — the description of every input dataset (what
  columns it has, of what types, in what role). The format depends on
  the method. See :doc:`data_structures`.
* The **datapoints** — the actual rows. Again, the format depends on
  the method. See :doc:`datapoints`.

A few optional inputs let you extend or tune the run:

* ``value_domains`` — codelists referenced by the ``in`` operator.
  See :doc:`extra_inputs`.
* ``external_routines`` — SQL snippets invoked from inside ``eval``.
  See :doc:`extra_inputs`.
* ``scalar_values`` — scalar inputs to the script (``run`` only). See
  Example 2 below.
* ``output_folder`` — write results to disk instead of returning them
  in memory.
* ``sdmx_mappings`` — alias an SDMX short-URN to whatever name the
  script uses (exposed as ``mappings`` in ``run_sdmx``).
  See :doc:`sdmx_inputs`.


.. _run-section:

***
Run
***

Use :meth:`vtlengine.run` when your data is in pandas DataFrames or CSV
files.

.. note::
    If you're working with SDMX data, prefer :ref:`run_sdmx <run-sdmx-section>`
    — it's the idiomatic path. ``run`` can also read SDMX inputs directly
    (file paths and ``pysdmx`` structure objects), but you have to pass
    structure and data as two separate arguments and manage the mapping
    yourself. See :doc:`sdmx_inputs` if you need to go that route.

For ``run`` you pass the structure and the data separately:

* **Data structures** — a VTL JSON ``dict`` or a ``Path`` to a ``.json``
  file. If you have SDMX structure files or ``pysdmx`` objects instead,
  the engine accepts those too — see :doc:`sdmx_inputs`.
* **Datapoints** — a ``dict`` keyed by dataset name, where each value
  is a ``pandas.DataFrame`` or a ``Path`` to a plain CSV file. For SDMX
  data files (SDMX-ML, SDMX-CSV), see :doc:`sdmx_inputs`.

The method returns a dictionary of all generated datasets, keyed by
their VTL output name. If you'd rather have the results written to disk
than held in memory, set ``output_folder``.

Before executing, ``run`` performs two validations — either can raise an
error and stop the run:

* **Semantic analysis** — the same check :meth:`vtlengine.semantic_analysis`
  performs, applied to your script against the declared structures.
* **Data load analysis** — a basic check that the data's column names
  and types line up with the structures.

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

Use :meth:`vtlengine.run_sdmx` when you're working with SDMX data. In SDMX, data and structure are inseparable — every dataset
arrives with its structure attached. ``pysdmx`` mirrors this by
representing each dataset as a single object (the class is called
``PandasDataset`` in its model) that carries the rows and the structure
together. You hand those objects directly to ``run_sdmx``, which
unpacks the bundles internally and delegates execution to
:meth:`vtlengine.run`.

If you're curious about the internals: each dataset exposes its rows
as ``.data`` (a ``pandas.DataFrame``) and its structure as ``.structure``
(a ``pysdmx`` ``Schema``). When the ``Schema`` is present, the engine
auto-casts the DataFrame columns to the declared component types, so
you don't have to.

For the common case — your input list contains a single dataset and
your script references a single input — the engine auto-matches them.
Whatever name your script uses for the input becomes the alias for that
dataset; no ``mappings`` argument is required.

If your script references multiple inputs, or your input list contains
datasets the script doesn't all use, you must pass a ``mappings``
argument: a ``dict`` mapping each SDMX short-URN to a VTL alias, or a
``pysdmx`` ``VtlDataflowMapping`` object.

.. note::
    **Multiple datasets in one source.** The 1-to-1 auto-match only
    works when both the input list and the script's inputs have exactly
    one entry each. As soon as either side has more than one, you must
    pass an explicit ``mappings`` argument: every mapped dataset must
    appear in the script (extras error out), and every script input
    must be covered by the mapping.

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

The example below loads a small SDMX-ML structure file and a matching data
file, then applies a single VTL transformation that adds a new measure
``Me_4`` equal to the existing ``OBS_VALUE`` column.

The structure file defines one ``Dataflow=MD:TEST_DF(1.0)`` with
components ``DIM_1``, ``DIM_2``, and ``OBS_VALUE``; the data file contains
six observations. The script references the dataset as ``DS_1``. Because
there's only one dataset in the list and the script references only one
input, the engine auto-matches them — no ``mappings`` argument needed.

.. code-block:: python

    from pathlib import Path

    from pysdmx.io import get_datasets

    from vtlengine import run_sdmx

    data = Path("docs/_static/data.xml")
    structure = Path("docs/_static/metadata.xml")
    datasets = get_datasets(data, structure)

    # Add a new measure Me_4 with the same value as OBS_VALUE, and
    # call the resulting dataset DS_r.
    script = "DS_r <- DS_1 [calc Me_4 := OBS_VALUE];"
    print(run_sdmx(script, datasets)['DS_r'].data)


.. csv-table:: Returns:
    :file: _static/DS_r_run_sdmx.csv
    :header-rows: 1

For VTL syntax reference, see the
`VTL 2.1 manual <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf>`_.

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


.. _prettify:

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
