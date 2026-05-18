################
Data Structures
################

Before the engine can run your VTL script, it needs to know what your
data looks like. How many columns does each dataset have? Are they
integers, strings, dates? Which columns uniquely identify a row, and
which carry the values you'll be transforming? That description — the
**data structure** — is one of the three required inputs to every run.

The concept and the vocabulary both come from SDMX. Each **component**
of a dataset (i.e. each column) has a name, a type, and a role:

* **Identifiers** are the columns that together uniquely identify a row.
  In SDMX they're called dimensions.
* **Measures** carry your actual observations — the values you'll be
  transforming.
* **Attributes** are extra metadata attached to an observation, like
  quality flags or units.

Components can also be marked **nullable** (allowed to contain nulls).
Identifiers, by definition, never are.

So how do you actually tell the engine about this structure? You have a
few options, and you pick whichever matches the data you already have.


******************
Option 1: VTL JSON
******************

The engine's native format is a small JSON document, formally specified
by the `VTL JSON Schema
<https://github.com/Meaningful-Data/vtlengine/blob/main/src/vtlengine/API/data/schema/json_schema_2.1.json>`_
shipped with the engine. You can write it directly as a Python ``dict``
and pass it in:

.. code-block:: python

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number",  "role": "Measure",    "nullable": True}
                ]
            }
        ]
    }

The shape mirrors what we just described: a list of datasets, each with
a name and a list of components. Each component has the four fields:

.. list-table::
    :widths: 20 15 65
    :header-rows: 1

    * - Field
      - Type
      - Description
    * - ``name``
      - string
      - Component name. Must match the column name in the datapoints.
    * - ``type``
      - string
      - VTL data type. See :doc:`data_types` for the full list.
    * - ``role``
      - string
      - One of ``Identifier``, ``Measure``, or ``Attribute``.
    * - ``nullable``
      - bool
      - Whether the component allows null values. Identifiers must be
        non-nullable.

If you have many datasets, or you'd rather keep your Python code tidy,
save the same JSON to a ``.json`` file and pass the ``Path`` instead.

Scalar inputs (used with ``run``'s ``scalar_values`` argument) can be
declared alongside datasets:

.. code-block:: json

    {
      "datasets": [ /* ... */ ],
      "scalars": [
        {"name": "Sc_1", "type": "Number"}
      ]
    }


*******************************
Option 2: SDMX structure files
*******************************

If you're already working with SDMX, you don't need to convert anything.
Hand the engine an SDMX-ML (``.xml``) or SDMX-JSON (``.json``) structure
file as a ``Path``, and the engine will read it via ``pysdmx``, extract
each ``DataStructureDefinition``, and translate the components to VTL's
equivalent representation internally. For the exact SDMX versions
supported (SDMX-ML 2.1 / 3.0 / 3.1, SDMX-JSON 2.0.0), see pysdmx's
`Formats and versions supported
<https://py.sdmx.io/api/io/general_reader.html#formats-and-versions-supported>`_.

.. code-block:: python

    from pathlib import Path
    from vtlengine import run

    result = run(
        script="DS_r <- TEST_DSD [calc Me_2 := OBS_VALUE * 2];",
        data_structures=Path("path/to/structure.xml"),
        datapoints=Path("path/to/data.csv"),
    )

By default, the SDMX DSD's (or Dataflow's) ID becomes the VTL dataset
name. If your script refers to the dataset by a different name, you can
remap it using ``sdmx_mappings`` — see :doc:`sdmx_inputs`.


*************************
Option 3: pysdmx objects
*************************

If you've already loaded your SDMX structures into memory through
``pysdmx`` — typically as ``Schema``, ``DataStructureDefinition``, or
``Dataflow`` objects — you can hand them straight to the engine, no
file paths needed.

.. code-block:: python

    from pathlib import Path

    from pysdmx.io import read_sdmx
    from vtlengine import run

    msg = read_sdmx(Path("path/to/structure.xml"))
    dsds = msg.get_data_structure_definitions()

    result = run(script=script, data_structures=dsds, datapoints=datapoints)

Behind the scenes the engine converts each pysdmx structure to VTL JSON,
using a public helper called :func:`to_vtl_json`. You can also call it
yourself when you need to inspect or tweak the result — see
`Inside the conversion`_ below.

A useful trick when the same ``Dataflow`` should appear in your script
under two different aliases (e.g. a previous-vs-current snapshot): call
:func:`to_vtl_json` twice with different ``dataset_name`` arguments. See
:ref:`sharing_one_dataflow_between_two_datasets` in :doc:`sdmx_inputs`.


********************
Mixing input formats
********************

You don't have to commit to a single format. If you have some structures
in VTL JSON and others in SDMX, pass a list mixing all of them:

.. code-block:: python

    data_structures = [
        {"datasets": [...]},                       # VTL JSON dict
        Path("structure.xml"),                     # SDMX-ML file
        some_pysdmx_dataflow,                      # pysdmx object
    ]

The engine walks the list, processes each element with the right loader,
and merges the resulting dataset definitions before execution.


**********************
Inside the conversion
**********************

This section is for the curious — you don't need it to get a script
running, but it's useful to understand what's happening when you hand
the engine an SDMX structure.

Whenever you pass an SDMX structure (a file path or a pysdmx object),
the engine calls :func:`vtlengine.files.sdmx_handler.to_vtl_json` to
translate it into the VTL JSON shape shown in Option 1.

.. code-block:: python

    from vtlengine.files.sdmx_handler import to_vtl_json

    vtl_structure = to_vtl_json(my_dsd)
    # {"datasets": [{"name": "...", "DataStructure": [...]}]}

It accepts three pysdmx types — ``Schema``, ``DataStructureDefinition``,
``Dataflow`` — plus an optional ``dataset_name`` to override the default
(which is the structure's ID). For a ``Dataflow``, the embedded DSD is
extracted automatically; if the Dataflow has no resolved DSD attached,
the function raises an ``InputValidationException``.

The mapping itself is deliberately simple. **Roles** translate one-to-one:

.. list-table::
    :widths: 30 30 40
    :header-rows: 1

    * - SDMX role
      - VTL role
      - Nullable?
    * - ``Role.DIMENSION``
      - ``Identifier``
      - ``false``
    * - ``Role.MEASURE``
      - ``Measure``
      - ``true``
    * - ``Role.ATTRIBUTE``
      - ``Attribute``
      - ``true``

**Nullability** falls out of the role: dimensions are never nullable;
everything else is.

**Types** are collapsed from the long SDMX list onto VTL's smaller set,
following the official SDMX-to-VTL correspondence published in the
`SDMX 3.0 Technical Notes
<https://sdmx.org/wp-content/uploads/SDMX_3-0-0_SECTION_6_FINAL-1_0.pdf>`_:

.. list-table::
    :widths: 50 50
    :header-rows: 1

    * - SDMX data type
      - VTL type
    * - ``String``, ``Alpha``, ``AlphaNumeric``, ``Numeric``, ``URI``,
        ``Month``, ``MonthDay``, ``Day``, ``Time``
      - ``String``
    * - ``BigInteger``, ``Integer``, ``Long``, ``Short``, ``Count``
      - ``Integer``
    * - ``Decimal``, ``Float``, ``Double``, ``InclusiveValueRange``,
        ``ExclusiveValueRange``, ``Incremental``
      - ``Number``
    * - ``Boolean``
      - ``Boolean``
    * - ``BasicTimePeriod``, ``GregorianTimePeriod``, ``GregorianYear``,
        ``GregorianYearMonth``, ``GregorianMonth``, ``GregorianDay``,
        ``DateTime``
      - ``Date``
    * - ``ObservationalTimePeriod``, ``StandardTimePeriod``,
        ``ReportingTimePeriod`` and all reporting period variants
        (Year, Semester, Trimester, Quarter, Month, Week, Day)
      - ``Time_Period``
    * - ``TimeRange``
      - ``Time``
    * - ``Duration``
      - ``Duration``

See :doc:`data_types` for the VTL type semantics and which casts are
supported.


.. seealso::

    * :doc:`vtl_scripts` — accepted forms of the script argument.
    * :doc:`datapoints` — how datapoints reference these structures by name.
    * :doc:`sdmx_inputs` — SDMX-specific patterns and ``sdmx_mappings``.
