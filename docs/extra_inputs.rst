############
Extra Inputs
############

Both :meth:`vtlengine.run` and :meth:`vtlengine.semantic_analysis`
accept two optional parameters — ``value_domains`` and
``external_routines`` — that extend a VTL script with
membership checks and SQL-based transformations respectively.

This page documents the definition format, input options,
VTL syntax, and validation for each feature.

.. seealso::

    - :ref:`example_5_run_with_multiple_value_domains_and_external_routines`
      — Combined example using both features as dictionaries
    - :ref:`example_6_run_using_paths`
      — Combined example using both features as Path objects
    - `VTL 2.1 Reference Manual
      <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf>`_
      — Full VTL specification


Value Domains
*************

A Value Domain is a named set of unique values that share
a common data type. Value domains are used with the ``in``
and ``not_in`` operators to perform membership checks
in VTL scripts, and a Component or a scalar can be defined on one
(see :ref:`components_on_value_domains` and
:ref:`scalars_on_value_domains`).


Definition Format
=================

Each value domain is a JSON object with three required fields:

.. list-table::
    :widths: 20 15 65
    :header-rows: 1

    * - Field
      - Type
      - Description
    * - ``name``
      - string
      - Unique identifier referenced in the VTL script.
    * - ``type``
      - string
      - Data type of the values. See supported types below.
    * - ``setlist``
      - array
      - List of unique values belonging to the domain.
        Items must match the declared ``type``.

Example:

.. code-block:: json

    {
        "name": "Countries",
        "type": "String",
        "setlist": ["DE", "FR", "IT", "ES"]
    }

Multiple value domains can be provided as a JSON array:

.. code-block:: json

    [
        {
            "name": "Countries",
            "type": "String",
            "setlist": ["DE", "FR", "IT"]
        },
        {
            "name": "Thresholds",
            "type": "Integer",
            "setlist": [10, 20, 50, 100]
        }
    ]


Supported Types
===============

.. list-table::
    :widths: 25 75
    :header-rows: 1

    * - Type
      - ``setlist`` item type
    * - ``Integer``
      - JSON integer (e.g. ``1``, ``42``)
    * - ``Number``
      - JSON number (e.g. ``3.14``, ``100``)
    * - ``String``
      - JSON string (e.g. ``"DE"``, ``"hello"``)
    * - ``Boolean``
      - JSON boolean (``true`` or ``false``)
    * - ``Date``
      - JSON string in date format (e.g. ``"2024-01-15"``)
    * - ``Time_Period``
      - JSON string in time period format
        (e.g. ``"2024Q1"``)
    * - ``Time``
      - JSON string as ISO 8601 interval
        (e.g. ``"2024-01-01/2024-12-31"``)
    * - ``Duration``
      - JSON string in duration format (e.g. ``"P1Y"``)


Input Options
=============

The ``value_domains`` parameter accepts the following formats:

- **Dictionary**: A single value domain as a Python dict.
- **Path to a JSON file**: A ``Path`` pointing to a ``.json``
  file containing one or more value domain definitions.
- **Path to a directory**: A ``Path`` pointing to a directory;
  all ``.json`` files in the directory are loaded.
- **ValueDomain object**: A ``vtlengine.Model.ValueDomain``, for example one
  built from an SDMX codelist.
- **List**: A list mixing any of the above formats.

.. code-block:: python

    from pathlib import Path

    from vtlengine.DataTypes import String
    from vtlengine.Model import ValueDomain

    # Single dict
    value_domains = {
        "name": "Countries",
        "type": "String",
        "setlist": ["DE", "FR", "IT"],
    }

    # Path to file
    value_domains = Path("data/value_domains.json")

    # Path to directory (loads all .json files)
    value_domains = Path("data/value_domains/")

    # ValueDomain object
    value_domains = ValueDomain(name="Countries", type=String, setlist=["DE", "FR", "IT"])

    # List of mixed formats
    value_domains = [
        {"name": "Countries", "type": "String", "setlist": ["DE", "FR"]},
        Path("data/extra_domains.json"),
        ValueDomain(name="Currencies", type=String, setlist=["EUR", "USD"]),
    ]


VTL Usage
=========

Value domains are referenced in VTL scripts using the ``in``
and ``not_in`` operators. These can be used in both scalar
and component contexts.

.. code-block:: text

    /* Scalar membership: returns Boolean */
    DS_r <- DS_1 [calc Me_2 := Me_1 in Countries];

    /* Negated membership */
    DS_r <- DS_1 [calc Me_2 := Me_1 not_in Countries];

    /* Filter rows where a component belongs to a domain */
    DS_r <- DS_1 [filter Me_1 in Thresholds];


Example
=======

.. code-block:: python

    import pandas as pd

    from vtlengine import run

    script = """
        DS_r <- DS_1 [calc Me_2 := Me_1 in Countries];
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Me_1",
                        "type": "String",
                        "role": "Measure",
                        "nullable": True,
                    },
                ],
            }
        ]
    }

    datapoints = {
        "DS_1": pd.DataFrame(
            {"Id_1": [1, 2, 3], "Me_1": ["DE", "US", "FR"]}
        ),
    }

    value_domains = {
        "name": "Countries",
        "type": "String",
        "setlist": ["DE", "FR", "IT"],
    }

    result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        value_domains=value_domains,
    )

    print(result["DS_r"])


.. _components_on_value_domains:

Components Defined on a Value Domain
====================================

The ``type`` (or ``data_type``) of a Component in the data
structures can name a value domain given in ``value_domains``,
instead of a basic type. The Component is then defined on that
value domain and has its basic type:

.. code-block:: json

    {"name": "Id_2", "type": "VD_FLOW", "role": "Identifier", "nullable": false}

A type that is neither a basic type nor the name of a value
domain given raises ``0-2-1-1``. Loading the datapoints checks
that every value of the Component is a value of its value
domain, and raises ``0-3-1-6`` on another one. Nulls follow
``nullable``, as with a basic type.

The operators handle the Component as one of its basic type:
the same type checks and the same result types. ``Id_2 || "_X"``
is a ``String``, and ``Me_1 + 1`` on a value domain of integers
is an ``Integer``. The Components that an operator keeps as they
are keep their value domain: the Identifiers, the Components that
``filter``, ``keep``, ``drop``, ``rename``, ``sub`` and the joins
keep, and a Component written alone in ``calc``
(``calc Me_3 := Id_2``). ``run()`` and ``semantic_analysis()``
write ``"type": "VD_FLOW"`` for them.

The operators that put the values of several operands into one
Component (the set operators, the Identifiers that a join
matches, ``nvl``, ``if-then-else`` and ``case``) follow the VTL
type conversion rules:

- Operands on the same value domain give that value domain.
- A value domain and its basic type give the value domain: the
  values of the basic type are converted into it (downcast), and
  one that is not a value of the value domain raises ``2-1-5-4``.
- Two different value domains on the same basic type give that
  basic type, and any other mix follows the promotion of the basic
  types (a value domain of integers and a ``Number`` give a
  ``Number``).

.. code-block:: text

    /* Id_2 keeps VD_FLOW; the Strings of DS_2 must be values of it */
    DS_r <- union(DS_1, DS_2);

    /* Me_2 is on the value domain of Me_1, and 0 must be one of its values */
    DS_r <- DS_1 [calc Me_2 := nvl(Me_1, 0)];

``cast`` converts a value of the basic type, or of another value
domain on it, into a value domain, with the same check:

.. code-block:: text

    /* Me_2 is on VD_1, and every value of Me_1 must be one of its values */
    DS_r <- DS_1 [calc Me_2 := cast(Me_1, VD_1)];

``pivot`` needs the Identifier it pivots defined on a value domain:
the result has one Measure per value of the value domain, named after
the value, of the type (and on the value domain) of the pivoted
Measure. The other Identifiers group the data points, a value with no
data point in a group leaves its Measure null, the other Measures and
the non-viral Attributes are dropped, and the viral Attributes of a
group are combined by their propagation rule. An Identifier on no
value domain raises ``1-1-6-18``.

.. code-block:: text

    /* Id_2 is on a value domain of A and B: DS_r has the Measures A and B */
    DS_r <- DS_1 [pivot Id_2, Me_1];

Value domains also appear in the VTL-DL definitions:

- A datapoint ruleset on value domains binds them, or their aliases,
  in order to the components listed in ``check_datapoint ...
  components``. A component of the basic type of its value domain
  has its values checked against it (``2-1-5-4``), and one of
  another basic type raises ``1-1-10-12``. A value domain that is
  not given is not checked. Without ``components``, the names of the
  signature are read as components of the Data Set. For backwards
  compatibility, a ruleset whose value domains are all named as
  components of the Data Set reads those components, whatever
  components the call lists.
- A viral propagation rule on a value domain applies to the viral
  attributes defined on it, unless a rule on the variable overrides
  it.
- A value domain types a scalar or a component parameter of a user
  defined operator (``x VD_FLOW``, ``x component<VD_FLOW>``): the
  argument is of the basic type of the value domain, and its values
  must be values of it (``2-1-5-4``). In the body a scalar parameter
  is defined on the value domain, and a component parameter behaves
  as its basic type.

.. code-block:: text

    define datapoint ruleset DPR_1 (valuedomain VD_FLOW as F, VD_AMOUNT as A) is
        when F = "CREDIT" then A >= 0
    end datapoint ruleset;

    define viral propagation VP_1 (valuedomain CL_CONF) is
        when "C" then "C";
        else "F"
    end viral propagation;

    define operator with_suffix (x component<VD_FLOW>)
        returns component<string> is
            x || "_X"
    end operator;

    DS_r <- check_datapoint(DS_1, DPR_1 components Id_2, Me_1 all);
    DS_s <- DS_1 [calc Me_2 := with_suffix(Id_2)];

The value domains ``errorcode_vd`` and ``errorlevel_vd``, when given,
type the error codes and the error levels of ``check``,
``check_datapoint`` and ``check_hierarchy``: the ``errorcode`` and
``errorlevel`` Measures take their basic types, and every error code
and level of a rule or of ``check`` must be one of their values
(``1-1-10-13``). Without them, the Measures keep their usual types.

.. _scalars_on_value_domains:

Scalars Defined on a Value Domain
=================================

The ``type`` (or ``data_type``) of a scalar in the data structures
can name a value domain given in ``value_domains``, as the type of a
Component can. The scalar is then defined on that value domain and
has its basic type:

.. code-block:: json

    {"scalars": [{"name": "SC_1", "type": "VD_FLOW"}]}

The value given in ``scalar_values`` must be one of the values of
the value domain, or ``run()`` raises ``0-1-1-36``. A null value is
allowed.

The operators handle a scalar on a value domain as a Component on
it:

- An assignment (``SC_r <- SC_1``) and a scalar written alone in
  ``calc`` (``calc Me_2 := SC_1``) keep the value domain.
- ``nvl``, ``if-then-else`` and ``case`` follow the type conversion
  rules above. A value of the basic type put into the value domain
  must be one of its values (``2-1-5-4``), and two different value
  domains give the basic type.
- ``cast(SC_2, VD_FLOW)`` gives a scalar on ``VD_FLOW``.
- An operator that computes a new value gives its basic type:
  ``SC_1 || "_X"`` is a ``String``.
- A scalar parameter of a user defined operator typed by a value
  domain is defined on it. One of a basic type keeps the value domain
  of an argument of that same type.

``run()`` and ``semantic_analysis()`` write ``"type": "VD_FLOW"`` for
the scalar results on a value domain.

.. code-block:: text

    /* SC_r is on VD_FLOW, and "OTHER" must be one of its values */
    SC_r <- nvl(SC_1, "OTHER");

    /* Me_2 is on VD_FLOW */
    DS_r <- DS_1 [calc Me_2 := SC_1];

The value domains in the ``returns`` of a user defined operator are
not supported (``1-3-2-6``).


Validation
==========

Use :meth:`vtlengine.validate_value_domain` to validate
the JSON structure of value domains before execution:

.. code-block:: python

    from vtlengine import validate_value_domain

    value_domains = {
        "name": "Countries",
        "type": "String",
        "setlist": ["DE", "FR", "IT"],
    }

    # Raises an exception if the structure is invalid
    validate_value_domain(value_domains)


External Routines
*****************

External Routines allow VTL scripts to execute SQL queries
through the ``eval()`` operator. Queries are executed in
a sandboxed `DuckDB <https://duckdb.org/>`_ environment.

.. note::

    Currently, only SQL external routines are supported.
    The ``language`` parameter in the ``eval()`` call must
    be set to ``"SQL"``.


Definition Format
=================

Each external routine is a JSON object with two required fields:

.. list-table::
    :widths: 20 15 65
    :header-rows: 1

    * - Field
      - Type
      - Description
    * - ``name``
      - string
      - Identifier referenced in the VTL ``eval()`` call.
    * - ``query``
      - string
      - SQL query to execute. Table names in the query must
        match the dataset names passed as operands.

**JSON format**:

.. code-block:: json

    {
        "name": "SQL_1",
        "query": "SELECT Id_1, SUM(Me_1) AS Me_1 FROM DS_1 GROUP BY Id_1;"
    }

Multiple routines can be provided as a JSON array:

.. code-block:: json

    [
        {
            "name": "SQL_1",
            "query": "SELECT Id_1, COUNT(*) AS Me_1 FROM DS_1 GROUP BY Id_1;"
        },
        {
            "name": "SQL_2",
            "query": "SELECT Id_1, Me_1 FROM DS_1 WHERE Me_1 > 10;"
        }
    ]


Input Options
=============

The ``external_routines`` parameter accepts the following formats:

- **Dictionary**: A single routine as a Python dict.
- **Path to a file**: A ``Path`` pointing to a ``.json`` file.
- **Path to a directory**: A ``Path`` pointing to a directory;
  all ``.json`` files in the directory are loaded.
- **ValueDomain object**: A ``vtlengine.Model.ValueDomain``, for example one
  built from an SDMX codelist.
- **List**: A list mixing any of the above formats.

.. code-block:: python

    from pathlib import Path

    from vtlengine.DataTypes import String
    from vtlengine.Model import ValueDomain

    # Single dict
    external_routines = {
        "name": "SQL_1",
        "query": "SELECT Id_1, COUNT(*) AS Me_1 FROM DS_1 GROUP BY Id_1;",
    }

    # Path to file
    external_routines = Path("data/SQL_1.json")

    # Path to directory (loads all .json files)
    external_routines = Path("data/routines/")

    # List of mixed formats
    external_routines = [
        {"name": "SQL_1", "query": "SELECT * FROM DS_1;"},
        Path("data/SQL_2.json"),
    ]


VTL Syntax
==========

The ``eval()`` operator invokes an external routine:

.. code-block:: text

    DS_r := eval(
        SQL_NAME(DS_1, DS_2)
        language "SQL"
        returns dataset {
            identifier<integer> Id_1,
            measure<number> Me_1
        }
    );

- **SQL_NAME**: Name matching the external routine definition.
- **Operands**: Input datasets passed to the SQL query
  (``DS_1``, ``DS_2``, etc.).
- **language**: Must be ``"SQL"``.
- **returns dataset**: Defines the output structure with
  component roles (``identifier``, ``measure``,
  ``attribute``) and types.

.. note::

    The column names in the SQL query result must match the
    component names declared in the ``returns dataset`` clause.


Security
========

External routines are executed in a sandboxed DuckDB
in-memory database with the following restrictions:

- ``INSTALL`` and ``LOAD`` commands are forbidden.
- URLs (``http://``, ``https://``) in ``FROM`` clauses
  are forbidden.
- External file access is disabled.
- Extension loading is disabled.
- Configuration is locked after initialization.
- Results are checked for infinite values.


Examples
========

Using a dictionary
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pandas as pd

    from vtlengine import run

    script = """
        DS_r <- eval(
            SQL_1(DS_1)
            language "SQL"
            returns dataset {
                identifier<integer> Id_1,
                measure<number> Me_1
            }
        );
    """

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {
                        "name": "Id_1",
                        "type": "Integer",
                        "role": "Identifier",
                        "nullable": False,
                    },
                    {
                        "name": "Me_1",
                        "type": "Number",
                        "role": "Measure",
                        "nullable": True,
                    },
                ],
            }
        ]
    }

    datapoints = {
        "DS_1": pd.DataFrame(
            {"Id_1": [1, 2, 3, 4, 5], "Me_1": [10, 20, 30, 40, 50]}
        ),
    }

    external_routines = {
        "name": "SQL_1",
        "query": "SELECT Id_1, Me_1 * 2 AS Me_1 FROM DS_1;",
    }

    result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        external_routines=external_routines,
    )

    print(result["DS_r"])

Validation
==========

Use :meth:`vtlengine.validate_external_routine` to validate
the JSON structure and SQL syntax before execution:

.. code-block:: python

    from vtlengine import validate_external_routine

    external_routines = {
        "name": "SQL_1",
        "query": "SELECT Id_1, SUM(Me_1) AS Me_1 FROM DS_1 GROUP BY Id_1;",
    }

    # Raises an exception if the structure or SQL is invalid
    validate_external_routine(external_routines)


Combined Examples
*****************

The following examples combine ``value_domains`` and ``external_routines``
in a single ``run`` call.

.. _example_5_run_with_multiple_value_domains_and_external_routines:

Using dictionaries
==================

Pass both ``value_domains`` and ``external_routines`` as Python
dictionaries:

.. code-block:: python

    from pathlib import Path

    import pandas as pd

    from vtlengine import run

    def main():
        script = """
                    Example_5 <- DS_1 [ calc Me_2:= Id_2 in Countries];
                    Example_5_2 <- eval(SQL_3(DS_1) language "SQL" returns dataset { identifier<integer> Id_1,
                    measure<number> Me_1});
                """

        data_structures = {
            "datasets": [
                {
                    "name": "DS_1",
                    "DataStructure": [
                        {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                        {"name": "Id_2", "type": "String", "role": "Identifier", "nullable": False},
                        {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                    ],
                }
            ]
        }

        data_df = pd.DataFrame(
            {"Id_1": [2012, 2012, 2012], "Id_2": ["AT", "DE", "FR"], "Me_1": [0, 4, 9]}
        )

        datapoints = {"DS_1": data_df}

        external_routines = {
            "name": "SQL_3",
            "query": "SELECT Id_1, COUNT(*) AS Me_1 FROM DS_1 GROUP BY Id_1;",
        }


        value_domains = {
            "name": "Countries",
            "setlist": ["DE", "FR", "IT"],
            "type": "String",
        }

        run_result = run(
            script=script,
            data_structures=data_structures,
            datapoints=datapoints,
            value_domains=value_domains,
            external_routines=external_routines,
        )
        print(run_result)

Returns:

.. csv-table::
    :file: _static/Example_5.csv
    :header-rows: 1

.. csv-table::
    :file: _static/Example_5_2.csv
    :header-rows: 1


.. _example_6_run_using_paths:

Using Path objects
==================

Pass all inputs (script, data structures, datapoints, value domains, and
external routines) as ``Path`` objects. Datapoints are still keyed by
dataset name; the file itself can have any name.

.. code-block:: python

    from pathlib import Path

    import pandas as pd

    from vtlengine import run

    def main():
        filepath_external_routines = Path("docs/_static/SQL_4.json")
        filepath_ValueDomains = Path("docs/_static/VD_2.json")
        filepath_vtl_script = Path("docs/_static/Example_6.vtl")
        filepath_data_structures = Path("docs/_static/Example_6.json")
        filepath_data = Path("docs/_static/Example_6_input.csv")

        data_structures = filepath_data_structures
        datapoints = {"DS_1": filepath_data}
        script = filepath_vtl_script
        external_routines = filepath_external_routines
        value_domains = filepath_ValueDomains
        run_result = run(
            script=script,
            data_structures=data_structures,
            datapoints=datapoints,
            value_domains=value_domains,
            external_routines=external_routines,
        )
        print(run_result)

Returns:

.. csv-table::
    :file: _static/Example_6_output.csv
    :header-rows: 1

.. csv-table::
    :file: _static/Example_6_2_output.csv
    :header-rows: 1
