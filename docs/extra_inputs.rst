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
      — Walkthrough example using both features as dictionaries
    - :ref:`example_6_run_using_paths`
      — Walkthrough example using both features as Path objects
    - `VTL 2.1 Reference Manual
      <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf>`_
      — Full VTL specification


Value Domains
*************

A Value Domain is a named set of unique values that share
a common data type. Value domains are used with the ``in``
and ``not_in`` operators to perform membership checks
in VTL scripts.


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
      - JSON string in time format
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
- **List**: A list mixing any of the above formats.

.. code-block:: python

    from pathlib import Path

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

    # List of mixed formats
    value_domains = [
        {"name": "Countries", "type": "String", "setlist": ["DE", "FR"]},
        Path("data/extra_domains.json"),
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

**SQL file format**: A plain ``.sql`` file containing only
the query. The filename (without extension) is used as the
routine name.

.. code-block:: sql

    -- File: SQL_1.sql
    SELECT Id_1, SUM(Me_1) AS Me_1 FROM DS_1 GROUP BY Id_1;

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
- **Path to a file**: A ``Path`` pointing to a ``.json``
  or ``.sql`` file.
- **Path to a directory**: A ``Path`` pointing to a directory;
  all ``.json`` and ``.sql`` files in the directory are loaded.
- **List**: A list mixing any of the above formats.

.. code-block:: python

    from pathlib import Path

    # Single dict
    external_routines = {
        "name": "SQL_1",
        "query": "SELECT Id_1, COUNT(*) AS Me_1 FROM DS_1 GROUP BY Id_1;",
    }

    # Path to file
    external_routines = Path("data/SQL_1.json")

    # Path to directory (loads all .json and .sql files)
    external_routines = Path("data/routines/")

    # List of mixed formats
    external_routines = [
        {"name": "SQL_1", "query": "SELECT * FROM DS_1;"},
        Path("data/SQL_2.sql"),
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


Example
=======

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
