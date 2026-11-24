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
    - `VTL 2.2 Reference Manual
      <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/index.html>`_
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
      - JSON string with a period indicator (``"A"``, ``"S"``,
        ``"Q"``, ``"M"``, ``"W"`` or ``"D"``)

Each value of the ``setlist`` must be a valid value of the type
(``0-1-1-37``): ``"2024-13-45"`` is not a ``Date`` and
``"2024Q9"`` is not a ``Time_Period``. The values must be
unique, also when written in another form of the type
(``0-1-1-21``): ``"2024Q1"`` and ``"2024-Q1"`` are the same
``Time_Period``.

The names of the value domains given must be unique
(``0-1-1-38``), and cannot be a basic scalar type of VTL:
``string``, ``integer``, ``number``, ``boolean``, ``date``,
``time``, ``time_period``, ``duration`` or ``scalar``
(``0-1-1-39``).


Input Options
=============

The ``value_domains`` parameter accepts the following formats:

- **Dictionary**: A single value domain as a Python dict.
- **Path to a JSON file**: A ``Path`` pointing to a ``.json``
  file containing one value domain definition, or a JSON array of
  them.
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

A ``.json`` file that is not valid JSON raises ``0-1-1-40``, naming
the file. ``ValueDomain.to_dict()`` and ``ValueDomain.to_json()``
write the definition in this format, so it can be given back.


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

A value is compared with the values of a value domain as a value of
its type, not as the text it is written in: a ``Time_Period``
``2024Q1`` or ``2024-Q1`` is in a value domain whose ``setlist``
holds ``"2024-Q1"`` or ``"2024Q1"``, and a ``Date`` ``2024-1-5`` in
one holding ``"2024-01-05"``. The same holds wherever a value is
checked against a value domain: ``in`` and ``not_in``, the values of
a Component or a scalar on a value domain, ``cast`` into a value
domain and the parameters of a user defined operator typed by one.
The elements of a literal set are compared in the same way, as values
of their type: ``Me_1 in {cast("2024-Q2", time_period)}`` is true for
the ``Time_Period`` ``2024Q2``, as ``Me_1 = cast("2024-Q2",
time_period)`` is, and two elements that are one value written in two
forms (``cast("2024Q1", time_period)`` and ``cast("2024-Q1",
time_period)``) are a repeated value, which raises ``1-2-27``.


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
domain given raises ``0-2-1-1``, and so does the legacy ``subset``
key of a Component or a scalar, which is not accepted: the value
domain is named in the ``type``. Loading the datapoints checks
that every value of the Component is a value of its value
domain, and raises ``0-3-1-6`` on another one. Nulls follow
``nullable``, as with a basic type.

The operators handle the Component as one of its basic type:
the same type checks and the same result types. ``Id_2 || "_X"``
is a ``String``, and ``Me_1 + 1`` on a value domain of integers
is an ``Integer``. The Components that an operator keeps as they
are keep their value domain: the Identifiers (except the ones
below), the Components that ``filter``, ``keep``, ``drop``,
``rename``, ``sub`` and the joins keep, the Measures that
``flow_to_stock`` and ``stock_to_flow`` pass through (all but the
``Integer`` and ``Number`` ones, which they accumulate), and a
Component written alone in ``calc`` (``calc Me_3 := Id_2``). The
operators that give one of the values of their operand keep its
value domain too: ``max`` and ``min``, in an aggregation and as
analytic operators, ``first_value``, ``last_value``, and ``lag`` and
``lead`` without a default value. With a default value, ``lag`` and
``lead`` give the basic type, as the default is a literal of it and
``nvl`` gives that mix the basic type.
``run()`` and ``semantic_analysis()`` write ``"type": "VD_FLOW"``
for them, and ``Dataset.from_json``, ``Component.from_json``,
``DataComponent.from_json`` and ``Scalar.from_json`` read such a type
back on the value domain given in their ``value_domains`` parameter,
a dict by name (``{"VD_FLOW": vd_flow}``); a type that is neither a
basic type nor one of them raises ``0-2-1-1``. The data structures
given to ``vtlengine.duckdb_transpiler.transpile()`` can name the
value domains given to it as well. A join ``apply`` gives each Measure the type and the value
domain its expression gives, as ``calc`` does: ``apply nvl(a, b)``
follows the rules of ``nvl`` below, and ``apply a + b`` gives the
basic type.

A Component into which an operator writes new values has its basic
type, as those values are not values of the value domain: the time
Identifier of ``timeshift``, ``fill_time_series`` and an aggregation
grouped with ``time_agg``, the time Measure of ``time_agg`` on a
Data Set, the Components that ``dateadd`` shifts, the Measures of
``hierarchy``, the Measures of ``check_hierarchy`` in
``non_zero``, ``partial_zero`` and ``always_zero``, which write a
``0`` for a missing code item, and the ``Integer`` and ``Number``
Measures of ``flow_to_stock`` and ``stock_to_flow``.

The rule Identifier of ``hierarchy`` and ``check_hierarchy`` keeps
the value domain a ruleset is defined on (``valuedomain rule VD``).
For a ruleset on a variable, it keeps the value domain of the
Component when every code item that the rules compute (for
``check_hierarchy``, check) is a value of it, and has its basic type
otherwise: with ``C = A + B`` on a value domain of ``A`` and ``B``,
the result holds ``C`` and the rule Identifier has its basic type.

The operators that put the values of several operands into one
Component (the set operators, the Identifiers that a join
matches, ``nvl``, ``if-then-else`` and ``case``, and the new
Measure of ``unpivot``, which holds the values of the unpivoted
Measures) follow the VTL type conversion rules:

- Operands on the same value domain, nulls aside, give that value
  domain.
- Any other mix gives the promotion of the basic types, as a value
  domain is always converted into its basic type (upcast): a value
  domain and its basic type, or two value domains on the same
  basic type, give that basic type, and a value domain of integers
  and a ``Number`` give a ``Number``.

.. code-block:: text

    /* Id_2 is on VD_FLOW in DS_1 and DS_2: it keeps VD_FLOW */
    DS_r <- union(DS_1, DS_2);

    /* Me_1 is on a value domain of integers: Me_2 is an Integer */
    DS_r <- DS_1 [calc Me_2 := nvl(Me_1, 0)];

    /* Me_1 and Me_2 are on VD_FLOW: Me_3 keeps VD_FLOW (Id_3 is a String) */
    DS_r <- DS_1 [unpivot Id_3, Me_3];

The Identifiers of the conditions and of the Data Set operands of
``if-then-else`` and ``case`` match on their names and basic types,
as in ``nvl`` and the set operators: an Identifier on a value domain
matches one of its basic type, and the result Identifier follows the
rules above, the conditions included. Identifiers of different basic
types raise ``1-1-9-10``.

``cast`` converts a value into a value domain in two steps: it
casts the value to the basic type of the value domain, as a cast
to that type does, and the result must be one of the values of the
value domain (``2-1-5-4``):

.. code-block:: text

    /* Me_1 is a String: Me_2 is on VD_INT, a value domain of integers,
       and every value of Me_1 cast to integer must be one of its values */
    DS_r <- DS_1 [calc Me_2 := cast(Me_1, VD_INT)];

``pivot`` needs the Identifier it pivots defined on a value domain:
the result has one Measure per value of the value domain, named after
the value as a cast to ``string`` writes it, whatever form the
``setlist`` writes it in (``true`` and ``false`` for a ``Boolean``
value domain, so ``[keep 'true']`` keeps the first one; ``2024Q1``
for a ``Time_Period`` written ``"2024-Q1"``; ``2024-01-05`` for a
``Date`` written ``"2024-1-5"``). A ``Duration`` value domain names
them as its ``setlist`` writes the values (``A``, ``Q``), not as the
cast does (``P1Y``, ``P3M``). The Measures are of the type (and on
the value domain) of the pivoted Measure. The other Identifiers group
the data points, a value with no data point in a group leaves its Measure null, the other Measures and
the non-viral Attributes are dropped, and the viral Attributes of a
group are combined by their propagation rule. An Identifier on no
value domain raises ``1-1-6-18``.

.. code-block:: text

    /* Id_2 is on a value domain of A and B: DS_r has the Measures A and B */
    DS_r <- DS_1 [pivot Id_2, Me_1];

Value domains also appear in the VTL-DL definitions:

- A datapoint ruleset on value domains binds them, or their aliases,
  in order to the components listed in ``check_datapoint ...
  components``, which is required (``1-1-10-14``), with one
  component per value domain (``1-1-10-15``). Each component must be
  defined on the value domain it is bound to, or be of a basic type
  that the implicit promotion takes to the type of the value domain
  (a ``String`` for a ``String`` value domain, an ``Integer`` for a
  ``Number`` one), and then it keeps its basic type; a component on
  another value domain, or of a type that does not promote, raises
  ``1-1-10-12``. The value domains must be given (``1-2-22``,
  ``1-2-8``).
- A hierarchical ruleset on value domains binds the value domains of
  its condition, in order, to the components listed in ``condition``
  (one per value domain, ``1-1-10-16``), and the value domain of its
  rule to the component in ``rule`` (required, ``1-1-10-4``), in
  ``check_hierarchy`` and ``hierarchy``. The same checks apply: each
  component must be defined on its value domain or be of a basic type
  that promotes to its type (``1-1-10-12``), and the value domains
  must be given (``1-2-22``, ``1-2-8``). Every code
  item its rules name, on either side, must be a value of the value
  domain of its rule, compared as a value of its type (``1-1-10-13``),
  so its results are values of it.
- A viral propagation rule on a value domain needs it given
  (``1-2-8``), and the constants of its ``when``, ``then`` and
  ``else`` must be values of it (``1-1-10-13``); the definition
  checks both, also when no statement uses the rule. The constants
  are values of the type of the value domain, whatever form they are
  written in: ``when "2024Q1"`` matches the ``Time_Period``
  ``2024-Q1``, and ``then "2024-1-6"`` gives the ``Date``
  ``2024-01-06``. A rule on a variable reads its constants as values
  of the type of the Attribute it applies to in the same way, and two
  ``when`` on the same values written in two forms raise
  ``1-3-3-4``. The rule
  applies to the viral attributes defined on the value domain,
  unless a rule on the variable overrides
  it. Where a viral attribute is combined, the rule on its value
  domain applies only when it is on that value domain in every
  operand: when an operand gives it on no value domain or on another
  one, the mix is promoted to the basic type, only a rule on the
  variable applies, and without one the statement raises
  ``1-3-3-6``, in either operand order (binary operators, joins,
  ``if-then-else`` and ``case``). A rule on a value domain with
  ``aggregate sum`` or ``avg`` gives a result of the basic type
  (``avg`` a ``Number``), which is no longer on the value domain, so
  a further combination of that result needs a rule on the variable
  (otherwise ``1-3-3-6``). With ``aggregate max`` or ``min`` the
  result keeps the value domain, as it is one of the values combined.
- A value domain types a scalar or a component parameter of a user
  defined operator (``x VD_FLOW``, ``x component<VD_FLOW>``): the
  argument is of the basic type of the value domain, and its values
  must be values of it (``2-1-5-4``). A literal argument of a scalar
  parameter, known before the script runs, must be one of its values
  as its default must (``1-1-10-13``, also in ``semantic_analysis``):
  ``f("IT")`` on ``x VD_CTY`` raises it, as ``cast("IT", VD_CTY)``
  raises ``1-1-5-6``. In the body a scalar parameter
  is defined on the value domain, and a component parameter behaves
  as its basic type. A value of a type promoted to the basic type is
  compared as the value it is promoted to, here and in the forms
  below: an ``Integer`` as a ``Number``, a ``Boolean`` as the
  ``String`` ``true`` or ``false``.
- A value domain types the values of a ``set`` parameter
  (``s set<VD_FLOW>``): the argument is a value domain of its basic
  type (another one raises ``1-2-18``, as ``set<string>`` does), and
  its values must be values of it (``2-1-5-4``): ``VD_FLOW`` itself
  or a value domain whose values are all in ``VD_FLOW``. In the body
  the parameter holds the values of the argument, wherever a set can
  be used (``Me_1 in s``). ``set<VD_FLOW not null>`` raises
  ``1-3-13``, as ``set<string not null>`` does.
- A value domain types the components of a ``dataset`` parameter,
  with any role and with a component name or a multiplicity modifier
  (``dataset {identifier<VD_FLOW> Id_2}``,
  ``dataset {measure<VD_FLOW> _+}``): the constraint takes the
  Components of the basic type of the value domain, as
  ``x component<VD_FLOW>`` does, and their values must be values of
  it (``2-1-5-4``). A Component of another basic type raises what
  the basic type raises (``1-3-1-12``, ``1-3-1-13``, ``1-3-1-14``).
  The argument is passed as it is: its Components keep their types
  and value domains.
- A value domain types the result of a user defined operator
  (``returns VD_FLOW``, ``returns component<VD_FLOW>``,
  ``returns measure<VD_FLOW>``,
  ``returns dataset {measure<VD_FLOW> Me_1}``): the result, or the
  Components of a Data Set result that the type names on a value
  domain, must be of its basic type (otherwise what the basic type
  raises: ``1-3-1-1``, or ``1-3-1-11``, ``1-3-1-12``, ``1-3-1-14``
  for a Data Set), their values must be values of it (``2-1-5-4``),
  and they are then defined on it. The other Components keep their
  types. A ``returns`` of a basic type keeps the value domain of a
  result of that type already on one. A result of another type is
  converted to the type the ``returns`` declares where the VTL implicit
  cast applies (an ``Integer`` to a ``Number``, a ``Boolean`` to the
  ``String`` ``true`` or ``false``), for a scalar, a Component and the
  Components a ``dataset`` type names, and is then on no value domain.
  A ``Number`` is not cast to ``integer``: a scalar or Component
  ``Number`` result of a ``returns integer`` raises ``1-3-1-1``, as a
  result of any other type does.

.. code-block:: text

    define datapoint ruleset DPR_1 (valuedomain VD_FLOW as F, VD_AMOUNT as A) is
        when F = "CREDIT" then A >= 0
    end datapoint ruleset;

    define hierarchical ruleset HR_1 (valuedomain condition VD_YEAR as Y rule VD_CTY) is
        when Y >= 2020 then EU = ES + FR + DE
    end hierarchical ruleset;

    define viral propagation VP_1 (valuedomain CL_CONF) is
        when "C" then "C";
        else "F"
    end viral propagation;

    define operator with_suffix (x component<VD_FLOW>)
        returns component<string> is
            x || "_X"
    end operator;

    /* Keeps the data points whose Me_1 is one of the values of s */
    define operator in_flows (ds dataset {measure<VD_FLOW> Me_1}, s set<VD_FLOW>)
        returns dataset is
            ds [filter Me_1 in s]
    end operator;

    /* Me_2 is on VD_FLOW: each value of Me_3 must be one of its values */
    define operator as_flow (x component)
        returns measure<VD_FLOW> is
            x
    end operator;

    DS_r <- check_datapoint(DS_1, DPR_1 components Id_2, Me_1 all);
    DS_h <- check_hierarchy(DS_2, HR_1 condition Id_2 rule Id_1 all);
    DS_s <- DS_1 [calc Me_2 := with_suffix(Id_2)];
    DS_t <- in_flows(DS_2, VD_FLOW);
    DS_u <- DS_1 [calc Me_2 := as_flow(Me_3)];

The value domains ``errorcode_vd`` and ``errorlevel_vd``, when given,
type the error codes and the error levels of ``check``,
``check_datapoint`` and ``check_hierarchy``: the ``errorcode`` and
``errorlevel`` Measures take their basic types, and every error code
and level of a rule or of ``check`` must be one of their values
(``1-1-10-13``). A literal of another type is not converted and
raises the same error: ``errorcode "100"`` on an ``Integer``
``errorcode_vd``, ``errorlevel 1`` on a ``String`` ``errorlevel_vd``
or ``errorlevel 1.0`` on an ``Integer`` one (an ``Integer`` literal is
a value of a ``Number`` value domain). Without them, the Measures
keep their usual types.

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
  rules above: a scalar on a value domain and a value of its basic
  type, or two different value domains, give the basic type.
- ``cast(SC_2, VD_FLOW)`` gives a scalar on ``VD_FLOW``.
- An operator that computes a new value gives its basic type:
  ``SC_1 || "_X"`` is a ``String``.
- A scalar parameter of a user defined operator typed by a value
  domain is defined on it. One of a basic type keeps the value domain
  of an argument of that same type.
- The result of a user defined operator that ``returns`` a value
  domain is defined on it, and must be one of its values
  (``2-1-5-4``).

``run()`` and ``semantic_analysis()`` write ``"type": "VD_FLOW"`` for
the scalar results on a value domain.

.. code-block:: text

    /* SC_r is a String: "OTHER" is a value of the basic type of VD_FLOW */
    SC_r <- nvl(SC_1, "OTHER");

    /* Me_2 is on VD_FLOW */
    DS_r <- DS_1 [calc Me_2 := SC_1];

    /* SC_r is on VD_FLOW: "CREDIT" is one of its values */
    define operator flow (x string) returns VD_FLOW is x end operator;
    SC_r <- flow("CREDIT");


Validation
==========

Use :meth:`vtlengine.validate_value_domain` to validate
value domains before execution: their JSON structure, the values
of their ``setlist`` and their names, as described in
`Supported Types`_. ``run()`` and ``semantic_analysis()`` check
the same when they load the value domains, also the ones that the
script does not use:

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
      - SQL query to execute, in DuckDB SQL. A double-quoted
        name is an identifier when it names a component or an
        operand, or the query defines it (a column or table
        alias, a ``WITH`` name or column), ignoring case as
        DuckDB does: ``"Me_1"`` is the column ``Me_1``. Any
        other double-quoted token is a text, as between single
        quotes: ``"N/A"`` is ``'N/A'``. Table names in the query
        must match the dataset names passed as operands; the
        names a ``WITH`` clause defines are not operands.

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
- **Path to a file**: A ``Path`` pointing to a ``.json`` file
  holding one routine, or a JSON array of them. Each routine is
  used under its ``name``, whatever the name of the file.
- **Path to a directory**: A ``Path`` pointing to a directory;
  all ``.json`` files in the directory are loaded.
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

    # Path to directory (loads all .json files)
    external_routines = Path("data/routines/")

    # List of mixed formats
    external_routines = [
        {"name": "SQL_1", "query": "SELECT * FROM DS_1;"},
        Path("data/SQL_2.json"),
    ]

A ``.json`` file that is not valid JSON raises ``0-1-1-40``, naming
the file.

The names of the routines given must be unique, whether they are
given in a list, in one JSON array or in the files of a directory:
two routines of one name raise ``0-1-1-41``.


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

The values of the result are read as values of the types the
``returns dataset`` clause declares, as datapoints are: a value
that is not one raises ``2-1-5-1`` (``2-1-5-2`` for a number with
decimals given to an ``integer``), a null Identifier raises
``2-1-1-16``, and Identifiers repeated in the result raise
``2-1-15-4``.


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
