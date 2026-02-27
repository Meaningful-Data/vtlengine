##########
Data Types
##########

This page documents the data types supported by vtlengine,
covering input formats, internal representation, output formats,
and type casting rules based on the
`VTL 2.2 specification <https://sdmx-twg.github.io/vtl/2.2/>`_.

.. seealso::

    - `VTL Data Types
      <https://sdmx-twg.github.io/vtl/2.2/user_manual/types.html>`_
      — Full type system in the VTL 2.2 User Manual
    - `Scalar type definitions
      <https://sdmx-twg.github.io/vtl/2.2/user_manual/types.html
      #scalar-types>`_
      — Detailed scalar type descriptions
    - `Type conversion\: cast
      <https://sdmx-twg.github.io/vtl/2.2/reference_manual/operators
      /General%20purpose%20operators/Type%20conversion/index.html>`_
      — Cast operator reference
    - `Type Conversion and Formatting Mask
      <https://sdmx-twg.github.io/vtl/2.2/reference_manual
      /typical_behaviour.html
      #type-conversion-and-formatting-mask>`_
      — Conversion rules and masks

Type Hierarchy
**************

The VTL 2.2 specification defines a hierarchy of
`scalar types
<https://sdmx-twg.github.io/vtl/2.2/user_manual/types.html
#basic-scalar-types>`_:

.. code-block:: text

    Scalar
    ├── String
    ├── Number
    │   └── Integer          (subtype of Number)
    ├── Time
    │   ├── Date             (subtype of Time)
    │   └── Time_Period      (subtype of Time)
    ├── Duration
    └── Boolean

.. note::

    In vtlengine, the VTL ``Time`` type is implemented as
    ``TimeInterval``, and ``Time_Period`` as ``TimePeriod``.
    The user-facing names remain ``Time`` and ``Time_Period``.


Data Types Reference
********************

Each type below describes how vtlengine handles input, storage,
and output. For the formal VTL definitions, see
`External representations and literals
<https://sdmx-twg.github.io/vtl/2.2/user_manual/types.html
#external-representations-and-literals-used-in-the-vtl-manuals>`_.

String
======

.. list-table::
    :widths: 30 70
    :header-rows: 0

    * - **Input (CSV)**
      - Any text value. Surrounding double quotes are
        stripped automatically.
    * - **Input (DataFrame)**
      - Any value (all values pass validation).
    * - **Internal representation**
      - Python ``str``, stored as ``string[pyarrow]``.
    * - **Output dtype**
      - ``string[pyarrow]``

Integer
=======

.. list-table::
    :widths: 30 70
    :header-rows: 0

    * - **Input (CSV)**
      - Whole numbers: ``"42"``, ``"0"``, ``"-7"``.
    * - **Input (DataFrame)**
      - Values are cast via ``str → float → int``.
        Non-integer floats (e.g. ``3.5``) are rejected.
    * - **Internal representation**
      - Python ``int``, stored as ``int64[pyarrow]``.
    * - **Output dtype**
      - ``int64[pyarrow]``

Integer is a **subtype of Number** — anywhere a Number is
expected, an Integer is accepted automatically.

Number
======

.. list-table::
    :widths: 30 70
    :header-rows: 0

    * - **Input (CSV)**
      - Decimal or integer numbers: ``"3.14"``, ``"1e5"``,
        ``"42"``.
    * - **Input (DataFrame)**
      - Values are cast via ``str → float``.
    * - **Internal representation**
      - Python ``float``, stored as ``double[pyarrow]``.
    * - **Output dtype**
      - ``double[pyarrow]``

Boolean
=======

.. list-table::
    :widths: 30 70
    :header-rows: 0

    * - **Input (CSV)**
      - ``"true"``, ``"false"`` (case-insensitive),
        ``"1"``, ``"0"``.
    * - **Input (DataFrame)**
      - Same string values or native Python
        ``bool``/``int``/``float``.
    * - **Internal representation**
      - Python ``bool``, stored as ``bool[pyarrow]``.
    * - **Output dtype**
      - ``bool[pyarrow]``

Date
====

.. list-table::
    :widths: 30 70
    :header-rows: 0

    * - **Input (CSV)**
      - ISO 8601 date: ``"2020-01-15"``,
        ``"2020-01-15 10:30:00"``,
        ``"2020-01-15T10:30:00"``.
        Nanosecond precision is truncated to
        microseconds. Year range: 1800–9999.
    * - **Input (DataFrame)**
      - String values validated against the same
        ISO 8601 formats.
    * - **Internal representation**
      - Python ``str`` in ``"YYYY-MM-DD"`` or
        ``"YYYY-MM-DD HH:MM:SS"`` format,
        stored as ``string[pyarrow]``.
    * - **Output dtype**
      - ``string[pyarrow]``

Date is a **subtype of Time** — anywhere a Time value is
expected, a Date is accepted automatically.

Time_Period
===========

.. list-table::
    :widths: 30 70
    :header-rows: 0

    * - **Input (CSV/DataFrame)**
      - Multiple formats accepted (see tables below).
    * - **Internal representation**
      - Hyphenated string (e.g. ``"2020-M01"``,
        ``"2020-Q1"``), stored as ``string[pyarrow]``.
    * - **Output dtype**
      - ``string[pyarrow]`` — format controlled by
        ``time_period_output_format``.

**Accepted input formats:**

.. list-table::
    :widths: 15 45 40
    :header-rows: 1

    * - Period
      - Formats
      - Examples
    * - Annual
      - ``YYYY``, ``YYYYA``, ``YYYY-A1``
      - ``2020``, ``2020A``, ``2020-A1``
    * - Semester
      - ``YYYYSx``, ``YYYY-Sx``
      - ``2020S1``, ``2020-S1``
    * - Quarter
      - ``YYYYQx``, ``YYYY-Qx``
      - ``2020Q1``, ``2020-Q1``
    * - Monthly
      - ``YYYYMm``, ``YYYYMmm``, ``YYYY-MM``,
        ``YYYY-M``, ``YYYY-Mxx``, ``YYYY-Mx``
      - ``2020M1``, ``2020M01``, ``2020-01``,
        ``2020-1``, ``2020-M01``, ``2020-M1``
    * - Weekly
      - ``YYYYWw``, ``YYYYWww``, ``YYYY-Wxx``
      - ``2020W1``, ``2020W01``, ``2020-W01``
    * - Daily
      - ``YYYYD[dd]d``, ``YYYY-D[xx]x``, ``YYYY-MM-DD``
      - ``2020D1``, ``2020D01``, ``2020D001``,
        ``2020D-1``, ``2020D-01``, ``2020D-001``,
        ``2020-01-01``

**Output formats** (controlled by ``time_period_output_format``
parameter):

.. list-table::
    :widths: 18 14 14 14 14 13 13
    :header-rows: 1

    * - Format
      - Annual
      - Semester
      - Quarter
      - Month
      - Week
      - Day
    * - ``"vtl"`` (default)
      - ``2020``
      - ``2020S1``
      - ``2020Q1``
      - ``2020M1``
      - ``2020W15``
      - ``2020D100``
    * - ``"sdmx_reporting"``
      - ``2020-A1``
      - ``2020-S1``
      - ``2020-Q1``
      - ``2020-M01``
      - ``2020-W15``
      - ``2020-D100``
    * - ``"sdmx_gregorian"``
      - ``2020``
      - Not supported
      - Not supported
      - ``2020-01``
      - Not supported
      - ``2020-01-15``
    * - ``"legacy"``
      - ``2020``
      - ``2020-S1``
      - ``2020-Q1``
      - ``2020-M01``
      - ``2020-W15``
      - ``2020-01-15``

Time_Period is a **subtype of Time** — anywhere a Time value
is expected, a Time_Period is accepted automatically.

Time (TimeInterval)
===================

.. list-table::
    :widths: 30 70
    :header-rows: 0

    * - **Input (CSV/DataFrame)**
      - ISO 8601 interval: ``"2020-01-01/2020-12-31"``.
        Also accepts ``"YYYY"`` (expanded to full year
        interval) and ``"YYYY-MM"`` (expanded to full
        month interval).
    * - **Internal representation**
      - Python ``str`` in ``"YYYY-MM-DD/YYYY-MM-DD"``
        format, stored as ``string[pyarrow]``.
    * - **Output dtype**
      - ``string[pyarrow]``

Duration
========

.. list-table::
    :widths: 30 70
    :header-rows: 0

    * - **Input (CSV/DataFrame)**
      - Single-letter period indicator: ``"A"`` (annual),
        ``"S"`` (semester), ``"Q"`` (quarter),
        ``"M"`` (month), ``"W"`` (week), ``"D"`` (day).
    * - **Internal representation**
      - Python ``str`` (single letter),
        stored as ``string[pyarrow]``.
    * - **Output dtype**
      - ``string[pyarrow]``


Null Handling
*************

All VTL scalar types support ``null`` values (represented as
``pd.NA`` / ``None``), with one exception:

- **Identifiers** cannot be null — loading data with null
  identifiers raises an error.
- **Measures** and **Attributes** can be nullable (controlled
  by the ``nullable`` flag in the data structure definition).

During operations, ``null`` propagates: any operation involving
a ``null`` operand typically produces a ``null`` result.
The ``Null`` type is compatible with all other types for
implicit promotion.


Type Casting
************

Implicit Casting (Automatic)
============================

Implicit casts happen automatically when operators receive
operands of different but compatible types.
The engine resolves the common type using the
`type promotion rules
<https://sdmx-twg.github.io/vtl/2.2/reference_manual
/typical_behaviour.html#operators-changing-the-data-type>`_
defined in VTL 2.2.

.. list-table::
    :widths: 14 10 10 10 10 10 10 12 10
    :header-rows: 1

    * - From / To
      - String
      - Number
      - Integer
      - Boolean
      - Time
      - Date
      - Time_Period
      - Duration
    * - **String**
      - |y|
      - —
      - —
      - —
      - —
      - —
      - —
      - —
    * - **Number**
      - —
      - |y|
      - |y|
      - —
      - —
      - —
      - —
      - —
    * - **Integer**
      - —
      - |y|
      - |y|
      - —
      - —
      - —
      - —
      - —
    * - **Boolean**
      - |y|
      - —
      - —
      - |y|
      - —
      - —
      - —
      - —
    * - **Time**
      - —
      - —
      - —
      - —
      - |y|
      - —
      - —
      - —
    * - **Date**
      - —
      - —
      - —
      - —
      - |y|
      - |y|
      - —
      - —
    * - **Time_Period**
      - —
      - —
      - —
      - —
      - |y|
      - —
      - |y|
      - —
    * - **Duration**
      - —
      - —
      - —
      - —
      - —
      - —
      - —
      - |y|

Key rules:

- **Integer / Number**: Both directions are implicit
  (Integer is a subtype of Number).
- **Date to Time**: A Date is implicitly converted to a
  Time interval
  (``"2020-01-15"`` becomes ``"2020-01-15/2020-01-15"``).
- **Time_Period to Time**: A Time_Period is implicitly
  converted to a Time interval
  (``"2020-Q1"`` becomes ``"2020-01-01/2020-03-31"``).
- **Boolean to String**: ``true`` becomes ``"True"``,
  ``false`` becomes ``"False"``.
- **Null to any type**: Null is compatible with every type.


Explicit Casting (cast operator)
================================

The `cast
<https://sdmx-twg.github.io/vtl/2.2/reference_manual/operators
/General%20purpose%20operators/Type%20conversion/index.html>`_
operator converts values from one type to another:

.. code-block::

    /* Without mask */
    DS_r <- cast(DS_1, integer);

    /* With mask */
    DS_r <- cast(DS_1, date, MASK);

.. note::

    VTL type names in the ``cast`` operator are lowercase:
    ``string``, ``integer``, ``number``, ``boolean``,
    ``time``, ``date``, ``time_period``, ``duration``.

Supported conversions without mask
-----------------------------------

.. list-table::
    :widths: 14 10 10 10 10 10 10 12 10
    :header-rows: 1

    * - From / To
      - String
      - Number
      - Integer
      - Boolean
      - Time
      - Date
      - Time_Period
      - Duration
    * - **String**
      - |y|
      - |y|
      - |y|
      - —
      - |y|
      - |y|
      - |y|
      - |y|
    * - **Number**
      - |y|
      - |y|
      - |y|
      - |y|
      - —
      - —
      - —
      - —
    * - **Integer**
      - |y|
      - |y|
      - |y|
      - |y|
      - —
      - —
      - —
      - —
    * - **Boolean**
      - |y|
      - |y|
      - |y|
      - |y|
      - —
      - —
      - —
      - —
    * - **Time**
      - |y|
      - —
      - —
      - —
      - |y|
      - —
      - —
      - —
    * - **Date**
      - |y|
      - —
      - —
      - —
      - —
      - |y|
      - |y|
      - —
    * - **Time_Period**
      - |y|
      - —
      - —
      - —
      - —
      - —
      - |y|
      - —
    * - **Duration**
      - |y|
      - —
      - —
      - —
      - —
      - —
      - —
      - |y|

Conversion details:

- **Number/Integer to Boolean**: ``0`` becomes ``false``,
  any other value becomes ``true``.
- **Boolean to Number/Integer**: ``true`` becomes ``1``
  (or ``1.0``), ``false`` becomes ``0`` (or ``0.0``).
- **String to Integer**: Must be a valid integer string
  (rejects ``"3.5"``).
- **Date to Time_Period**: Converts to daily period
  (e.g. ``"2020-01-15"`` becomes ``"2020D15"``
  with the default ``vtl`` output format).

Supported conversions with mask
-------------------------------

.. list-table::
    :widths: 18 14 14 14 14 14 14
    :header-rows: 1

    * - From / To
      - String
      - Number
      - Time
      - Date
      - Time_Period
      - Duration
    * - **String**
      - —
      - |p|
      - |p|
      - |p|
      - |p|
      - |p|
    * - **Time**
      - |p|
      - —
      - —
      - —
      - —
      - —
    * - **Date**
      - |p|
      - —
      - —
      - —
      - —
      - —
    * - **Time_Period**
      - —
      - —
      - —
      - |p|
      - —
      - —
    * - **Duration**
      - |p|
      - —
      - —
      - —
      - —
      - —

Legend: |y| = implemented, |p| = defined in VTL 2.2 but not
yet implemented (raises ``NotImplementedError``).

.. |y| unicode:: U+2705
.. |p| unicode:: U+23F3

.. note::

    Formal definition of masks is still to be decided.


Cast on datasets
----------------

When ``cast`` is applied to a Dataset, it must have exactly
**one measure** (monomeasure). The measure is renamed to a
generic name based on the target type:

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Target type
      - Renamed measure
    * - String
      - ``str_var``
    * - Number
      - ``num_var``
    * - Integer
      - ``int_var``
    * - Boolean
      - ``bool_var``
    * - Time
      - ``time_var``
    * - Time_Period
      - ``time_period_var``
    * - Date
      - ``date_var``
    * - Duration
      - ``duration_var``

.. note::

    When the source type can be implicitly promoted to the
    target type (e.g. Boolean to String, Integer to Number,
    or Number to Integer), the measure is **not** renamed.
