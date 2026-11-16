:Tests: ``cast(Me_1, time)`` on a String measure, without a mask: ISO-8601 intervals.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "To **time**: the input string is parsed, according to the ISO-8601 standard for time intervals, up to a precision of days, and the corresponding ``time`` value is returned."
:Expected: ``2020-01-01/2020-06-30`` and ``2021-03-15/2021-03-20``, a ``time`` written ``start/end``, as vtlengine reads it. The inputs have no blanks for the conversion to remove.
