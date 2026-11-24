:Tests: ``cast(Me_1, date)`` on a String measure, without a mask: the default parsing.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__, "To **date**": "exactly four digits are consumed to determine the year", then a ``-``, "exactly two digits are consumed to determine the month in the range 01-12", a ``-`` again, and "exactly two digits are consumed to determine the day of the month".
:Expected: ``2020-01-15`` and ``2021-12-31``. The inputs have no blanks for the conversion to remove.
