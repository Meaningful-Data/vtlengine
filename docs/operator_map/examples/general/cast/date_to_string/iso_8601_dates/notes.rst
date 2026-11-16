:Tests: ``cast(Me_1, string)`` on a Date measure, without a mask: the default format.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **date**: the ``date`` value is printed using the ISO-8601 representation for dates, with the year, month and day of month fields always present."
:Expected: The ISO-8601 extended form ``YYYY-MM-DD``: ``2020-01-15`` and ``2021-12-31``.
