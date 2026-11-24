:Tests: ``cast(Me_1, string)`` on a Time measure, without a mask: the default format.
:Manual: `Type conversion <https://sdmx-twg.github.io/vtl/v2.2/reference_manual/operators/General%20purpose%20operators/Type%20conversion/index.html>`__: "From **time**: the ``time`` value is printed using the ISO-8601 representation for generic time intervals, up to a maximum of day precision, in the form start/end."
:Expected: ``2020-01-01/2020-06-30`` and ``2021-03-15/2021-03-20``, with the dates in the ISO-8601 extended form ``YYYY-MM-DD``.
